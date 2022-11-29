# stdlib
import sys
import copy
from typing import Any, List, Optional, Union
from abc import abstractmethod

# third party
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# interpretability relative
from .utils import data
from .base import Explainer, FeatureExplanation

# Interpretability absolute
from interpretability.utils.pip import install
from interpretability.exceptions import exceptions

# dynamask
for retry in range(2):
    try:
        # third party
        import dynamask

        break
    except ImportError:
        depends = ["dynamask"]
        install(depends)

from dynamask.attribution import mask, mask_group, perturbation
from dynamask.utils import losses


class DynamaskExplainer(Explainer):
    def __init__(
        self,
        model: Any,
        perturbation_method: str = "gaussian_blur",
        group: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Initialises the mask.

        Args:
            model (Any): The model to explain. Must be a trained pytorch model.
            perturbation_method (str): The method to create and apply perturbation on inputs based on masks. Defaults to "gaussian_blur".
            group (bool): Boolean value to select whether or not to use a MaskGroup. MaskGroups allow fitting several masks of different areas simultaneously. Defaults to False.
            device (str,): The device to send torch.tensors. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
        """
        self.DEVICE = device
        model = model.train()
        model = model.to(self.DEVICE)

        def f(x):
            x = x.unsqueeze(0)
            out = model(x).float()
            return out

        self.model = f
        available_perturbation_methods = {
            "fade_moving_average": perturbation.FadeMovingAverage,
            "gaussian_blur": perturbation.GaussianBlur,
            "fade_moving_average_window": perturbation.FadeMovingAverageWindow,
            "fade_moving_average_past_window": perturbation.FadeMovingAveragePastWindow,
            "fade_reference": perturbation.FadeReference,
        }
        self.perturbation_method = available_perturbation_methods[perturbation_method]
        self.perturbation = None
        self.perturbation_baseline = None
        self.group = group
        self.mask_class = mask_group.MaskGroup if group else mask.Mask
        self.mask = None
        self.loss_function = None
        self.all_data = None
        self.explain_data = None
        self.explain_target = None
        super().__init__()

    def fit(
        self,
        explain_id: int,
        X: Optional[np.array] = None,
        loss_function: str = "mse",
        target: Optional[np.array] = None,
        baseline: Optional[torch.tensor] = None,
        area_list: Union[np.array, List] = np.arange(0.001, 0.051, 0.001),
    ):
        """
        Trains the mask.

        Args:
            X (np.array, optional): The data to be explained.
            loss_function (str): The name of the loss function to use, e.g. "cross_entropy", "log_loss", "log_loss_target", or "mse" Defaults to "mse".. Defaults to "mse".
            target (np.array, optional): The target for the data being explained. Defaults to None. If none provided targets are generated from the blackbox model.
            baseline (torch.tensor, optional): A baseline for the perturbation method. Only required for fade_reference. Defaults to None.
            area_list (Union[np.array, List]): List of areas for the group mask. Defaults to np.arange(0.001, 0.051, 0.001).

        Returns:
            None
            pd.DataFrame: The importance dataframe. This is of shape time_steps x features that contains the calculated importance values.
        """
        if X is not None:
            self.all_data = X
        else:
            if self.all_data is None:
                raise exceptions.NoDataToExplain
        if target is not None:
            self.target = torch.tensor(target).to(self.DEVICE).detach()
        self.explain_data = (
            torch.tensor(self.all_data[explain_id]).float().to(self.DEVICE)
        )

        available_loss_functions = {
            "cross_entropy": losses.cross_entropy,
            "log_loss": losses.log_loss,
            "log_loss_target": losses.log_loss_target,
            "mse": losses.mse,
        }
        self.loss_function = available_loss_functions[loss_function]
        # Fit a mask to the input with a Gaussian Blur perturbation:
        if self.perturbation_method == perturbation.FadeReference:
            if baseline:
                self.perturbation_baseline = baseline
            else:
                self.perturbation_baseline = torch.zeros(size=self.explain_data.shape)
            self.perturbation = self.perturbation_method(
                self.DEVICE, self.perturbation_baseline
            )
        else:
            self.perturbation = self.perturbation_method(self.DEVICE)
        self.mask = self.mask_class(self.perturbation, self.DEVICE)

        print("Fitting Dynamask")
        if self.group:
            self.mask.fit(
                self.explain_data,
                self.model,
                area_list,
                loss_function=self.loss_function,
                n_epoch=1000,
                initial_mask_coeff=0.5,
                size_reg_factor_init=0.1,
                size_reg_factor_dilation=100,
                learning_rate=0.1,
                momentum=0.9,
                time_reg_factor=0,
            )
        else:
            self.mask.fit(
                self.explain_data,
                self.model,
                loss_function=self.loss_function,
                target=self.target,
                n_epoch=500,
                keep_ratio=0.5,
                initial_mask_coeff=0.5,
                size_reg_factor_init=0.5,
                size_reg_factor_dilation=100,
                time_reg_factor=0,
                learning_rate=1.0e-1,
                momentum=0.9,
            )
        self.has_been_fit = True

    def refit(self, explain_id: int):
        """A Helper function to fit the model again with the same parameters but for a different data record.

        Args:
            explain_id (int): The id of the record to get the explanation for by refitting
        """
        print("Re-fitting dynamask")
        self.fit(explain_id)

    def explain(
        self,
        ids_time: Union[List, np.array] = None,
        ids_feature: Union[List, np.array] = None,
        smooth: bool = False,
        sigma: float = 1.0,
        get_mask_from_group_method: str = "best",
        extremal_mask_threshold: float = 0.01,
    ) -> FeatureExplanation:
        """
        Get the explanation from the trained mask.

        Args:
            ids_time (Union[list, np.array], optional): A list of time steps to focus to explanation on. Defaults to None leading to all time steps being included in the explanation.
            ids_feature (Union[list, np.array], optional):  A list of features to focus to explanation on. Defaults to None leading to all features being included in the explanation.
            smooth (bool, optional): A boolean value to state weather or not to smooth the mask (i.e. interpolate between extreme values to provide a smooth transition in the time dimention). Defaults to False.
            sigma (float, optional): Width of the smoothing Gaussian kernel.. Defaults to 1.0.
            get_mask_from_group_method (str, optional): Can take values of "best" or "extremal". "best" returns the mask with lowest error. "extremal" returns the extremal mask for the acceptable error threshold. Defaults to "best".
            extremal_mask_threshold (float, optional): The acceptable error threshold for extremal masks. Defaults to 0.01.

        Returns:
            FeatureExplanation: A simple feature importance pd.dataframe where columns refer to the time steps and rows refer to the features.
        """
        if self.has_been_fit:
            if self.group:
                available_get_mask_from_group_method = {
                    "best": self.mask.get_best_mask(),
                    "extremal": self.mask.get_extremal_mask(extremal_mask_threshold),
                }
                mask_tensor_list = available_get_mask_from_group_method[
                    get_mask_from_group_method
                ]
                submask_tensor_np = mask_tensor_list.mask_tensor.numpy()
                df = pd.DataFrame(
                    data=np.transpose(submask_tensor_np),
                )
            else:
                if smooth:
                    mask_tensor = self.mask.get_smooth_mask(sigma)
                else:
                    mask_tensor = self.mask.mask_tensor
                # Extract submask from ids
                submask_tensor_np = self.mask.extract_submask(
                    mask_tensor, ids_time, ids_feature
                ).numpy()
                df = pd.DataFrame(
                    data=np.transpose(submask_tensor_np),
                    index=ids_feature,
                    columns=ids_time,
                )
            self.explanation = FeatureExplanation(df)
            return self.explanation
        else:
            raise exceptions.ExplainCalledBeforeFit(self.has_been_fit)

    def summary_plot(
        self,
        explanation: List = None,
        show: bool = True,
        save_path: str = "temp_dynamask_plot.png",
    ) -> None:
        """This method plots (part of) the mask.

        Args:
            explanation (List, optional): The FeatureExplanation returned by .explain(). Defaults to None, in which case it is assumed the explanation is from the result of the previous explain() call.
            show (bool, optional): Boolean value to decide if the plot is displayed. Defaults to True.
            save_path (str, optional): The path with which to save the plot if show is set to false. Defaults to "temp_dynamask_plot.png".
        Returns:
            None
        """
        if not explanation:
            explanation = self.explanation
        sns.set()
        # Generate heatmap plot
        color_map = sns.diverging_palette(10, 133, as_cmap=True)
        sns.heatmap(
            data=explanation.feature_importances,
            cmap=color_map,
            cbar_kws={"label": "Mask"},
            vmin=0,
            vmax=1,
        )
        plt.xlabel("Time")
        plt.ylabel("Feature Number")
        plt.title("Mask coefficients over time")
        if show:
            plt.show()
        else:
            plt.savefig(save_path)

    @staticmethod
    def name() -> str:
        return "dynamask"

    @staticmethod
    def pretty_name() -> str:
        return "Dynamask"

    @staticmethod
    def type() -> str:
        return "explainer"
