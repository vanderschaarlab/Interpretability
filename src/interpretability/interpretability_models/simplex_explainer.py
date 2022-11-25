# stdlib
import sys
import os
import copy
from typing import Any, List, Tuple, Optional, Union
from abc import abstractmethod

# third party
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import webbrowser
from bs4 import BeautifulSoup

# from sklearn import neural_network # TODO: Add compatibility for sklearn


# Interpretability relative
from .utils import data
from .utils import simplex_schedulers
from .base import Explainer, Explanation

# Interpretability absolute
from interpretability.utils.pip import install
from interpretability.exceptions import exceptions

# SimplEx
for retry in range(2):
    try:
        # third party
        import simplexai

        break
    except ImportError:
        depends = ["simplexai"]
        install(depends)

from simplexai.explainers import simplex

# TODO: State assuption of input shape =(Samples X Time steps X Features)


def df_values_to_colors(df, exclude_trailing_n_cols: int = 3):
    """Gets color values based in values relative to all other values in df."""
    if exclude_trailing_n_cols:
        min_val = np.nanmin(df.values[:-exclude_trailing_n_cols])
        max_val = np.nanmax(df.values[:-exclude_trailing_n_cols])
    else:
        min_val = np.nanmin(df.values)
        max_val = np.nanmax(df.values)

    for col in df:
        # map values to colors in hex via
        # creating a hex Look up table table and apply the normalized data to it
        norm = mcolors.Normalize(
            vmin=min_val,
            vmax=max_val,
            clip=True,
        )
        lut = plt.cm.bwr(np.linspace(0.2, 0.75, 256))
        lut = np.apply_along_axis(mcolors.to_hex, 1, lut)
        if exclude_trailing_n_cols:
            a = (norm(df[col].values[:-exclude_trailing_n_cols]) * 255).astype(np.int16)
            df[col] = list(lut[a]) + ["#ffffff" for i in range(exclude_trailing_n_cols)]
        else:
            a = (norm(df[col].values) * 255).astype(np.int16)
            df[col] = lut[a]
    return df


# sort order function for decomposition
def apply_sort_order(in_list, sort_order):
    if isinstance(in_list, list):
        return [in_list[idx] for idx in sort_order]
    if torch.is_tensor(in_list):
        return [in_list.cpu().numpy()[idx] for idx in sort_order]


class SimplexBase(Explainer):
    def __init__(
        self,
        estimator: Any,
        estimator_type: str,
        feature_names: List = [],
        corpus_size: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.feature_names = feature_names
        self.corpus_size = corpus_size
        self.DEVICE = device
        estimator = copy.deepcopy(estimator)
        estimator = estimator.train()  # This causes error for sklearn models
        self.estimator = estimator.to(self.DEVICE)
        self.estimator_type = estimator_type
        super().__init__()

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        The function to fit the explainer to the data
        """
        ...

    @abstractmethod
    def explain(
        self, X: pd.DataFrame, baseline: Union[str, torch.Tensor]
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def summary_plot(self, test_record_idx) -> None:
        ...

    @staticmethod
    def name() -> str:
        return "simplex"

    @staticmethod
    def pretty_name() -> str:
        return "SimplEx"

    @staticmethod
    def type() -> str:
        return "explainer"


class SimplexExplanation(Explanation):
    def __init__(
        self,
        test_record: Union[pd.DataFrame, List],
        corpus_importances: List,
        corpus_breakdown: Union[pd.DataFrame, List],
        feature_importances: Union[pd.DataFrame, List],
        sort_order: np.array,
    ) -> None:

        self.test_record = test_record
        self.corpus_importances = corpus_importances
        self.corpus_breakdown = corpus_breakdown
        self.feature_importances = feature_importances
        self.sort_order = sort_order
        super().__init__()

    @staticmethod
    def name() -> str:
        return "Simplex Explanation"


class SimplexTabluarExplainer(SimplexBase):
    """
    A SimplEx interpretability model to explain tabluar data.

        corpus_size: The number of examples used in the corpus
        feature_names: The names of the features in the input data
    """

    def __init__(
        self,
        estimator: Any,
        corpus_X: pd.DataFrame,
        corpus_y: Union[pd.DataFrame, pd.Series],
        estimator_type: str = "classifier",
        feature_names: Optional[List] = None,
        corpus_size: int = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        random_state: int = 0,
        latent_representation: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialises the explainer.

        Args:
            estimator (Any): The model to explain. Must be trained. It must also have the method "latent_representation", which recieves the input data and returns their latent space representation. #TODO: Check, classify, and describe what models can be used. pytorch, simple python white boxes etc.
            corpus_X (pd.DataFrame): The set of records used to explain the test record(s). The individual cases for the case-based explanation.
            corpus_y (Union[pd.DataFrame, pd.Series]): The labels/targets for the corpus.
            feature_names (Optional[List], optional): The names of the feature of the dataset. If None is passed (which is the default), the feature_names will be taken from the column names of the corpus_X DataFrame.
            corpus_size (int, optional): The size of the corpus to use. If corpus_size < the length of the corpus provided, the corpus will be formed from the first `corpus_size` records. If None is passed (which is the default), the corpus_size will be taken from the length of the corpus_y object.
            device (str, optional): The device to run the. Defaults to: "cuda" if torch.cuda.is_available() else "cpu".
            random_state (int, optional): Fixes the random seed. Defaults to 0. #TODO: Use this in the code!
        """
        valid_estimator_types = ["classifier", "regressor"]
        if estimator_type not in valid_estimator_types:
            exceptions.InvalidEstimatorType(estimator_type, valid_estimator_types)

        feature_names = (
            feature_names
            if isinstance(feature_names, list)
            else pd.DataFrame(corpus_X).columns
        )
        corpus_size = (
            corpus_size
            if (corpus_size and corpus_size <= len(corpus_y))
            else len(corpus_y)
        )
        super().__init__(estimator, estimator_type, feature_names, corpus_size, device)

        self.explain_inputs = None
        self.explain_targets = None
        self.explain_predictions = None
        self.baseline = None

        corpus_data = data.TabularDataset(corpus_X, corpus_y)
        corpus_loader = DataLoader(corpus_data, batch_size=corpus_size, shuffle=False)
        self.corpus_inputs, self.corpus_targets = next(iter(corpus_loader))
        self.corpus_inputs = self.corpus_inputs.to(self.DEVICE)
        self.corpus_targets = self.corpus_targets.to(self.DEVICE)

        # Compute corpus model predictions
        if self.estimator_type == "classifier":
            if self.estimator.forward(self.corpus_inputs).shape[1] == 1:
                self.corpus_predictions = (
                    self.estimator.forward(self.corpus_inputs)
                    .to(self.DEVICE)
                    .detach()
                    .round()
                )
            elif self.estimator.forward(self.corpus_inputs).shape[1] > 1:
                self.corpus_predictions = torch.argmax(
                    self.estimator.forward(self.corpus_inputs).to(self.DEVICE).detach(),
                    dim=1,
                )
            else:
                exceptions.InvalidShapeForModelOutput(
                    self.estimator.forward(self.corpus_inputs).shape
                )
        else:
            if self.estimator.forward(self.corpus_inputs).shape[1] == 1:
                self.corpus_predictions = (
                    self.estimator.forward(self.corpus_inputs).to(self.DEVICE).detach()
                )
            else:
                exceptions.InvalidShapeForModelOutput(
                    self.estimator.forward(self.corpus_inputs).shape
                )
        try:
            # Compute the corpus and test latent representations
            corpus_latents = (
                self.estimator.latent_representation(self.corpus_inputs)
                .to(self.DEVICE)
                .detach()
            )
        except:
            raise exceptions.ModelsLatentRepresentationsNotAccessible()

        # Fit SimplEx with corpus
        self.explainer = simplex.Simplex(
            corpus_examples=self.corpus_inputs, corpus_latent_reps=corpus_latents
        )

    def fit(
        self,
        X2explain: pd.DataFrame,
        y2explain: Union[pd.DataFrame, pd.Series],
        n_epochs: int = 10000,
        n_keep: int = 5,
        reg_factor: float = 1.0,
        reg_factor_scheduler: Union[simplex_schedulers.Scheduler, None] = None,
    ) -> None:
        """
        Fits the corpus decomposition to the data to explain. This is done by learning the
        combination of corpus records that most closely resemble the test records in the latent space.

        Args:
            X2explain (pd.DataFrame): The test records to explain. Must be of shape: records x features.
            y2explain (Union[pd.DataFrame, pd.Series]): The labels/targets for the test records to explain.
            n_epochs (int, optional): The number of training epochs for the explainer. Defaults to 10000.
            n_keep (int, optional): number of corpus members allowed in the decomposition. Defaults to 5.
            reg_factor (float, optional): regularization prefactor in the objective to control the number of allowed corpus members. Defaults to 1.0.
            reg_factor_scheduler (Union[simplex_schedulers.Scheduler, None], optional): scheduler for the variation of the regularization prefactor during optimization. Defaults to None.
        """
        num_records2explain = len(y2explain)

        data2explain = data.TabularDataset(X2explain, y2explain)
        explain_data_loader = DataLoader(
            data2explain, batch_size=num_records2explain, shuffle=False
        )  # TODO: is data loader the best way to handle this single batch? Should this just be a torch.tensor?
        explain_inputs, explain_targets = next(iter(explain_data_loader))
        self.explain_inputs = explain_inputs.to(self.DEVICE)
        self.explain_targets = explain_targets.to(self.DEVICE)

        # Compute corpus model predictions
        if self.estimator_type == "classifier":
            if self.estimator.forward(self.explain_inputs).shape[1] == 1:
                self.explain_predictions = (
                    self.estimator.forward(self.explain_inputs)
                    .to(self.DEVICE)
                    .detach()
                    .round()
                )
            elif self.estimator.forward(self.explain_inputs).shape[1] > 1:
                self.explain_predictions = torch.argmax(
                    self.estimator.forward(self.explain_inputs)
                    .to(self.DEVICE)
                    .detach(),
                    dim=1,
                )
            else:
                exceptions.InvalidShapeForModelOutput(
                    self.estimator.forward(self.explain_inputs).shape
                )
        else:
            if self.estimator.forward(self.explain_inputs).shape[1] == 1:
                self.explain_predictions = (
                    self.estimator.forward(self.explain_inputs).to(self.DEVICE).detach()
                )
            else:
                exceptions.InvalidShapeForModelOutput(
                    self.estimator.forward(self.explain_inputs).shape
                )

        latents2explain = (
            self.estimator.latent_representation(self.explain_inputs)
            .to(self.DEVICE)
            .detach()
        )

        self.explainer.fit(
            test_examples=self.explain_inputs,
            n_epoch=n_epochs,
            test_latent_reps=latents2explain,
            n_keep=n_keep,
            reg_factor=reg_factor,
            reg_factor_scheduler=reg_factor_scheduler,
        )
        self.has_been_fit = True

    def explain(
        self,
        explain_id: int,
        baseline: Union[str, torch.Tensor],
        constant_val: float = 0,
    ) -> SimplexExplanation:
        """
        Gets the case-based explanation from the fit explainer. Fit() must be run before explain().

        Args:
            explain_id (int): The id of the record from the DataFrame X2explain to get the explanation for.
            baseline (Union[str, torch.Tensor]): The baseline to measure the test record against. This can be passed as a custom tensor or one of the available defaults can be used by passing the string 'zeros' or 'median'.
            constant_val (float, optional): If "constant" is passed as the baseline, constant_val defines the value at each point in that constant basleine. If baseline receives any other value, constant_val is ignored. Defaults to 0.

        Raises:
            ExplainCalledBeforeFit: raised if explain() is called before fit().

        Returns:
            A SimplexExplanation object. This has the following attributes:
                SimplexExplanation.test_record: The data for the chosen test_record for which the explanation in obtained
                SimplexExplanation.corpus_importances: A list of example importances that correspond to the corpus breakdown records.
                SimplexExplanation.corpus_breakdown: The corpus records in descending order of importance to the explanation. Each record has a 'Example Importance' value.
                SimplexExplanation.feature_importances: feature_df, # The importances of the features in the corpus breakdown. As for the corpusbreakdown, records are sorted into descending order of importance
                SimplexExplanation.sort_order: The sort order that was used to sort the records into descending importance order.
            }
        """

        self.explain_id = explain_id
        if self.has_been_fit:
            default_available_baselines = {
                "zeros": torch.zeros(size=self.corpus_inputs.shape),
                "median": self.corpus_inputs.median(dim=0, keepdim=True).values.expand(
                    self.corpus_size, -1
                ),  # Baseline tensor of the same shape as corpus_inputs,
                "constant": constant_val * torch.ones(self.corpus_inputs.shape),
                "mean": torch.mean(self.corpus_inputs, 0, keepdim=True).repeat(100, 1),
            }
            # Define baseline
            if isinstance(baseline, str):
                self.baseline = default_available_baselines[baseline].to(self.DEVICE)
            if isinstance(baseline, torch.Tensor):
                self.baseline = baseline.to(self.DEVICE)

            # Compute corpus decomposition using the jacobian projection
            self.explainer.jacobian_projection(
                test_id=explain_id, model=self.estimator, input_baseline=self.baseline
            )
            self.explainer.jacobian_projections = self.explainer.jacobian_projections
            result, sort_order = self.explainer.decompose(explain_id, return_id=True)

            test_record_df = pd.DataFrame(
                [[x.cpu().numpy() for x in self.explain_inputs[explain_id]]],
                columns=self.feature_names,
            )

            corpus_df = pd.DataFrame(
                [result[j][1].cpu().numpy() for j in range(len(result))],
                columns=self.feature_names,
            )
            example_importances = [result[j][0] for j in range(len(result))]

            feature_df = pd.DataFrame(
                [result[j][2].cpu().numpy() for j in range(len(result))],
                columns=[f"{col}_fi" for col in self.feature_names],
            )

            self.explanation = SimplexExplanation(
                test_record_df,
                example_importances,
                corpus_df,
                feature_df,
                sort_order,
            )
            return self.explanation
        else:
            raise exceptions.ExplainCalledBeforeFit(self.has_been_fit)

    def summary_plot(
        self,
        rescale_dict: Optional[
            dict
        ] = None,  # TODO: rescales with scaler object.inverse_transform
        example_importance_threshold=0.0,
        output_file_prefix: str = "",
        open_in_browser: bool = True,
        return_type: str = "styled_df",
        output_html: bool = True,
    ) -> Optional[tuple]:
        def highlight(x):
            return pd.DataFrame(
                importance_df_colors.values, index=x.index, columns=x.columns
            )

        output_file_prefix = (
            output_file_prefix + "_"
            if output_file_prefix and output_file_prefix[-1] != "_"
            else output_file_prefix
        )
        # Create test record df
        explain_record_df = self.explanation.test_record
        explain_record_df.index = ["Test Record"]
        explain_record_df["Test Prediction"] = self.explain_predictions[
            self.explain_id
        ].item()
        explain_record_df["Test Label"] = self.explain_targets[self.explain_id].item()

        if rescale_dict:
            for col_name, rescale_value in rescale_dict.items():
                explain_record_df[col_name] = explain_record_df[col_name].apply(
                    lambda x: x * rescale_value
                )

        # Create corpus df
        corpus_df = self.explanation.corpus_breakdown
        corpus_df.index = [f"Corpus member {i}" for i in range(len(corpus_df))]
        if rescale_dict:
            for col_name, rescale_value in rescale_dict.items():
                corpus_df[col_name] = corpus_df[col_name].apply(
                    lambda x: x * rescale_value
                )
        corpus_df["Example Importance"] = self.explanation.corpus_importances
        corpus_df["Corpus Prediction"] = self.corpus_predictions.cpu()
        corpus_df["Corpus Label"] = self.corpus_targets.cpu()
        corpus_df = corpus_df.loc[
            corpus_df["Example Importance"] >= example_importance_threshold
        ].copy()

        # Create importance df
        importances_df = self.explanation.feature_importances
        importances_df["Example Importance"] = self.explanation.corpus_importances
        importances_df["Corpus Prediction"] = self.corpus_predictions.cpu()
        importances_df["Corpus Label"] = self.corpus_targets.cpu()
        importances_df = importances_df.loc[
            importances_df["Example Importance"] >= example_importance_threshold
        ].copy()

        corpus_df["Example Importance"] = (
            corpus_df["Example Importance"]
            .astype(float)
            .map(lambda n: "{:.2%}".format(n))
        )

        importance_df_colors = df_values_to_colors(
            importances_df.transpose().copy(), exclude_trailing_n_cols=3
        ).transpose()
        importance_df_colors = importance_df_colors.applymap(
            lambda x: f"background-color: {x}"
        )
        display_corpus_df = corpus_df.style.apply(highlight, axis=None)
        if return_type == "html":
            html = """
                <html>
                    <head>
                        <style>
                            body{
                                margin: 0px;
                            }
                            div{
                                padding-left: 50px;
                            }
                            div#header{
                                background-color: #d97c7c;
                                padding-top: 40px;
                                padding-left: 50px;
                                padding-bottom: 15px;
                            }
                            h {
                                font-family:"Source Sans Pro", sans-serif;
                                font-weight: bold;
                                font-size: 36pt;
                                color: #ffffff;
                                text-align: center;
                                padding-top: 20px;
                                padding-bottom: 20px;
                            }
                            h3 {
                                font-family:"Source Sans Pro", sans-serif;
                                color: #333236;
                                font-size: 20pt;
                                padding-top: 20px;
                            }
                            p {
                                font-family:"Source Sans Pro", sans-serif;
                                color: #ffffff;
                                font-size: 20pt;
                                padding-top: 20px;
                            }
                            #record {
                                font-weight: bold;
                            }
                            table {
                                background-color: #ffffff;
                                border-collapse: collapse;
                                border: none;
                                margin: 50px;
                                white-space: nowrap;
                            }
                            th {
                                font-family:"Source Sans Pro", sans-serif;
                                color: #333236;
                                background-color: #d5d1ed;
                                text-align: center;
                                padding:7px;
                            }
                            td {
                                font-family:"Source Sans Pro", sans-serif;
                                text-align: center;
                                vertical-align: middle;
                                padding: 4px;
                                // background-color: #e3dfda
                            }
                        </style>
                    </head>
                    <body>
                        <div id="header">
                            <h id="header1">SimplEx Tabular Explanation</h>
                        </div>
                        <div id="test_table">
                            <h3 id="header3">Test Record</h>
                        </div>
                        <div id="corpus_table">
                            <h3 id="header3">Corpus Records</h>
                        </div>
                    </body>
                </html>
                """

            soup = BeautifulSoup(html, "lxml")

            test_record_html = explain_record_df.to_html()
            test_table_soup = BeautifulSoup(test_record_html, "lxml")
            test_table = test_table_soup.find("table", attrs={"border": "1"})
            test_table["border"] = "0"

            corpus_html_output = display_corpus_df.to_html()
            corpus_table_soup = BeautifulSoup(corpus_html_output, "lxml")

            test_table_div = soup.find_all("div", {"id": "test_table"})[0]
            test_table_div.append(test_table_soup)

            test_table_div = soup.find_all("div", {"id": "corpus_table"})[0]
            test_table_div.append(corpus_table_soup)
            with open(f"output/{output_file_prefix}SimplEx_tabular.html", "w") as f:
                f.write(str(soup))
            if open_in_browser:
                filename = (
                    "file:///"
                    + os.getcwd()
                    + "/"
                    + f"output/{output_file_prefix}SimplEx_tabular.html"
                )
                webbrowser.open_new_tab(filename)
        if return_type == "styled_df":
            return explain_record_df, display_corpus_df
        elif return_type == "html":
            return str(soup)
        else:
            return None

    @staticmethod
    def name() -> str:
        return "simplex"

    @staticmethod
    def pretty_name() -> str:
        return "SimplEx"


class SimplexTimeSeriesExplainer(SimplexBase):
    """
    A SimplEx interpretability model to explain time series data. This is
    also compatible with any other 3 dimensional dataset, such as black & white
    image data.

        corpus_size: The number of examples used in the corpus
        feature_names: The names of the features in the input data
    """

    def __init__(
        self,
        estimator: Any,
        corpus_X: np.array,
        corpus_y: np.array,
        estimator_type: str = "classifier",
        feature_names: Optional[List] = None,
        corpus_size: int = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        random_state: int = 0,
    ) -> None:

        """
        Initializes the explainer.

        Args:
            estimator (Any): The model to explain. Must be trained. #TODO: Check, classify and, describe what models can be used. pytorch, simple python white boxes etc.
            corpus_X (np.array): The set of records used to explain the test record(s). The individual cases for the case-based explanation. Must be of shape: records x time_steps x features.
            corpus_y (np.array): The labels/targets for the corpus.
            feature_names (Optional[List], optional): The names of the feature of the dataset. If None is passed (which is the default), the feature_names will be a sequential list of numbers.
            corpus_size (int, optional): The size of the corpus to use. If corpus_size < the length of the corpus provided, the corpus will be formed from the first `corpus_size` records. If None is passed (which is the default), the corpus_size will be taken from the length of the corpus_y object.
            device (str, optional): The device to run the. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
            random_state (int, optional): Fixes the random seed. Defaults to 0. #TODO: Use this in the code!
        """
        valid_estimator_types = ["classifier", "regressor"]
        if estimator_type not in valid_estimator_types:
            raise exceptions.InvalidEstimatorType(estimator_type, valid_estimator_types)

        self.max_time_points = max([corpus_X[i].shape[0] for i in range(len(corpus_X))])
        feature_names = (
            feature_names
            if isinstance(feature_names, list)
            else list(range(max([corpus_X[i].shape[1] for i in range(len(corpus_X))])))
        )
        corpus_size = (
            corpus_size
            if (corpus_size and corpus_size <= len(corpus_y))
            else len(corpus_y)
        )

        super().__init__(estimator, estimator_type, feature_names, corpus_size, device)

        corpus_data = data.TimeSeriesDataset(corpus_X, corpus_y)
        corpus_loader = DataLoader(corpus_data, batch_size=corpus_size, shuffle=False)
        self.corpus_inputs, self.corpus_targets = next(iter(corpus_loader))
        self.corpus_inputs = self.corpus_inputs.to(self.DEVICE)
        self.corpus_targets = self.corpus_targets.to(self.DEVICE)

        # Compute corpus model predictions
        self.corpus_predictions = None
        if self.estimator_type == "classifier":
            if self.estimator.forward(self.corpus_inputs).shape[1] == 1:
                self.corpus_predictions = (
                    self.estimator.forward(self.corpus_inputs)
                    .to(self.DEVICE)
                    .detach()
                    .round()
                )
            elif self.estimator.forward(self.corpus_inputs).shape[1] > 1:
                self.corpus_predictions = torch.argmax(
                    self.estimator.forward(self.corpus_inputs).to(self.DEVICE).detach(),
                    dim=1,
                )
            else:
                raise exceptions.InvalidShapeForModelOutput(
                    self.estimator.forward(self.corpus_inputs).shape
                )
        else:
            if self.estimator.forward(self.corpus_inputs).shape[1] == 1:
                self.corpus_predictions = (
                    self.estimator.forward(self.corpus_inputs).to(self.DEVICE).detach()
                )
            else:
                raise exceptions.InvalidShapeForModelOutput(
                    self.estimator.forward(self.corpus_inputs).shape
                )

        # Compute the corpus and test latent representations
        try:
            corpus_latents = (
                self.estimator.latent_representation(self.corpus_inputs)
                .to(self.DEVICE)
                .detach()
            )
        except:
            raise exceptions.ModelsLatentRepresentationsNotAccessible()

        # Fit SimplEx
        self.explainer = simplex.Simplex(
            corpus_examples=self.corpus_inputs, corpus_latent_reps=corpus_latents
        )

    def fit(
        self,
        X2explain: Union[pd.DataFrame, np.array],
        y2explain: Union[pd.DataFrame, pd.Series, np.array],
        n_epochs: int = 10000,
        n_keep: int = 5,
        reg_factor: float = 1.0,
        reg_factor_scheduler: Union[simplex_schedulers.Scheduler, None] = None,
    ) -> None:
        """
        Fits the corpus decomposition to the data to explain. This is done by learning the
        combination of corpus records that most closely resemble the test records in the latent space.

        Args:
            X2explain (pd.DataFrame): The test records to explain. Must be of shape: records x time_steps x features.
            y2explain (Union[pd.DataFrame, pd.Series]): The labels/targets for the test records to explain.
            n_epochs (int, optional): The number of training epochs for the explainer. Defaults to 10000.
            n_keep (int, optional): number of corpus members allowed in the decomposition. Defaults to 5.
            reg_factor (float, optional): regularization prefactor in the objective to control the number of allowed corpus members. Defaults to 1.0.
            reg_factor_scheduler (Union[simplex_schedulers.Scheduler, None], optional): scheduler for the variation of the regularization prefactor during optimization. Defaults to None.
        """

        num_records2explain = len(y2explain)

        data2explain = data.TimeSeriesDataset(X2explain, y2explain)
        explain_data_loader = DataLoader(
            data2explain, batch_size=num_records2explain, shuffle=False
        )  # TODO: Think, is data loader the bets way to handle this single batch?
        self.explain_inputs, self.explain_targets = next(iter(explain_data_loader))
        self.explain_inputs = self.explain_inputs.to(self.DEVICE)
        self.explain_targets = self.explain_targets.to(self.DEVICE)

        # Compute corpus model predictions
        if self.estimator_type == "classifier":
            if self.estimator.forward(self.explain_inputs).shape[1] == 1:
                self.explain_predictions = (
                    self.estimator.forward(self.explain_inputs)
                    .to(self.DEVICE)
                    .detach()
                    .round()
                )
            elif self.estimator.forward(self.explain_inputs).shape[1] > 1:
                self.explain_predictions = torch.argmax(
                    self.estimator.forward(self.explain_inputs)
                    .to(self.DEVICE)
                    .detach(),
                    dim=1,
                )
            else:
                exceptions.InvalidShapeForModelOutput(
                    self.estimator.forward(self.explain_inputs).shape
                )
        else:
            if self.estimator.forward(self.explain_inputs).shape[1] == 1:
                self.explain_predictions = (
                    self.estimator.forward(self.explain_inputs).to(self.DEVICE).detach()
                )
            else:
                exceptions.InvalidShapeForModelOutput(
                    self.estimator.forward(self.explain_inputs).shape
                )

        latents2explain = (
            self.estimator.latent_representation(self.explain_inputs)
            .to(self.DEVICE)
            .detach()
        )

        self.explainer.fit(
            test_examples=self.explain_inputs,
            n_epoch=n_epochs,
            test_latent_reps=latents2explain,
            reg_factor=reg_factor,
            n_keep=n_keep,
            reg_factor_scheduler=reg_factor_scheduler,
        )
        self.has_been_fit = True

    def explain(
        self,
        explain_id: int,
        baseline: Union[str, torch.Tensor],
        constant_val: float = 0,
    ) -> Tuple:
        """
        Gets the case-based explanation from the fit explainer. Fit() must be run before explain().

        Args:
            explain_id (int): The id of the record from the DataFrame X2explain to get the explanation for.
            baseline (Union[str, torch.Tensor]): The baseline to measure the test record against. This can be passed as a custom tensor or one of the available defaults can be used by passing the string 'zeros' or 'median'.
            constant_val (float, optional): If "constant" is passed as the baseline, constant_val defines the value at each point in that constant basleine. If baseline receives any other value, constant_val is ignored. Defaults to 0.

        Raises:
            ExplainCalledBeforeFit: raised if explain() is called before fit().

        Returns:
            A SimplexExplanation object. This has the following attributes:
                SimplexExplanation.test_record: The data for the chosen test_record for which the explanation in obtained
                SimplexExplanation.example_importances: A list of example importances that correspond to the corpus breakdown records.
                SimplexExplanation.corpus_breakdown: The corpus records in descending order of importance to the explanation. Each record has a 'Example Importance' value.
                SimplexExplanation.feature_importances: feature_df, # The importances of the features in the corpus breakdown. As for the corpusbreakdown, records are sorted into descending order of importance
                SimplexExplanation.sort_order: The sort order that was used to sort the records into descending importance order.
            }
        """
        self.explain_id = explain_id
        expand_dim = list(self.corpus_inputs.median(dim=0, keepdim=True).values.shape)
        expand_dim[0] = self.corpus_size
        expand_dim[1] = self.max_time_points
        if self.has_been_fit:
            default_available_baselines = {
                "zeros": torch.zeros(size=self.corpus_inputs.shape),
                "constant": constant_val * torch.ones(self.corpus_inputs.shape),
                "median": self.corpus_inputs.median(dim=0, keepdim=True).values.expand(
                    *expand_dim
                ),
                "mean": torch.mean(torch.mean(self.corpus_inputs, 1), 0).expand(
                    *expand_dim
                ),
            }
            # Define baseline
            if isinstance(baseline, str):
                self.baseline = default_available_baselines[baseline].to(self.DEVICE)
            if isinstance(baseline, torch.Tensor):
                self.baseline = baseline.to(self.DEVICE)

            self.explainer.jacobian_projection(
                test_id=self.explain_id,
                model=self.estimator,
                input_baseline=self.baseline,
            )
            result, sort_order = self.explainer.decompose(
                self.explain_id, return_id=True
            )

            self.explanation = SimplexExplanation(
                self.explain_inputs[self.explain_id].cpu().numpy(),
                [result[j][0] for j in range(len(result))],
                [result[j][1].cpu().numpy() for j in range(len(result))],
                [result[j][2].cpu().numpy() for j in range(len(result))],
                sort_order,
            )
            return self.explanation

        else:
            raise exceptions.ExplainCalledBeforeFit(self.has_been_fit)

    def summary_plot(
        self,
        rescale_dict: Optional[dict] = None,
        plot_test: bool = True,
        example_importance_threshold=0.1,
        time_steps_to_display=10,
        output_file_prefix: str = "",
        open_in_browser: bool = True,
        return_type="styled_df",
    ) -> Optional[tuple]:
        def highlight(x):
            return pd.DataFrame(
                importance_df_colors.values, index=x.index, columns=x.columns
            )

        output_file_prefix = (
            output_file_prefix + "_"
            if output_file_prefix and output_file_prefix[-1] != "_"
            else output_file_prefix
        )

        # Test record
        if plot_test:

            test_record_last_time_step = (
                self.explanation.test_record[
                    ~np.all(self.explanation.test_record == 0, axis=1)
                ].shape[0]
                - 1
            )
            test_record_df = (
                pd.DataFrame(
                    self.explanation.test_record[
                        test_record_last_time_step
                        - (time_steps_to_display - 1) : test_record_last_time_step
                        + 1,
                    ],
                    columns=self.feature_names,
                    index=[
                        f"(t_max) - {i}" if i != 0 else "(t_max)"
                        for i in reversed(range(time_steps_to_display))
                    ],
                )
                if time_steps_to_display <= test_record_last_time_step
                else pd.DataFrame(
                    self.explanation.test_record[0 : test_record_last_time_step + 1],
                    columns=self.feature_names,
                    index=[
                        f"(t_max) - {i}" if i != 0 else "(t_max)"
                        for i in reversed(range(test_record_last_time_step + 1))
                    ],
                )
            )
            if rescale_dict:
                for col_name, rescale_value in rescale_dict.items():
                    test_record_df[col_name] = test_record_df[col_name].apply(
                        lambda x: x * rescale_value
                    )
            try:
                display(test_record_df)
            except:
                pass

            if return_type == "html":
                html_output = test_record_df.transpose().to_html()
                test_soup = BeautifulSoup(html_output, "lxml")
                head_tag = test_soup.new_tag("head")
                test_soup.html.append(head_tag)

                style_tag = test_soup.new_tag("style")
                style_tag.string = """
                    body{
                        margin: 0px;
                    }
                    h {
                        font-family:'Source Sans Pro', sans-serif;
                        font-weight: bold;
                        font-size: 36pt;
                        color: #ffffff;
                        text-align: center;
                        padding-top: 20px;
                    }
                    p {
                        font-family:'Source Sans Pro', sans-serif;
                        color: #ffffff;
                        font-size: 20pt;
                    }
                    p#record {
                        font-family:'Source Sans Pro', sans-serif;
                        font-weight: bold;
                    }
                    div{
                        background-color: #d97c7c;
                        padding-top: 40px;
                        padding-left: 50px;
                        padding-bottom: 15px;
                    }
                    table {
                        background-color: #ffffff;
                        border-collapse: collapse;
                        border: none;
                        margin: 50px;
                        white-space: nowrap;
                    }
                    th {
                        font-family:'Source Sans Pro', sans-serif;
                        background-color: #d5d1ed;
                        text-align: center;
                        padding:7px;
                    }
                    td {
                        font-family:'Source Sans Pro', sans-serif;
                        background-color: #e3dfda
                        text-align: center;
                        vertical-align: middle;
                        padding: 4px;
                    }
                """
                test_soup.html.head.append(style_tag)

                new_div = test_soup.new_tag("div")
                test_soup.html.body.insert(0, new_div)

                new_header_tag = test_soup.new_tag("h")
                new_header_tag.string = f"SimplEx Explanation"
                test_soup.html.body.div.insert(0, new_header_tag)

                new_p_tag = test_soup.new_tag("p", id="record")
                new_p_tag.string = f"Record: Test record"
                test_soup.html.body.div.insert(1, new_p_tag)

                new_p_tag = test_soup.new_tag("p", id="pred/label")
                new_p_tag.string = f"Test prediction: {self.explain_predictions[self.explain_id].item():0.0f} \xa0\xa0|\xa0\xa0 Test label: {self.explain_targets[self.explain_id]:0.0f}"
                test_soup.html.body.div.insert(2, new_p_tag)
                with open(
                    f"output/{output_file_prefix}SimplEx_time_series_test_record.html",
                    "w",
                ) as f:
                    f.write(str(test_soup))

                if open_in_browser:
                    filename = (
                        "file:///"
                        + os.getcwd()
                        + "/"
                        + f"output/{output_file_prefix}SimplEx_time_series_test_record.html"
                    )
                    webbrowser.open_new_tab(filename)

        # Corpus Feature values
        last_time_step_idx = [
            self.explanation.corpus_breakdown[j][
                ~np.all(self.explanation.corpus_breakdown[j] == 0, axis=1)
            ].shape[0]
            - 1
            for j in range(len(self.explanation.corpus_breakdown))
        ]

        corpus_dfs = [
            pd.DataFrame(
                self.explanation.corpus_breakdown[j][
                    idx - (time_steps_to_display - 1) : idx + 1
                ],
                columns=self.feature_names,
            )
            if time_steps_to_display <= idx
            else pd.DataFrame(
                self.explanation.corpus_breakdown[j][0 : idx + 1],
                columns=self.feature_names,
            )
            for j, idx in zip(
                range(len(self.explanation.corpus_breakdown)), last_time_step_idx
            )
        ]
        # Patient importances
        importance_dfs = [
            pd.DataFrame(
                self.explanation.feature_importances[j][
                    idx - (time_steps_to_display - 1) : idx + 1
                ],
                columns=[f"{col}_fi" for col in self.feature_names],
            )
            if time_steps_to_display <= idx
            else pd.DataFrame(
                self.explanation.feature_importances[j][0 : idx + 1],
                columns=[f"{col}_fi" for col in self.feature_names],
            )
            for j, idx in zip(
                range(len(self.explanation.feature_importances)), last_time_step_idx
            )
        ]

        if rescale_dict:
            for corpus_df in corpus_dfs:
                for col_name, rescale_value in rescale_dict.items():
                    corpus_df[col_name] = corpus_df[col_name].apply(
                        lambda x: x * rescale_value
                    )

        corpus_data = [
            {
                "feature_vals": corpus_dfs[i].transpose(),
                "Label": apply_sort_order(
                    self.corpus_targets, self.explanation.sort_order
                )[i],
                "Prediction": apply_sort_order(
                    self.corpus_predictions, self.explanation.sort_order
                )[i],
                "Example Importance": self.explanation.corpus_importances[i],
            }
            for i in range(len(corpus_dfs))
        ]
        importance_data = [
            {
                "importance_vals": importance_dfs[i].transpose(),
                "Label": apply_sort_order(
                    self.corpus_targets, self.explanation.sort_order
                )[i],
                "Prediction": apply_sort_order(
                    self.corpus_predictions, self.explanation.sort_order
                )[i],
                "Example Importance": self.explanation.corpus_importances[i],
            }
            for i in range(len(corpus_dfs))
        ]

        max_importance = max([example["Example Importance"] for example in corpus_data])
        corpus_data = [
            example
            for example in corpus_data
            if example["Example Importance"] >= example_importance_threshold
        ]
        importance_data = [
            example
            for example in importance_data
            if example["Example Importance"] >= example_importance_threshold
        ]

        display_corpus_dfs = []
        if len(corpus_data) == 0:
            raise exceptions.ExampleImportanceThresholdTooHigh(
                example_importance_threshold, max_importance
            )
        for example_i in range(len(corpus_data)):
            importance_df_colors = df_values_to_colors(
                importance_data[example_i]["importance_vals"].copy(),
                exclude_trailing_n_cols=0,
            )
            importance_df_colors = importance_df_colors.applymap(
                lambda x: f"background-color: {x}"
            )
            display_corpus_df = corpus_data[example_i]["feature_vals"].style.apply(
                highlight, axis=None
            )
            display_corpus_dfs.append(display_corpus_df)
            print(f"Corpus Example: {example_i}")
            print(f"Example Importance: {corpus_data[example_i]['Example Importance']}")
            try:
                display(display_corpus_df)
            except:
                pass
            if return_type == "html":
                html_output = display_corpus_df.to_html()
                soup = BeautifulSoup(html_output, "lxml")
                soup.select_one("style").append(
                    """
                    body{
                        margin: 0px;
                    }
                    h {
                        font-family:'Source Sans Pro', sans-serif;
                        font-weight: bold;
                        font-size: 36pt;
                        color: #ffffff;
                        text-align: center;
                        padding-top: 20px;
                        padding-bottom: 20px;
                    }
                    p {
                        font-family:'Source Sans Pro', sans-serif;
                        color: #ffffff;
                        font-size: 20pt;
                    }
                    #record {
                        font-weight: bold;
                    }
                    div{
                        background-color: #d97c7c;
                        padding-top: 40px;
                        padding-left: 50px;
                        padding-bottom: 15px;
                    }
                    table {
                        background-color: #ffffff;
                        border-collapse: collapse;
                        border: none;
                        margin: 50px;
                        white-space: nowrap;
                    }
                    th {
                        font-family:'Source Sans Pro', sans-serif;
                        background-color: #d5d1ed;
                        text-align: center;
                        padding:7px;
                    }
                    td {
                        font-family:'Source Sans Pro', sans-serif;
                        text-align: center;
                        vertical-align: middle;
                        padding: 4px;
                    }
                """
                )

                new_div = soup.new_tag("div")
                soup.html.body.insert(0, new_div)

                new_tag = soup.new_tag("h")
                new_tag.string = f"SimplEx Explanation"
                soup.html.body.div.append(new_tag)

                new_tag = soup.new_tag("p", id="record")
                new_tag.string = f"Record: Corpus Example {example_i}"
                soup.html.body.div.append(new_tag)

                new_tag = soup.new_tag("p", "importance")
                new_tag.string = f"Example Importance: {100*corpus_data[example_i]['Example Importance']:0.2f}% \xa0\xa0|\xa0\xa0 Corpus Prediction: {corpus_data[example_i]['Prediction'].item():0.0f} \xa0\xa0|\xa0\xa0 Corpus Label: {corpus_data[example_i]['Label']:0.0f}"
                soup.html.body.div.append(new_tag)
                with open(
                    f"output/{output_file_prefix}SimplEx_time_series_corpus_member_{example_i}.html",
                    "w",
                ) as f:
                    f.write(str(soup))
                if open_in_browser:
                    filename = (
                        "file:///"
                        + os.getcwd()
                        + "/"
                        + f"output/{output_file_prefix}SimplEx_time_series_corpus_member_{example_i}.html"
                    )
                    webbrowser.open_new_tab(filename)

        if return_type == "html":
            return str(test_soup), str(soup)
        else:
            return None

    @staticmethod
    def name() -> str:
        return "simplex"

    @staticmethod
    def pretty_name() -> str:
        return "SimplEx"
