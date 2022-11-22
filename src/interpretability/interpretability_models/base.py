# stdlib
from abc import ABCMeta, abstractmethod
from typing import Optional, Union, List

# third party
import numpy as np
import pandas as pd


class Explainer(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.has_been_fit = False
        self.explanation = None
        ...

    @staticmethod
    @abstractmethod
    def name() -> str:
        ...

    @staticmethod
    @abstractmethod
    def pretty_name() -> str:
        ...

    @staticmethod
    def type() -> str:
        return "explainer"

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        The function to fit the explainer to the data
        """
        ...

    @abstractmethod
    def explain(self) -> pd.DataFrame:
        """
        The function to get the explanation data from the explainer
        """
        ...


class Explanation(metaclass=ABCMeta):
    def __init__(self) -> None:
        ...

    @staticmethod
    @abstractmethod
    def name() -> str:
        ...

    @staticmethod
    def type() -> str:
        return "explanation"


class FeatureExplanation(Explanation):
    def __init__(self, feature_importances: Union[pd.DataFrame, List]) -> None:
        self.feature_importances = feature_importances
        super().__init__()

    @staticmethod
    def name() -> str:
        return "Feature Explanation"
