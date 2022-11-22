class ExplainCalledBeforeFit(Exception):
    """
    Exception raised when explain is called before fit.
    """

    def __init__(
        self,
        explainer_has_been_fit,
        exception_message="The explainer must be fit before explain() is called. Please call .fit() first.",
    ):
        self.explainer_has_been_fit = explainer_has_been_fit
        self.message = exception_message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class MeasureFitQualityCalledBeforeFit(Exception):
    """
    Exception raised when measure_fit_quality() is called before fit().
    """

    def __init__(
        self,
        explainer_has_been_fit,
        exception_message="The explainer must be fit before measure_fit_quality() is called. Please call .fit() first.",
    ):
        self.explainer_has_been_fit = explainer_has_been_fit
        self.message = exception_message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class ModelsLatentRepresentationsNotAccessible(Exception):
    """
    Exception raised when latent_representation() is called on the model object, but the model has no such method.
    """

    def __init__(
        self,
        exception_message="The model object has no 'latent_representation()' method, which receives the input data and returns their latent space representation. This method is a requirement for using SimplEx. The method should be the same as the full forward() method, but stop short of the final layer. For help in adding a method please see the examples of the models here: https://github.com/vanderschaarlab/Simplex/tree/main/src/simplexai/models",
    ):
        self.message = exception_message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class InvalidEstimatorType(Exception):
    """
    Exception raised when the estimator type is in the list of valid types (usually classifier or regressor).
    """

    def __init__(
        self,
        estimator_type,
        valid_estimator_types=[],
    ):
        self.estimator_type = estimator_type
        self.valid_estimator_types = valid_estimator_types
        self.message = f"Estimator_type \"{self.estimator_type}\" not valid. Please use one of the following values: {', '.join(self.valid_estimator_types)}."
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class InvalidShapeForModelOutput(Exception):
    """
    Exception raised when the estimator output is of an invalid shape.
    """

    def __init__(
        self,
        output_shape: int,
    ):
        self.message = f"Invalid shape of {output_shape} for output from the forward call of the estimator. The explainer supports single or multi-label classification and single label regression only."
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class ExampleImportanceThresholdTooHigh(Exception):
    """
    Exception raised when the Example Importance Threshold is too high, such that there are no examples left for the explanation with an importance above the threshold.
    """

    def __init__(
        self,
        example_importance_threshold: int,
        max_importance: float,
    ):
        self.message = f"example_importance_threshold of {example_importance_threshold} is highest example importance value of {max_importance:0.2f}. Please reduce the example_importance_threshold to below {max_importance:0.2f} in order to see the examples."
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class NoDataToExplain(Exception):
    """
    Exception raised when a fit call is made without any data to explain.
    """

    def __init__(
        self,
    ):
        self.message = f"No data to explain has been passed to the explainer. This is only allowed when re-fitting a previously fit explainer."
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


# class InvalidTimeStepAxis(Exception):
#     """
#
#     """

#     def __init__(
#         self,
#         time_step_axis,
#     ):
#         self.message = (
#             f"Value given for time_step_axis must be 0 or 1, not {time_step_axis}."
#         )
#         super().__init__(self.message)

#     def __str__(self):
#         return f"{self.message}"
