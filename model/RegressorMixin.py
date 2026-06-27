from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from sklearn import metrics
from types_config import MetricFunctionType


class RegressorMixin:
    """Mixin class to define specific logic of regressor."""
    ############
    # PROPERTY #
    ############
    @property
    def is_regressor(self) -> bool:
        return True


    ###################
    # TRAIN & PREDICT #
    ###################
    def predict(self,
                x_train: pd.DataFrame,
                x_test: pd.DataFrame,
                y_train: pd.Series,
                y_test: pd.Series) -> None:
        """
        Predict the target and check model performance.

        Parameters
        ----------
        model: machine learning model

        y_train: training target

        y_train_predicted: predicted training target

        y_test: testing target

        y_test_predicted: predicted testing target
        """
        # Get model
        model = getattr(self, "model")

        # Predict target
        y_train_predicted = getattr(model, "predict")(x_train)
        y_test_predicted = getattr(model, "predict")(x_test)

      # Check the training error
        self.print_model_prediction_error(y_train, y_train_predicted, dataset="training")

      # Check the testing error
        self.print_model_prediction_error(y_test, y_test_predicted, dataset="testing")


    ################
    # METRIC CURVE #
    ################
    def compute_metric_curve(self,
                             y_data: pd.Series,
                             y_predicted: pd.Series,
                             kind: Literal["actual_vs_predicted", "residual_vs_predicted"],
                             ax: Axes | None = None):
        """
        Compute metric curve either a precision recall (PR) curve or a receiver operating
        characteristic (ROC) curve.
        """
        curve = metrics.PredictionErrorDisplay.from_predictions(y_data,
                                                                y_predicted,
                                                                kind=kind,
                                                                scatter_kwargs={"alpha": 0.5},
                                                                ax=ax)

        # Label
        curve.ax_.set_xlabel("Predicted values")
        if kind == "actual_vs_predicted":
            curve.ax_.set_ylabel("Original values")
        else:
            curve.ax_.set_ylabel("Residual values")

        curve.ax_.axis("square")


    #####################
    # MODEL PERFORMANCE #
    #####################
    def print_model_prediction_error(self,
                                     y_data: pd.Series,
                                     y_predicted: npt.NDArray[np.generic],
                                     dataset: Literal["training", "testing", "whole"],
                                     metric_func: MetricFunctionType = "mean_absolute_error") \
                                     -> None:
        """
        Print the model prediction error, either for the training dataset, the testing dataset or
        the whole dataset.

        Parameters
        ----------
        y_data: the original targets

        y_predicted: the predicted targets

        dataset: either training dataset, testing dataset or the whole dataset

        metric_func: which metric function used to evaluate the model prediction error
        """
        print(f"\nThe regressor {dataset} error is "
              f"{self.get_model_prediction_error(y_data, y_predicted):.2f} ({metric_func}).")

    def get_model_prediction_error(self,
                                   y_data: pd.Series,
                                   y_predicted: npt.NDArray[np.generic],
                                   metric_func: MetricFunctionType = "mean_absolute_error") \
                                   -> float:
        """Get regressor error."""
        return getattr(metrics, metric_func)(y_data, y_predicted)

    def evaluate_generalization_performance(self,
                                            x_test: pd.DataFrame,
                                            y_test: pd.Series,
                                            **kwargs) -> None:
        """
        Use various evaluation metrics to evaluate the predictive model generalization performance.

        [This method is higly specific to regressor.]

        Mean squared error (MSE). Basic loss function that can be used to evaluate the model since
        it's optimised by said model.

        Mean absolute error. It's the most important metric because it indicates the average
        difference between the predictions and the actual values.

        Median absolute error. Really important as for the mean absolute error with a reduce
        influence of large errors.

        Parameters
        ----------
        x_test: testing data

        y_test: testing targets

        **kwargs: additional parameters such as x_train and y_train
        """
        assert "x_train" in kwargs.keys() and "y_train" in kwargs.keys(), \
            "x_train and y_train are required to evaluate a regressor."
        x_train, y_train = kwargs["x_train"], kwargs["y_train"]

        # Get model
        model = getattr(self, "model")

        # Predict the testing target using the testing data
        y_train_predicted = getattr(model, "predict")(x_train)
        y_test_predicted = getattr(model, "predict")(x_test)

        metric_functions: list[MetricFunctionType] = [
            "mean_squared_error", "mean_absolute_error", "median_absolute_error",
            "mean_absolute_percentage_error"
        ]
        for metric_func in metric_functions:
            self.print_model_prediction_error(
                y_train, y_train_predicted, metric_func=metric_func, dataset="training"
            )
            self.print_model_prediction_error(
                y_test, y_test_predicted, metric_func=metric_func, dataset="testing"
            )

        # R². Coefficient of determination, which is used as the default score in sklearn. R2
        # represents the proportion of variance of the target that is explained by the independent
        # variables in the model. R2 gives insight about the quality of the model's fit but is not
        # really useful
        r2 = model.score(x_test, y_test)

        # Set up plots
        _, axs = plt.subplots(ncols=2, figsize=(24, 10))

        # Prediction error curve.
        self.compute_metric_curve(
            y_test, y_test_predicted, kind="actual_vs_predicted", ax=axs[0]
        )
        self.compute_metric_curve(
            y_test, y_test_predicted, kind="residual_vs_predicted", ax=axs[1]
        )
        plt.tight_layout()
        plt.show()
