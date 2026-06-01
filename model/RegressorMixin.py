import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import mean_absolute_error


class RegressorMixin:
    """Mixin class to define specific logic of regressor."""
    #########
    # ERROR #
    #########
    def print_training_error(self,
                             y_train: pd.Series,
                             y_predicted: npt.NDArray[np.generic]) -> None:
        """Print regressor training error."""
        print("The training error of the regressor is "
             f"{self.get_training_error(y_train, y_predicted):.2f}.")

    def get_training_error(self,
                           y_train: pd.Series,
                           y_predicted: npt.NDArray[np.generic]) -> float:
        """Get regressor training error."""
        return mean_absolute_error(y_train, y_predicted)

    def print_testing_error(self,
                            y_test: pd.Series,
                            y_predicted: npt.NDArray[np.generic]) -> None:
        """Print regressor testing error."""
        print("The testing error of the regressor is "
             f"{self.get_testing_error(y_test, y_predicted):.2f}.")

    def get_testing_error(self,
                          y_test: pd.Series,
                          y_predicted: npt.NDArray[np.generic]) -> float:
        """Get regressor testing error."""
        return mean_absolute_error(y_test, y_predicted)


    ###################
    # TRAIN & PREDICT #
    ###################
    def _predict_regressor(self,
                           y_train: pd.Series,
                           y_train_predicted: npt.NDArray[np.generic],
                           y_test: pd.Series,
                           y_test_predicted: npt.NDArray[np.generic]) -> None:
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
      # Check the training error
        self.print_training_error(y_train, y_train_predicted)

      # Check the testing error
        self.print_testing_error(y_test, y_test_predicted)
