from .Model import Model
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from .types import (
    Tpipelinesteps,
    Tregressor
)

import pandas as pd
import numpy as np
import numpy.typing as npt

class RegressionModel(Model[Tregressor]):
    """
    Decision tree regressor model.

    [To predict continuous target]
    """
    def __init__(self,
                 pipeline_steps: list[Tpipelinesteps] = []):
        super().__init__(pipeline_steps)


    #########
    # ERROR #
    #########
    """Print regressor training error."""
    def print_training_error(self,
                             y_train: pd.Series,
                             y_predicted: npt.NDArray[np.generic]) -> None:
        print("The training error of the regressor is "
             f"{self.get_training_error(y_train, y_predicted):.2f}.")

    """Get regressor training error."""
    def get_training_error(self,
                           y_train: pd.Series,
                           y_predicted: npt.NDArray[np.generic]) -> float:
        return mean_absolute_error(y_train, y_predicted)

    """Print regressor testing error."""
    def print_testing_error(self,
                            y_test: pd.Series,
                            y_predicted: npt.NDArray[np.generic]) -> None:
        print("The testing error of the regressor is "
             f"{self.get_testing_error(y_test, y_predicted):.2f}.")

    """Get regressor testing error."""
    def get_testing_error(self,
                          y_test: pd.Series,
                          y_predicted: npt.NDArray[np.generic]) -> float:
        return mean_absolute_error(y_test, y_predicted)


    ###################
    # TRAIN & PREDICT #
    ###################
    """
    Predict the target and check model performance.

    Parameters
    ----------
    model: machine learning model

    x_train: training data

    x_test: testing data

    y_train: training target

    y_test: testing target
    """
    def predict(
            self,
            x_train: pd.DataFrame,
            x_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series) -> None:
        # Predict target
        y_train_predicted = self.model.predict(x_train)
        y_test_predicted = self.model.predict(x_test)

        # Check the training error
        self.print_training_error(y_train, y_train_predicted)

        # Check the testing error
        self.print_testing_error(y_test, y_test_predicted)
