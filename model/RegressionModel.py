from .Model import Model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import (
    cross_validate,
    ShuffleSplit
)
from sklearn.tree import DecisionTreeRegressor
from typing import TypeVar

import pandas as pd
import numpy as np

Tregression = TypeVar('Tregression', bound=DecisionTreeRegressor)


class RegressionModel(Model):
    """
    Decision tree regressor model.
    
    [To predict continuous target]
    """
    def __init__(self):
        super().__init__()


    #########
    # ERROR #
    #########
    """Print regressor training error."""
    def print_training_error(self, y_train: pd.DataFrame, y_predicted: np.ndarray[any]) -> None:
        print("The training error of the regressor is "
             f"{self.get_training_error(y_train, y_predicted):.2f}.")

    """Get regressor training error."""
    def get_training_error(self, y_train: pd.DataFrame, y_predicted: np.ndarray[any]) -> float:
        return mean_absolute_error(y_train, y_predicted)

    """Print regressor testing error."""
    def print_testing_error(self, y_test: pd.DataFrame, y_predicted: np.ndarray[any]) -> None:
        print("The testing error of the regressor is "
             f"{self.get_testing_error(y_test, y_predicted):.2f}.")

    """Get regressor testing error."""
    def get_testing_error(self, y_test: pd.DataFrame, y_predicted: np.ndarray[any]) -> float:
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
            model: Tregression, 
            x_train: pd.DataFrame, 
            x_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series) -> np.ndarray[any]:
        # Predict target
        y_train_predicted = model.predict(x_train)
        y_test_predicted = model.predict(x_test)

        # Check the training error
        self.print_training_error(y_train, y_train_predicted)

        # Check the testing error
        self.print_testing_error(y_test, y_test_predicted)


    ####################
    # CROSS VALIDATION #
    ####################
    """
    To performe a shuffle split cross-validation strategy.

    Scikit-learn allow the use of any metric, like 'mean_absolute_error' into a score 
    to be used in cross validate. In order to do so, pass a string of the error metric
    with and additional neg_ such as 'neg_mean_absolute_error'.
    
    Parameter
    ---------
    x_data: the whole dataset

    y_data: the whole targets

    n_splits: number of re-shuffling & splitting iterations.
    """
    def shuffle_split_cross_validate(self, 
                                     x_data: pd.DataFrame, 
                                     y_data: pd.Series, 
                                     n_splits: int, 
                                     scoring: str,
                                     test_size: int = 0.3):
        # Set the cross-validation strategy and performe it 
        results = cross_validate(
            self.model, 
            x_data, 
            y_data, 
            cv=ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0), 
            scoring=scoring, 
            error_score="raise")
        
        # Revert the negation to get the actual error
        results["test_score"] = -results["test_score"]

        return results

    """Print the result (accuracy and fitting time) of a kfold cross-validation strategy."""
    def print_shuffle_split_cross_validation_accuracy(self, scores: dict[str, np.array[float]]) -> None:
        print(f"Shuffle split cross-validation with n={len(scores['test_score'])}.\n"
               "The mean cross-validated testing error is: "
             f"{scores['test_score'].mean():3f} ± {scores['test_score'].std():3f} "
             f" with an average fitting time of {scores['fit_time'].mean():3f}")