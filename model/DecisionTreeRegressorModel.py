import pandas as pd
from sklearn.model_selection import (LearningCurveDisplay, ShuffleSplit,
                                     ValidationCurveDisplay)
from sklearn.tree import DecisionTreeRegressor
from types_config import Tmodel

from .DecisionBoundaryMixin import DecisionBoundaryMixin
from .Model import Model
from .RegressorMixin import RegressorMixin
from .TreeMixin import TreeMixin


class DecisionTreeRegressorModel(DecisionBoundaryMixin,
                                 RegressorMixin,
                                 TreeMixin,
                                 Model[DecisionTreeRegressor, Tmodel]):
    """
    Decision tree regressor model.

    [To predict continuous target]
    """
    _estimator_class = DecisionTreeRegressor

    ####################
    # VALIDATION CURVE #
    ####################
    def compute_validation_curve(self,
                                 x_data: pd.DataFrame,
                                 y_data: pd.Series,
                                 scoring: str,
                                 score_name: str,
                                 negate_score: bool,
                                 cv: ShuffleSplit,
                                 **kwargs) -> ValidationCurveDisplay:
        """
        Use validation curve to try out hyperparameter max_depth.

        Parameters
        ----------
        **kwargs: for decision tree regressor, the max_depth hyperparameter is used to
              control the tradeoff between under-fitting and over-fitting.
              Therefore, always use this parameter to find the right tradeoff. Only
              pass param_range as parameter in kwargs to try various max_depth.
        """
        return super().compute_validation_curve(
            x_data,
            y_data,
            scoring=scoring,
            score_name=score_name,
            cv=cv,
            negate_score=negate_score,
            param_name="max_depth",
            param_range=kwargs["param_range"]
        )


    ##################
    # LEARNING CURVE #
    ##################
    def compute_learning_curve(self,
                               x_data: pd.DataFrame,
                               y_data: pd.Series,
                               cv: ShuffleSplit,
                               scoring: str,
                               score_name: str,
                               negate_score: bool,
                               **kwargs) -> LearningCurveDisplay:
        """
        Use learning curve to try out various training set size.

        Parameters
        ----------
        **kwargs: pass train_sizes to try out various training set sizes
        """
        return super().compute_learning_curve(
            x_data,
            y_data,
            cv=cv,
            scoring=scoring,
            score_name=score_name,
            negate_score=negate_score,
            train_sizes=kwargs["train_sizes"],
        )
