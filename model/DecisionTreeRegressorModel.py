from .RegressionModel import RegressionModel
from sklearn.model_selection import (
    LearningCurveDisplay,
    ShuffleSplit,
    ValidationCurveDisplay
)
from sklearn.tree import DecisionTreeRegressor


import pandas as pd


class DecisionTreeRegressorModel(RegressionModel[DecisionTreeRegressor]):
    """
    Decision tree regressor model.

    [To predict continuous target]
    """
    def __init__(self):
        super().__init__()

        self.model = \
            self._factory_model_initializer(DecisionTreeRegressor, random_state=0)


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
        Use validation curve to try out hyperparamerter max_depth.

        Parameter
        ---------
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

        Parameter
        ---------
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
