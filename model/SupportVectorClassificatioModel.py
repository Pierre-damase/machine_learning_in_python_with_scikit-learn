import pandas as pd
from sklearn.model_selection import (LearningCurveDisplay,
                                     ValidationCurveDisplay)
from sklearn.pipeline import Pipeline
from types_config import Tpipelinesteps

from .Model import Model


class SupportVectorClassificationModel(Model[Pipeline]):
    """
    Support vector machine classifier (SVM) model.

    In it's most simple form, a SVM classifier is a linear classifier behaving
    similarity to a logistic regression. Although, the optimization to find the
    optimal weights of the linear model are different.

    SVM classifier can become more flexible/expressive by using a so-called kernel
    which can make the model non-linear.

    Parameter gamma allows to tune the flexibility of the model.
    """
    def __init__(self,
                 pipeline_steps: list[Tpipelinesteps] = []):
        super().__init__(pipeline_steps=pipeline_steps)

        self.model = self._factory_pipeline_initializer(*pipeline_steps)


    ####################
    # VALIDATION CURVE #
    ####################
    def compute_validation_curve(self,
                                 x_data: pd.DataFrame,
                                 y_data: pd.Series,
                                 cv,
                                 scoring: str,
                                 score_name: str,
                                 **kwargs) -> ValidationCurveDisplay:
        """
        Use validation curve to try out hyperparamerter gamma.

        Parameter
        ---------
        **kwargs: for SVM, the gamma hyperparameter is used to control the tradeoff
              between under-fitting and over-fitting.
              Therefore, always use this parameter to find the right tradeoff. Only
              pass param_range as parameter in kwargs to try various gamma.
        """
        return super().compute_validation_curve(
            x_data,
            y_data,
            cv=cv,
            scoring=scoring,
            score_name=score_name,
            param_name="svc__gamma",
            param_range=kwargs["param_range"]
        )


    ##################
    # LEARNING CURVE #
    ##################
    def compute_learning_curve(self,
                               x_data: pd.DataFrame,
                               y_data: pd.Series,
                               cv,
                               scoring: str,
                               score_name: str,
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
            train_sizes=kwargs["train_sizes"],
        )
