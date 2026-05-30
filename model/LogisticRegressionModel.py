from sklearn.linear_model import LogisticRegression
from types_config import Tlinearmodel

from .LinearModel import LinearModel


class LogisticRegressionModel(LinearModel[LogisticRegression, Tlinearmodel]):
    """
    Logistic regression model.

    [To predict discrete target]

    C is the hyperparameter of this kind of model.
    """
    _estimator_class = LogisticRegression
