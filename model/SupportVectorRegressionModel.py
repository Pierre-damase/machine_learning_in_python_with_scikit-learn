from sklearn.svm import SVR
from types_config import Tmodel

from .Model import Model
from .RegressorMixin import RegressorMixin


class SupportVectorRegressionModel(RegressorMixin, Model[SVR, Tmodel]):
    """
    Epsilon-Support Vector Regression model.

    Very efficient for small to medium datasets.

    [To predict continuous target]
    """
    _estimator_class = SVR
