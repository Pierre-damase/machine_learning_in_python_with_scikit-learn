from sklearn.ensemble import BaggingRegressor
from types_config import Tmodel

from .Model import Model
from .RegressorMixin import RegressorMixin


class BaggingRegressorModel(RegressorMixin, Model[BaggingRegressor, Tmodel]):
    """
    Bagging regressor in order to use bootstrapping algorithm for prediction.

    [To predict continuous target]
    """
    _estimator_class = BaggingRegressor
