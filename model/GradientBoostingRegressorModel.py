from sklearn.ensemble import GradientBoostingRegressor
from types_config import Tmodel

from .Model import Model
from .RegressorMixin import RegressorMixin


class GradientBoostingRegressorModel(RegressorMixin, Model[GradientBoostingRegressor, Tmodel]):
    """
    Build a gradient-boosting regressor model.

    [To predict continuous target]
    """
    _estimator_class = GradientBoostingRegressor
