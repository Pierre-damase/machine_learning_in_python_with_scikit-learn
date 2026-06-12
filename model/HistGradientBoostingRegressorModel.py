from sklearn.ensemble import HistGradientBoostingRegressor
from types_config import Tmodel

from .Model import Model
from .RegressorMixin import RegressorMixin


class HistGradientBoostingRegressorModel(RegressorMixin,
                                         Model[HistGradientBoostingRegressor, Tmodel]):
    """
    To build histogram gradient-boosting model.

    [To predict continuous target]
    """
    _estimator_class = HistGradientBoostingRegressor
