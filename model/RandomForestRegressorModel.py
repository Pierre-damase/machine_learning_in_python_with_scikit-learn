from sklearn.ensemble import RandomForestRegressor
from types_config import Tmodel

from .Model import Model
from .RegressorMixin import RegressorMixin
from .TreeMixin import TreeMixin


class RandomForestRegressorModel(RegressorMixin, TreeMixin, Model[RandomForestRegressor, Tmodel]):
    """
    Random forest regressor.

    [To predict continuous target]
    """
    _estimator_class = RandomForestRegressor
