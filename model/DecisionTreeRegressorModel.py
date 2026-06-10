import pandas as pd
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
