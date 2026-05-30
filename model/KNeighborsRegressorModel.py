from sklearn.neighbors import KNeighborsRegressor
from types_config import Tlinearmodel

from .Model import Model
from .RegressorMixin import RegressorMixin


class KNeighborsRegressorModel(RegressorMixin, Model[KNeighborsRegressor, Tlinearmodel]):
    """
    K-nearest neighbors regressor.

    [To predict continuous target]
    """
    _estimator_class = KNeighborsRegressor
