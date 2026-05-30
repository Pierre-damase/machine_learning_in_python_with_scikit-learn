from sklearn.linear_model import Ridge
from types_config import Tlinearmodel

from .LinearModel import LinearModel
from .RegressorMixin import RegressorMixin


class RidgeRegressionModel(RegressorMixin, LinearModel[Ridge, Tlinearmodel]):
    """
    Ridge regression model to perform regularizzation.

    [To predict continous target]
    """
    _estimator_class = Ridge
