from sklearn.linear_model import LinearRegression
from types_config import Tlinearmodel

from .LinearModel import LinearModel
from .RegressorMixin import RegressorMixin


class LinearRegressionModel(RegressorMixin, LinearModel[LinearRegression, Tlinearmodel]):
    """
    Linear regression model.

    [To predict continuous target]
    """
    _estimator_class = LinearRegression
