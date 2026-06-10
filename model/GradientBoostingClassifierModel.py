from sklearn.ensemble import GradientBoostingClassifier
from types_config import Tmodel

from .Model import Model


class GradientBoostingClassifierModel(Model[GradientBoostingClassifier, Tmodel]):
    """
    Build a gradient-boosting classifier model.

    [To predict discrete target]
    """
    _estimator_class = GradientBoostingClassifier
