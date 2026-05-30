from sklearn.ensemble import HistGradientBoostingClassifier
from types_config import Tmodel

from .Model import Model


class GradientBoostingClassifierModel(Model[HistGradientBoostingClassifier, Tmodel]):
    """
    Build a gradient-boosting model.

    [To predict discrete target]
    """
    _estimator_class = HistGradientBoostingClassifier
