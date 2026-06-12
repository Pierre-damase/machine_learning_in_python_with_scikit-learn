from sklearn.ensemble import HistGradientBoostingClassifier
from types_config import Tmodel

from .Model import Model


class HistGradientBoostingClassifierModel(Model[HistGradientBoostingClassifier, Tmodel]):
    """
    Build a histogram gradient-boosting classifier model.

    [To predict discrete target]
    """
    _estimator_class = HistGradientBoostingClassifier
