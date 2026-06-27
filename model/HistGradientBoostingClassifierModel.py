from sklearn.ensemble import HistGradientBoostingClassifier
from types_config import Tmodel

from .ClassifierMixin import ClassifierMixin
from .Model import Model


class HistGradientBoostingClassifierModel(ClassifierMixin,
                                          Model[HistGradientBoostingClassifier, Tmodel]):
    """
    Build a histogram gradient-boosting classifier model.

    [To predict discrete target]
    """
    _estimator_class = HistGradientBoostingClassifier
