from sklearn.ensemble import GradientBoostingClassifier
from types_config import Tmodel

from .ClassifierMixin import ClassifierMixin
from .Model import Model


class GradientBoostingClassifierModel(ClassifierMixin, Model[GradientBoostingClassifier, Tmodel]):
    """
    Build a gradient-boosting classifier model.

    [To predict discrete target]
    """
    _estimator_class = GradientBoostingClassifier
