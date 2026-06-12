from sklearn.ensemble import BaggingClassifier
from types_config import Tmodel

from .Model import Model


class BaggingClassifierModel(Model[BaggingClassifier, Tmodel]):
    """
    Bagging classifier in order to use bootstrapping algorithm.

    [To predict discrete target]
    """
    _estimator_class = BaggingClassifier
