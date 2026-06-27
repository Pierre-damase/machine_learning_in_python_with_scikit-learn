from sklearn.ensemble import RandomForestClassifier
from types_config import Tmodel

from .ClassifierMixin import ClassifierMixin
from .Model import Model
from .TreeMixin import TreeMixin


class RandomForestClassifierModel(ClassifierMixin,
                                  TreeMixin,
                                  Model[RandomForestClassifier, Tmodel]):
    """
    Random forest classifier.

    [To predict discrete target]
    """
    _estimator_class = RandomForestClassifier
