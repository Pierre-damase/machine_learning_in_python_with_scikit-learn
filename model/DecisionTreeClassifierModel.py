from sklearn.tree import DecisionTreeClassifier
from types_config import Tmodel

from .ClassifierMixin import ClassifierMixin
from .DecisionBoundaryMixin import DecisionBoundaryMixin
from .Model import Model
from .TreeMixin import TreeMixin


class DecisionTreeClassifierModel(ClassifierMixin,
                                  DecisionBoundaryMixin,
                                  TreeMixin,
                                  Model[DecisionTreeClassifier, Tmodel]):
    """
    Decision tree classifier model.

    [To predict discrete targets]
    """
    _estimator_class = DecisionTreeClassifier
