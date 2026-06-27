from sklearn.neighbors import KNeighborsClassifier
from types_config import Tmodel

from .ClassifierMixin import ClassifierMixin
from .Model import Model


class KNeighborsClassifierModel(ClassifierMixin, Model[KNeighborsClassifier, Tmodel]):
    """
    K-nearest neighbors classifier.

    [To predict discrete target]
    """
    _estimator_class = KNeighborsClassifier
    _default_params = {"n_neighbors": 5}
