from sklearn.dummy import DummyClassifier
from types_config import Tmodel

from .ClassifierMixin import ClassifierMixin
from .Model import Model

DUMMY_STRATEGY = ["constant", "most_frequent"]


class DummyClassifierModel(ClassifierMixin, Model[DummyClassifier, Tmodel]):
    """
    Build a dummy classifier model, i.e. a simple baseline classifier that always predict the same
    class and that does not take into account the input features. In other word, such models are
    useful to confirme that the input features hold some informations and are useful to predict the
    target.

    This classifier serves as as simple baseline to compare against other more complex classifier.

    The specific behavior of the baseline is selected with the 'strategy' parameter. The default
    strategy is 'constant', which means thaht the model always predict the same class (the most
    present).

    [To predict discrete target]

    Parameters
    ----------
    strategy: strategy to use to generate predictions
    constant: the explicit class to predict by the constant strategy
    """
    _estimator_class = DummyClassifier
    _default_params = {"strategy": "constant"}
