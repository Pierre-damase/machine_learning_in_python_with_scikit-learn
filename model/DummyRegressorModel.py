from sklearn.dummy import DummyRegressor
from types_config import Tmodel

from .Model import Model
from .RegressorMixin import RegressorMixin

DUMMY_STRATEGY = ["mean"]

class DummyRegressorModel(RegressorMixin, Model[DummyRegressor, Tmodel]):
    """
    Build a dummy regressor model, i.e a simple baseline regressor that makes predictions using
    simple rules and that does not take into account the input features. In other word, such models
    are useful to confirme that the input features hold some informations and are useful to predict
    the target.

    This classifier serves as as simple baseline to compare against other more complex classifier.

    The specific behavior of the baseline is selected with the 'strategy' parameter. The default
    strategy is 'mean', which means that the model always predict the mean target value computed on
    the training target set.

    [To predict discrete target]

    Parameters
    ----------
    strategy: strategy to use to generate predictions
    """
    _estimator_class = DummyRegressor
    _default_params = {"strategy": "mean"}
