from .Model import Model
from sklearn.dummy import DummyClassifier


class DummyClassifierModel(Model):
    """
    Build a dummy classifier model, i.e. a simple baseline classifier that 
    always predict the same class.

    This classifier serves as as simple baseline to compare against other more
    complex classifier.

    The specific behavior of the baseline is selected with the 'strategy'
    parameter.

    [To predict discrete target]

    Parameters
    ----------
    constant: the class to predict
    """
    def __init__(self, constant: str):
        super().__init__()
        self.model = self._factory_model_initializer(DummyClassifier, strategy="constant", constant=constant)


    """Print model parameter at initialization."""
    def _print_model_initialization(self, model: DummyClassifier):
        print(f"Build a {model.__class__.__name__} model with "
              f"a constant strategy to predict {model.constant}.")