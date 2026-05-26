from sklearn.dummy import DummyClassifier

from .Model import Model

DUMMY_STRATEGY = ["constant", "most_frequent"]


class DummyClassifierModel(Model[DummyClassifier]):
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
    strategy: strategy to use to generate predictions
    constant: the explicit class to predict by the constant strategy
    """
    def __init__(self, **kwargs):
        super().__init__()

        self._validate_parameter(**kwargs)
        self.model = self._factory_model_initializer(
            DummyClassifier,
            strategy=kwargs["strategy"],
            constant=kwargs["constant"] if "constant" in kwargs.keys() else None
        )


    def _print_model_initialization(self, model) -> None:
        """Print model parameter at initialization."""
        print(f"Build a {model.__class__.__name__} model with "
              f"a constant strategy to predict {model.constant}.")


    def _validate_parameter(self, **kwargs):
        """Validate dummy classifier parameter."""
        if not "strategy" in kwargs.keys():
            raise Exception("The strategy to use to generate predictions is "
                            "required for dummy classifier.")

        if not kwargs["strategy"] in DUMMY_STRATEGY:
            raise Exception(f"Strategy {kwargs['strategy']} unknown.")

        if kwargs["strategy"] == "constant" and not "constant" in kwargs.keys():
            raise Exception("The explicit class to predict by the dummy "
                            "classifier is required for constant strategy.")
