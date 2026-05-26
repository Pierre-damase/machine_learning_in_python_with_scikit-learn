from .Model import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class KNeighborsClassifierModel(Model[KNeighborsClassifier | Pipeline]):
    """
    K-nearest neighbors classifier.

    [To predict discrete target]

    Parameters
    ----------
    n_neighbors: number of neighbors
    """
    def __init__(self,
                 n_neighbors: int = 5,
                 pipeline_steps: list[KNeighborsClassifier
                                      | StandardScaler] = []):
        super().__init__(pipeline_steps=pipeline_steps)

        self.model = self._model_initializer(n_neighbors)


    def _print_model_initialization(self, model):
        """Print model parameter at initialization."""
        print(f"Build a {model.__class__.__name__} model with "
              f"{model.n_neighbors} iterations.")


    def _model_initializer(self, n_neighbors: int):
        """Initialize a K-nearest neighbors model either within a pipeline or not."""
        if self.use_pipeline:
            # Initialize a simple pipeline to
            #   1. Scale the training data
            #   2. Performe a logistic regression
            return self._factory_pipeline_initializer(*self.pipeline_steps)
        else:
            # Set up a logictic regression model
            return self._factory_model_initializer(KNeighborsClassifier, n_neighbors=n_neighbors)
