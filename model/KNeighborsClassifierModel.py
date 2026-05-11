from .Model import Model
from sklearn.neighbors import KNeighborsClassifier


class KNeighborsClassifierModel(Model):
    """
    K-nearest neighbors model.

    [To predict discrete target]

    Parameters
    ----------
    n_neighbors: number of neighbors
    """
    def __init__(self, n_neighbors: int):
        super().__init__()
        
        self.model = self._factory_model_initializer(KNeighborsClassifier, n_neighbors=n_neighbors)


    """Print model parameter at initialization."""
    def _print_model_initialization(self, model: KNeighborsClassifier):
        print(f"Build a {model.__class__.__name__} model with "
              f"{model.n_neighbors} iterations.")