from sklearn.ensemble import AdaBoostClassifier
from types_config import Tmodel

from .ClassifierMixin import ClassifierMixin
from .EnsembleMixin import EnsembleMixin
from .Model import Model


class AdaBoostClassifierModel(ClassifierMixin, EnsembleMixin, Model[AdaBoostClassifier, Tmodel]):
    """
    AdaBoost classifier in order to use boosting algorithm.

    [To predict discrete target]
    """
    _estimator_class = AdaBoostClassifier

    def get_estimator_weights(self) -> list[float]:
        """Get the weight of each estimator in the boosted ensemble."""
        return getattr(self.model, "estimator_weights_")

    def get_estimator_error(self) -> list[float]:
        """Get the error of each estimator in the boosted ensemble."""
        return getattr(self.model, "estimator_errors_")

    def print_estimator_weights(self) -> None:
        """Print the weight of each estimator in the boosted ensemble."""
        print("The weight of each estimator in the boosted ensemble: "
              f"{self.get_estimator_weights()}")

    def print_estimator_error(self) -> None:
        """Print the error of each estimator in the boosted ensemble."""
        print("The error of each estimator in the boosted ensemble: "
              f"{self.get_estimator_error()}")
