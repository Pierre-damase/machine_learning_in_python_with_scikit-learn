from typing import Literal

import pandas as pd

from .DecisionBoundaryMixin import DecisionBoundaryMixin


class EnsembleMixin(DecisionBoundaryMixin):
    """Mixin class to add method specific to ensemble model."""
    def decision_boundary_display_for_estimators(
            self,
            data: pd.DataFrame,
            x_data: pd.DataFrame,
            response_method: Literal["predict", "predict_proba"],
            hue: str,
            cmap: Literal["Blues", "RdBu", "RdBu_r", "tab10"] = "RdBu",
            **kwargs) -> None:
        """
        For ensemble model, it's possible to plot the decision boundary of each fitted
        sub-estimator.
        """
        # Get model and the collection of fitted sub-estimators
        model = getattr(self, "model")
        estimators = getattr(model, "estimators_")

        # Plot a decision boundary for each fitted sub-estimators
        for i in range(len(estimators)):
            self.decision_boundary_display(data,
                                           x_data,
                                           response_method,
                                           hue,
                                           cmap = cmap,
                                           title=f"Fitted sub-estimator #{i}",
                                           estimator=estimators[i],
                                           **kwargs)
