import numpy as np
import numpy.typing as npt
from sklearn.linear_model import RidgeCV
from types_config import CvResults, Tlinearmodel

from .LinearModel import LinearModel
from .RegressorMixin import RegressorMixin


class RidgeRegressionModelCV(RegressorMixin, LinearModel[RidgeCV, Tlinearmodel]):
    """
    RidgeCV regression model to perform linear regression with regularizzation and alpha parameter
    tuning.

    [To predict continous target]
    """
    _estimator_class = RidgeCV

    def print_best_alpha_from_cv_estimator(self, scores: CvResults) -> None:
        """Print the optimal range for alpha."""
        alphas = self.get_best_alpha_from_cv_estimator(scores)
        print(f"Min optimal alpha: {np.min(alphas):.3f} & max optimal alpha {np.max(alphas):.3f}")

    def get_best_alpha_from_cv_estimator(self, scores: CvResults) -> npt.NDArray[np.float64]:
        """
        Get alpha from inner cross-validation estimator. In order to get the best possible value
        for alpha.

        Although, the optimal regularization strength is not necessarily the same among all the cv
        iterations, it's a common practice to choose the best alpha between the minimum and maximum
        values from all the cv iterations.
        """
        return [getattr(ele[-1], "alpha_") for ele in self.get_cv_estimator(scores)]
