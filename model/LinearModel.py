import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline
from types_config import (AcceptEstimatorType, AcceptPreprocessorType,
                          CvResults, Tlinearestimator, Tlinearmodel)

from .Model import Model


class LinearModel(Model[Tlinearestimator, Tlinearmodel]):
    ###############
    # COEFFICIENT #
    ###############
    def print_weights(self, features: list[str]) -> None:
        """
        Print the coefficient of the linear regression model, i.e. the associated weight of each
        feature.
        """
        coefficients = self.get_weights()
        for i in range(len(features)):
            print(f"The weight associated to {features[i]} is {coefficients[i]:.3f}")

    def get_weights(self) -> list[float]:
        """
        Get the coefficient of the linear regression model, i.e. the associated weight of each
        feature.

        self.model[-1] allow to access to the last step of the pipeline. Then it's possible to get
        attribute of that step such as coef_.
        """
        return self.model[-1].coef_[0]

    def _get_coef_from_pipeline(self,
                               estimators: list[AcceptEstimatorType | AcceptPreprocessorType]) \
                               -> tuple[list[str], list[npt.NDArray[np.float64]]]:
        """Get estimators' coefficients for model use within a pipeline."""
        # Get features. The last element of the pipeline is the actual classifier/regressor model
        # but no feature names defined at this step.
        features = [f.split('__')[-1] for f in estimators[0][:-1].get_feature_names_out()]

        # Get associated weight fro each feature. The last element of the pipeline is the model.
        coefficients = []
        for estimator in estimators:
            if self.is_regressor:
                coefficients.append(getattr(estimator[-1], "coef_"))
            else:
                coefficients.append(getattr(estimator[-1], "coef_")[0])

        return features, coefficients

    def _get_coef_from_model(self,
                            estimators: list[AcceptEstimatorType | AcceptPreprocessorType]) \
                            -> tuple[list[str], list[npt.NDArray[np.float64]]]:
        """Get estimators' coefficients for model use without a pipeline."""
        # Get features
        features = list(getattr(estimators[0], "feature_names_in_"))

        # Get associated weight fro each feature
        coefficients = []
        for estimator in estimators:
            if self.is_regressor:
                coefficients.append(getattr(estimator, "coef_"))
            else:
                coefficients.append(getattr(estimator, "coef_")[0])

        return features, coefficients

    def get_coefficients(self, scores: CvResults) -> dict[str, list[float]]:
        """Get coefficients of a logistic regression."""
        estimators = self.get_cv_estimator(scores)
        if isinstance(estimators[0], Pipeline):
            features, coefficients = self._get_coef_from_pipeline(estimators)
        else:
            features, coefficients = self._get_coef_from_model(estimators)

        return {k: list(v) for k, v in zip(features, zip(*coefficients))}

    def print_nlargest_coefficient(self,
                                   coefficients: dict[str, list[float]],
                                   n_largest: int = 5) -> None:
        """
        Return the n-largest coefficient for a linear model.

        Parameters
        ----------
        coef: a dictionary with key as coefficient name and value a list of coefficent values, one
        for each cross-validation step.

        n_largest: number of coefficient to print
        """
        data = pd.DataFrame.from_dict(coefficients)
        print("The nlargest absolute value of the coefficients are "
              f"{data.max().nlargest(n_largest).to_dict()}")
