from sklearn.pipeline import Pipeline
from types_config import CvResults, Tlinearmodel, Tpipelinesteps

from .Model import Model


class LinearModel(Model[Tlinearmodel]):
    """Linear model."""
    def __init__(self, pipeline_steps: list[Tpipelinesteps] = []):
        super().__init__(pipeline_steps)


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

    def get_coefficients(self, scores: CvResults) -> dict[str, list[float]]:
        """Get coefficients of a logistic regression."""
        data, features = [], []
        for estimator in self.get_cv_estimator(scores):
            if isinstance(estimator, Pipeline):
                if not features:
                    # 1st element is the transformer. Remove the transformer name.
                    features.extend([
                        f.split('__')[-1] for f in estimator[0].get_feature_names_out()
                    ])

                # Last element of the pipeline is the model
                if self.is_regressor:
                    data.append(estimator[-1].coef_)
                else:
                    data.append(estimator[-1].coef_[0])
            else:
                if not features:
                    features.extend(estimator.feature_names_in_)
                if self.is_regressor:
                    data.append(estimator.coef_)
                else:
                    data.append(estimator.coef_[0])

        return {k: list(v) for k, v in zip(features, zip(*data))}
