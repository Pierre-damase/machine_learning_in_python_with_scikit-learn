from .Model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel(Model[LogisticRegression | Pipeline]):
    """
    Logistic regression model.

    [To predict discrete target]

    C is the hyperparameter of this kind of model.

    Parameters
    ----------
    pipeline_steps: list of estimators object to build pipeline

                    to set up a column transformer pass first parameters
                    as tuples of the form (transformer, columns)
    """
    def __init__(self,
                 pipeline_steps: list[
                     LogisticRegression|
                     OneHotEncoder|
                     StandardScaler]|
                     tuple[OneHotEncoder|StandardScaler, any] = []):
        super().__init__(pipeline_steps=pipeline_steps)

        self.model = self._model_initializer()


    def _model_initializer(self):
        """Initialize logistic regression either within a pipeline or not."""
        if self.use_pipeline:
            # Initialize a simple pipeline to
            #   1. Scale the training data
            #   2. Performe a logistic regression
            return self._factory_pipeline_initializer(*self.pipeline_steps)

        # Set up a logictic regression model
        return self._factory_model_initializer(LogisticRegression, max_iter=1000)

    def _print_model_initialization(self, model) -> None:
        """Print model parameter at initialization."""
        print(f"Build a {model.__class__.__name__} model with "
              f"{model.max_iter} iterations.")


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
