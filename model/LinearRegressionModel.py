from .RegressionModel import RegressionModel

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


class LinearRegressionModel(RegressionModel[LinearRegression | Pipeline]):
    """
    Linear regression model.

    [To predict continuous target]
    """
    def __init__(self,
                 pipeline_steps: list[LinearRegression] = []):
        super().__init__(pipeline_steps)

        self.model = self._model_initializer()


    def _model_initializer(self):
        """Initialize linear regression either within a pipeline or not."""
        if self.use_pipeline:
            # Initialize a pipeline
            return self._factory_pipeline_initializer(*self.pipeline_steps)

        # Set up a logictic regression model
        return self._factory_model_initializer(LinearRegression)

