from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .RegressionModel import RegressionModel
from sklearn.neighbors import KNeighborsRegressor


class KNeighborsRegressorModel(RegressionModel[Pipeline]):
    """
    K-nearest neighbors regressor.

    [To predict continuous target]
    """
    def __init__(self,
                 pipeline_steps: list[KNeighborsRegressor
                                      | StandardScaler] = []):
        super().__init__(pipeline_steps=pipeline_steps)

        self.model = self._factory_pipeline_initializer(*self.pipeline_steps)
