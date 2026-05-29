from sklearn.pipeline import Pipeline
from types_config import Tpipelinesteps

from .Model import Model
from .RegressorMixin import RegressorMixin


class KNeighborsRegressorModel(RegressorMixin, Model[Pipeline]):
    """
    K-nearest neighbors regressor.

    [To predict continuous target]
    """
    def __init__(self,
                 pipeline_steps: list[Tpipelinesteps] = []):
        super().__init__(pipeline_steps=pipeline_steps)

        self.model = self._factory_pipeline_initializer(*self.pipeline_steps)
