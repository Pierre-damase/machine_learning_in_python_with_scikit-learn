from sklearn.pipeline import Pipeline
from types_config import Tpipelinesteps

from .Model import Model


class GradientBoostingClassifierModel(Model[Pipeline]):
    """
    Build a gradient-boosting model.

    [To predict discrete target]
    """
    def __init__(self,
                 pipeline_steps: list[Tpipelinesteps]):
        super().__init__(pipeline_steps=pipeline_steps)

        self.model = self._factory_pipeline_initializer(*self.pipeline_steps)
