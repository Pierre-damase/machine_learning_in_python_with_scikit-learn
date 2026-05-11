from .Model import Model
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


class GradientBoostingClassifierModel(Model):
    """
    Build a gradient-boosting model.

    [To predict discrete target]
    """
    def __init__(self,
                 pipeline_steps: list[
                     HistGradientBoostingClassifier|
                     OrdinalEncoder]):
        super().__init__()

        self.pipeline_steps = pipeline_steps
        self.model:Pipeline = self._factory_pipeline_initializer(*pipeline_steps)