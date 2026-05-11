from .RegressionModel import RegressionModel
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


class DecisionTreeRegressorModel(RegressionModel):
    """
    Decision tree regressor model.
    
    [To predict continuous target]
    """
    def __init__(self):
        super().__init__()

        self.model:DecisionTreeRegressor = \
            self._factory_model_initializer(DecisionTreeRegressor, random_state=0)