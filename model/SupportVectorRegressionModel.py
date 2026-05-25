from .RegressionModel import RegressionModel

from sklearn.svm import SVR


class SupportVectorRegressionModel(RegressionModel[SVR]):
    """
    Epsilon-Support Vector Regression model.

    Very efficient for small to medium datasets.

    [To predict continuous target]
    """
    def __init__(self, kernel: str, degree: int):
        super().__init__()

        self.model = self._factory_model_initializer(SVR, kernel=kernel, degree=degree)
