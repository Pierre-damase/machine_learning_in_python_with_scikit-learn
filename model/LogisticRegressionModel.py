from .Model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel(Model):
    """
    Logistic regression model.

    [To predict discrete target]

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
        super().__init__()
        
        self.pipeline_steps = pipeline_steps
        self.use_pipeline:bool = True if pipeline_steps else False
        self.model:LogisticRegression|Pipeline = self._model_initializer()


    """
    Performe a logistic regression.

    Parameters
    ----------
    x_train: training data
    
    x_test: testing data
    
    y_train: training target
    
    y_test: testing target
    """
    def start(self, x_train, x_test, y_train, y_test):
        if self.use_pipeline:
            # Start the pipeline to sequentially apply a list of transformers 
            self.start_pipeline(self.model, x_train, y_train)
        else:
            # Train the model using training data
            self.train(self.model, x_train, y_train)

        # Predict target + chek model performance
        self.predict(self.model, x_train, x_test, y_train, y_test)

    
    """Initialize logistic regression either within a pipeline or not."""
    def _model_initializer(self):
        if self.use_pipeline:
            # Initialize a simple pipeline to 
            #   1. Scale the training data
            #   2. Performe a logistic regression
            # model = self._factory_pipeline_initializer(StandardScaler(), LogisticRegression())
            model = self._factory_pipeline_initializer(*self.pipeline_steps)
        else:
            # Set up a logictic regression model
            model = self._factory_model_initializer(LogisticRegression, max_iter=1000)

        return model

    """Print model parameter at initialization."""
    def _print_model_initialization(self, model: LogisticRegression):
        print(f"Build a {model.__class__.__name__} model with "
              f"{model.max_iter} iterations.")