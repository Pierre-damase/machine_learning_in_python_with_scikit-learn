from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from typing import TypeVar

import pandas as pd
import numpy as np
import time

T = TypeVar('T', 
            DecisionTreeRegressor,
            DummyClassifier, 
            KNeighborsClassifier, 
            LogisticRegression)

class Model:
    """Base class to build machine learning model."""
    def __init__(self):
        pass


    ############
    # ACCURACY #
    ############
    """Print model performance with training data."""
    def print_training_accuracy(self, y_train: pd.DataFrame, y_predicted: np.ndarray[any]) -> None:
        print(f"The train accuracy is {self.get_training_accuracy(y_train, y_predicted):.3f}.")

    """Get model performance with training data."""
    def get_training_accuracy(self, y_train: pd.DataFrame, y_predicted: np.ndarray[any]) -> float:
        return (y_train == y_predicted).mean()

    """Print model performance with testing data."""
    def print_testing_accuracy(self, model: T, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        print(f"The test accuracy is {self.get_testing_accuracy(model, x_test, y_test):.3f}")

    """Get model performance with testing data."""
    def get_testing_accuracy(self, model: T, x_test: pd.DataFrame, y_test: pd.Series) -> float:
        return model.score(x_test, y_test)


    ###########
    # STARTER #
    ###########
    """
    Performe a machine learning model.

    Parameters
    ----------
    x_train: training data
    
    x_test: testing data
    
    y_train: training target
    
    y_test: testing target
    """
    def start(self, x_train, x_test, y_train, y_test):
        # Train the model using training data
        self.train(self.model, x_train, y_train)

        # Predict target + chek model performance
        self.predict(self.model, x_train, x_test, y_train, y_test)

    """
    Start the pipeline to sequentially apply a list of transformers
    to prepreocess the data.
    """
    def start_pipeline(self, model: T, x_train: pd.DataFrame, y_train: pd.Series):
        # Start the pipeline 
        print(f"Start a simple pipeline with the given steps: {model.named_steps}.")
        start = time.time()
        model.fit(x_train, y_train)
        print(f"Pipeline over: {self.model[-1].n_iter_[0]} iterations in {time.time() - start:3f}s.")


    ###################
    # TRAIN & PREDICT #
    ###################
    """Train the model using training data."""
    def train(self, model: T, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        model.fit(x_train, y_train)

    """
    Predict the target and check model performance.

    Parameters
    ----------
    model: machine learning model

    x_train: training data
    
    x_test: testing data
    
    y_train: training target
    
    y_test: testing target
    """
    def predict(
            self, 
            model: T, 
            x_train: pd.DataFrame, 
            x_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series) -> np.ndarray[any]:
        # Predict target
        y_train_predicted = model.predict(x_train)

        # Check model performance with training data
        self.print_training_accuracy(y_train, y_train_predicted)

        # Check model performance with testing data
        self.print_testing_accuracy(model, x_test, y_test)


    ####################
    # CROSS VALIDATION #
    ####################
    """
    To performe a kfold cross-validation strategy.
    
    Parameter
    ---------
    x_data: the whole dataset

    y_data: the whole targets

    nb_fold: the cross-validation splitting strategy
    """
    def kfold_cross_validate(self, x_data: pd.DataFrame, y_data: pd.Series, nb_fold: int) \
        -> dict[str, np.array[float]]:
        return cross_validate(self.model, x_data, y_data, cv=nb_fold, error_score="raise")

    """Print the result (accuracy and fitting time) of a kfold cross-validation strategy."""
    def print_kfold_cross_validation_accuracy(self, scores: dict[str, np.array[float]]) -> None:
        print(f"Kfold cross-validation with K={len(scores['test_score'])}.\n"
               "The mean cross-validation accuracy is: "
              f"{scores['test_score'].mean():3f} ± {scores['test_score'].std():3f} "
              f" with an average fitting time of {scores['fit_time'].mean():3f}")


    ###############
    # INITIALIZER #
    ###############
    """Print model parameter at initialization."""
    def _print_model_initialization(self, model: T):
        print(f"Build a {model.__class__.__name__} model.")

    """Initialize a T model."""
    def _factory_model_initializer(self, model_class: type[T], **kwargs) -> T:
        model = model_class(**kwargs)
        self._print_model_initialization(model)
        return model

    """
    Initialize a pipeline.
    
    Parameter
    ---------
    *steps: list of estimator objects
    """
    def _factory_pipeline_initializer(self, *steps) -> T:
        # Column transformer parameters past ?
        # Parameter past as tuples of the form (transformer, columns)
        column_transformer_steps, left_over_steps = [], []
        for ele in steps:
            if isinstance(ele, tuple):
                column_transformer_steps.append(ele)
            else:
                left_over_steps.append(ele)
        
        # Initialize a column transformer if needed
        if column_transformer_steps:
            preprocessor = self._factory_transformer_initialize(*column_transformer_steps)
            left_over_steps.insert(0, preprocessor)

        # Construct a pipeline from the given estimators
        model = make_pipeline(*left_over_steps)
        print(f"Set up a pipeline to build a machine learning model.")
        return model

    """
    Initialize a column transformer 
    in order to handle categorical and numerical values.

    Parameter
    ---------
    *transformers: tuples of the form (transformer, columns)
    """
    def _factory_transformer_initialize(self, *transformers):
        return make_column_transformer(*transformers)