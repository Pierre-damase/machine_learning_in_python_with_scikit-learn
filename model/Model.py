from sklearn.compose import make_column_transformer
from sklearn.model_selection import (
    cross_validate,
    GridSearchCV,
    LearningCurveDisplay,
    RandomizedSearchCV,
    ShuffleSplit,
    ValidationCurveDisplay
)
from sklearn.pipeline import (
    Pipeline,
    make_pipeline
)
from .types import (
    Tcv,
    Tmodel,
    Tmodelwithpipeline,
    Tpipelinesteps,
    Tpreprocessor
)
from typing import Generic

import pandas as pd
import numpy as np
import numpy.typing as npt
import time


class Model(Generic[Tmodel]):
    model: Tmodel

    """Base class to build machine learning model."""
    def __init__(self,
                 pipeline_steps: list[Tpipelinesteps] = []):
        self.pipeline_steps = pipeline_steps
        self.use_pipeline: bool = self._use_pipeline(self.pipeline_steps)

    """
    Set up a column transformers to deal with numerical and categorical values:
    - Encoding of categorical variables
    - Scaling of numerical variables

    Parameter
    ---------
    kwargs['transformers']: tuples of the form (transformer, columns)

    kwargs['classifier']: classifier model to apply
    """
    @classmethod
    def build_pipeline_with_transformer(cls,
                                        transformers: list[Tpreprocessor],
                                        classifier: Tmodelwithpipeline):
        return cls(pipeline_steps=[*transformers, classifier])


    ############
    # ACCURACY #
    ############
    """Print model performance with training data."""
    def print_training_accuracy(self,
                                y_train: pd.Series,
                                y_predicted: npt.NDArray[np.generic]) -> None:
        print(f"The train accuracy is {self.get_training_accuracy(y_train, y_predicted):.3f}.")

    """Get model performance with training data."""
    def get_training_accuracy(self,
                              y_train: pd.Series,
                              y_predicted: npt.NDArray[np.generic]) -> float:
        return (y_train == y_predicted).mean()

    """Print model performance with testing data."""
    def print_testing_accuracy(self,
                               x_test: pd.DataFrame,
                               y_test: pd.Series) -> None:
        print("The test accuracy is "
              f"{self.get_testing_accuracy(x_test, y_test):.3f}")

    """Get model performance with testing data."""
    def get_testing_accuracy(self,
                             x_test: pd.DataFrame,
                             y_test: pd.Series) -> float:
        return self.model.score(x_test, y_test)


    #############
    # PARAMETER #
    #############
    """Get model hyperparameters."""
    def get_hyperparameters(self) -> dict[str, any]:
        return self.model.get_params()

    """Set model hyperparameters."""
    def set_hyperparameters(self, **parameters):
        self.model.set_params(**parameters)


    ###########
    # STARTER #
    ###########
    """
    Either perform a machine learning model or start a pipeline to sequentially
    apply a list of transformers to pre-process the data.

    Parameters
    ----------
    x_train: training data

    x_test: testing data

    y_train: training target

    y_test: testing target
    """
    def start(self,
              x_train: pd.DataFrame = pd.DataFrame(),
              x_test: pd.DataFrame = pd.DataFrame(),
              y_train: pd.Series = pd.Series(),
              y_test: pd.Series = pd.Series()):
        if x_train.empty and y_train.empty:
            raise Exception("A training data set and target are required.")

        if isinstance(self.model, Pipeline):
            # Start the pipeline
            print("Start a simple pipeline with the given steps: "
                  f"{self.model.named_steps}.")
            start = time.time()
            self.train(x_train, y_train)
            print(f"Pipeline over: {self.model[-1].n_iter_[0]} iterations "
                  f"in {time.time() - start:.3f}s.")
        else:
            # Simply train the model
            self.train(x_train, y_train)

        if not x_test.empty and not y_test.empty:
            # Predict target + chek model performance
            self.predict(x_train, x_test, y_train, y_test)


    ###################
    # TRAIN & PREDICT #
    ###################
    """Train the model using training data."""
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(x_train, y_train)

    """
    Predict the target and check model performance.

    Parameters
    ----------
    x_train: training data

    x_test: testing data

    y_train: training target

    y_test: testing target
    """
    def predict(self,
                x_train: pd.DataFrame,
                x_test: pd.DataFrame,
                y_train: pd.Series,
                y_test: pd.Series) -> None:
        # Predict target
        y_train_predicted = self.model.predict(x_train)

        # Check model performance with training data
        self.print_training_accuracy(y_train, y_train_predicted)

        # Check model performance with testing data
        self.print_testing_accuracy(x_test, y_test)


    ####################
    # CROSS VALIDATION #
    ####################
    """Print cross validation result."""
    def print_cross_validate(self, scores: dict[str, npt.NDArray[np.float64]]) -> None:
        # Print testing error
        mean_score, std_score = \
            self.get_cross_validate_mean_and_std(scores['test_score'])
        # print(f"Shuffle split cross-validation with n={n_splits}.\n"
        print("The mean cross-validated testing error is: "
             f"{mean_score:.3f} ± {std_score:.3f}"
             f" with an average fitting time of {scores['fit_time'].mean():.3f}")

        # Print training error
        if "train_score" in scores.keys():
            mean_score, std_score = \
                self.get_cross_validate_mean_and_std(scores['train_score'])
            print("The mean cross-validated training error is "
                  f"{mean_score:.3f} ± {std_score:.3f}")


    ##########################
    # KFLOD CROSS VALIDATION #
    ##########################
    """
    To performe a kfold cross-validation strategy.

    Parameter
    ---------
    x_data: the whole dataset

    y_data: the whole targets

    nb_fold: the cross-validation splitting strategy

    scoring: strategy to evaluate the performance of the estimator across cross-
             validation splits. Default None.
    """
    def kfold_cross_validate(self,
                             x_data: pd.DataFrame,
                             y_data: pd.Series,
                             nb_fold: int = 5,
                             scoring: str = "",
                             return_train_score: bool = False) \
                             -> dict[str, npt.NDArray[np.float64]]:
        return cross_validate(self.model,
                              x_data,
                              y_data,
                              cv=nb_fold,
                              scoring=scoring if scoring else None,
                              return_train_score=return_train_score,
                              error_score="raise",
                              n_jobs=2)

    """
    Print the result (accuracy and fitting time) of a kfold cross-validation
    strategy.
    """
    def print_kfold_cross_validation_accuracy(
            self,
            scores: dict[str, npt.NDArray[np.float64]]) -> None:
        print(f"Kfold cross-validation with K={len(scores['test_score'])}.")
        self.print_cross_validate(scores)


    """Get the mean/std of a cross_validation strategy."""
    def get_cross_validate_mean_and_std(self,
                                        scores: npt.NDArray[np.float64]) \
                                        -> tuple[float, float]:
        return scores.mean(), scores.std()


    ##################################
    # SHUFFLE SPLIT CROSS VALIDATION #
    ##################################
    """Set up a shuffle split cross-validation strategy."""
    def shuffle_split_cv_generator(self, n_splits: int, test_size: float = 0.2):
        return ShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=0
        )

    """
    To performe a shuffle split cross-validation strategy.

    Scikit-learn allow the use of any metric, like 'mean_absolute_error' into
    a score to be used in cross validate. In order to do so, pass a string of
    the error metric with and additional neg_ such as 'neg_mean_absolute_error'.

    Parameter
    ---------
    x_data: the whole dataset

    y_data: the whole targets

    n_splits: number of re-shuffling & splitting iterations.
    """
    def shuffle_split_cross_validate(self,
                                     x_data: pd.DataFrame,
                                     y_data: pd.Series,
                                     n_splits: int,
                                     scoring: str = None,
                                     test_size: float = 0.2,
                                     return_train_score: bool = False) \
                                     -> dict[str, npt.NDArray[np.float64]]:
        # Set the cross-validation strategy and performe it
        results = cross_validate(self.model,
                                 x_data,
                                 y_data,
                                 cv=self.shuffle_split_cv_generator(n_splits, test_size),
                                 scoring=scoring,
                                 return_train_score=return_train_score,
                                 error_score="raise",
                                 n_jobs=2)

        # Revert the negation apply to the error metric to get the actual error
        if scoring and scoring.startswith("neg_"):
            results["test_score"] = -results["test_score"]
            if return_train_score:
                results["train_score"] = -results["train_score"]

        return results

    """
    Print the result (accuracy and fitting time) of a shuffle split
    cross-validation strategy.
    """
    def print_shuffle_split_cross_validation_accuracy(
            self,
            scores: dict[str, npt.NDArray[np.float64]]) -> None:
        print(f"Shuffle split cross-validation with n={len(scores['test_score'])}.")
        self.print_cross_validate(scores)


    ###############
    # INITIALIZER #
    ###############
    """Print model parameter at initialization."""
    def _print_model_initialization(self, model: Tmodel):
        print(f"Build a {model.__class__.__name__} model.")

    """Initialize a T model."""
    def _factory_model_initializer(self, model_class: type[Tmodel], **kwargs) -> Tmodel:
        model = model_class(**kwargs)
        self._print_model_initialization(model)
        return model

    """
    Initialize a pipeline.

    Parameter
    ---------
    *steps: list of estimator objects
    """
    def _factory_pipeline_initializer(self, *steps) -> Pipeline:
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
            preprocessor = \
                self._factory_transformer_initialize(*column_transformer_steps)
            left_over_steps.insert(0, preprocessor)

        # Construct a pipeline from the given estimators
        print(f"Set up a pipeline to build a machine learning model.")
        return make_pipeline(*left_over_steps)

    """Some models can either be used with or without a pipeline."""
    def _use_pipeline(self, pipeline_steps = []):
        return True if pipeline_steps else False

    """
    Initialize a column transformer
    in order to handle categorical and numerical values.

    Parameter
    ---------
    *transformers: tuples of the form (transformer, columns)
    """
    def _factory_transformer_initialize(self, *transformers):
        return make_column_transformer(*transformers)


    ####################
    # VALIDATION CURVE #
    ####################
    """
    Use validation curve to try out hyperparameters.

    Pass either a classifier or a pipeline (parameter self.model)
    to ValidationCurveDisplay.from_estimator
    """
    def compute_validation_curve(self,
                                 x_data: pd.DataFrame,
                                 y_data: pd.Series,
                                 scoring: str,
                                 score_name: str,
                                 cv: Tcv = None,
                                 negate_score: bool = False,
                                 **kwargs) -> ValidationCurveDisplay:
        return ValidationCurveDisplay.from_estimator(
            self.model,
            x_data,
            y_data,
            cv=cv,
            scoring=scoring,
            score_name=score_name,
            param_name=kwargs["param_name"],
            param_range=kwargs["param_range"],
            negate_score=negate_score,
            std_display_style="fill_between",
            n_jobs=2
        )


    ##################
    # LEARNING CURVE #
    ##################
    """
    Use learning curve to try out various training set size.

    Pass either a classifier or a pipeline (parameter self.model)
    to ValidationCurveDisplay.from_estimator
    """
    def compute_learning_curve(self,
                               x_data: pd.DataFrame,
                               y_data: pd.Series,
                               cv: Tcv,
                               scoring: str,
                               score_name: str,
                               negate_score: bool = False,
                               **kwargs) -> LearningCurveDisplay:
        return LearningCurveDisplay.from_estimator(
            self.model,
            x_data,
            y_data,
            train_sizes=kwargs["train_sizes"],
            cv=cv,
            score_type="both", # both train and test errors
            scoring=scoring,
            score_name=score_name,
            negate_score=negate_score,
            std_display_style="fill_between",
            n_jobs=2
        )


    ################################
    # MANUAL HYPERPARAMETER TUNING #
    ################################
    """
    Use recursion to build a nested for loop for each hyperparameters in
    order to perform cross-validation and tuned them.
    """
    def _hyperparameter_loop_recurcive(self,
                                       data: pd.DataFrame,
                                       targets: pd.Series,
                                       scores: dict[str, list[float|int]],
                                       index: int,
                                       **hyperparameters):
        if index < len(hyperparameters):
            name = list(hyperparameters.keys())[index]
            for value in hyperparameters[name]:
                # Set hyperparameter value
                self.set_hyperparameters(**{name: value})

                # Recursive call
                self._hyperparameter_loop_recurcive(data,
                                                    targets,
                                                    scores,
                                                    index+1,
                                                    **hyperparameters)
        else:
            # Compute cross-validation
            results = self.kfold_cross_validate(data,
                                                targets,
                                                5,
                                                return_train_score=True)

            # Save hyperparameter
            for k, v in {
                k: v for k, v in self.get_hyperparameters().items() if k in hyperparameters.keys()
            }.items() :
                scores[k].append(v)

            # Save cross-validation result
            scores["testing_mean"].append(results["test_score"].mean())
            scores["testing_std"].append(results["test_score"].std())
            scores["training_mean"].append(results["train_score"].mean())
            scores["training_std"].append(results["train_score"].std())

    """Manually tuned hyperparameters."""
    def manual_hyperparameter_tuning(self,
                                     data: pd.DataFrame,
                                     targets: pd.Series,
                                     **hyperparameters) \
                                     -> dict[str, list[float|int]]:
        # Build scores dict to keep track the mean/std testing and training error
        scores: dict[str, list[float|int]] = {k: [] for k in hyperparameters.keys()}
        scores.update(testing_mean=[], testing_std=[], training_mean=[], training_std=[])

        # Hyperparameter tuning
        self._hyperparameter_loop_recurcive(data,
                                            targets,
                                            scores,
                                            0,
                                            **hyperparameters)
        return scores


    #########################
    # HYPERPARAMETER TUNING #
    #########################
    """
    Hyperparameter tuning by grid-search on the training set.

    Some limitation:
      - does not scale well with the number of hyperparameters to tune
      - the grid imposes a regularity during the search which might miss better parameter
        values between two consecutive values on the grid.
      - hyperparameters value must be specified explicitly
    """
    def _grid_search_cv(self,
                       param_grid: dict[str, list[float|int]],
                       cv: int) -> GridSearchCV:
        return GridSearchCV(self.model,
                            param_grid=param_grid,
                            cv=cv,
                            n_jobs=2,
                            return_train_score=True)

    """
    Hyperparameter tuning by randomized-search on the training set.

    Randomly generate the parameters candidate
    """
    def _randomized_search_cv(self,
                             param_distribution: dict[str, list[float|int]],
                             n_iter: int,
                             cv: int) -> RandomizedSearchCV:
        return RandomizedSearchCV(self.model,
                                  param_distributions=param_distribution,
                                  n_iter=n_iter,
                                  cv=cv,
                                  verbose=1,
                                  return_train_score=True)

    """Print scores as a DataFrame."""
    def _print_results_as_dataframe(self, results: dict[str, list[float]]) -> None:
        columns = [
            "param_", "mean_test_score", "std_test_score", "rank_test_score",
            "mean_train_score", "std_train_score"
        ]
        scores = {
            k.rsplit("__")[-1]: v
            for k, v in results.items()
            if [col for col in columns if col in k] # Any hyperparameter starts with param_
        }
        cv_results = pd.DataFrame(scores).sort_values("mean_test_score", ascending=False)
        print(cv_results)

    """
    Parameter
    ---------
    automated_search_cv: which method to apply for hyperparameter tuning

    hyperparameters: hyperparameter values to try out for model tuning

    data: list containing the training and testing set (data and targets)
    """
    def automated_search_cross_validation(self,
                                          search_class: type[GridSearchCV|RandomizedSearchCV],
                                          hyperparameters: dict[str, list[float|int]],
                                          x_train: pd.DataFrame,
                                          x_test: pd.DataFrame,
                                          y_train: pd.Series,
                                          y_test: pd.Series,
                                          verbose: bool = False,
                                          **kwargs):
        # 1. Build an automated-search cross-validation
        if search_class is GridSearchCV:
            search_model = self._grid_search_cv(hyperparameters, cv=kwargs["cv"])
        elif search_class is RandomizedSearchCV:
            search_model = self._randomized_search_cv(
                hyperparameters, cv=kwargs["cv"], n_iter=kwargs["n_iter"]
            )
        else:
            raise Exception(f"Unsupported search methods to tune hyperparameters.")

        # 2. Train the model with the best set of hyperparameters
        search_model.fit(x_train, y_train)

        # 3. Display the optimum hyperparameters
        print(f"The optimum set of hyperparameters is {search_model.best_params_}")

        if verbose:
            self._print_results_as_dataframe(search_model.cv_results_)

        # 4. Compute the generalization performance of the model
        accuracy = search_model.score(x_test, y_test)
        print(f"The test accuracy score of the grid-search pipeline is {accuracy:.3f}")
