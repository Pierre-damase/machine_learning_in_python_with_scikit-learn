import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (GridSearchCV, LearningCurveDisplay,
                                     RandomizedSearchCV, ShuffleSplit,
                                     ValidationCurveDisplay, cross_validate)
from sklearn.pipeline import Pipeline, make_pipeline
from types_config import (AcceptModelType, CvResults, Tclassifier,
                          Tclassifierwithpipeline, Tcv, Tpipelinesteps,
                          Tpreprocessor, Tregressor, Tregressorwithpipeline)

from .RegressorMixin import RegressorMixin

# Expected parameter for a given search class use to tune hyperparameters
SEARCH_EXPECTED_PARAM = {
    GridSearchCV.__name__: ["cv"],
    RandomizedSearchCV.__name__: ["cv", "n_iter"]
}


class Model[Tmodel]():
    model: Tmodel # either a classifier or a regressor or a pipeline

    def __init__(self,
                 pipeline_steps: list[Tpipelinesteps] = []):
        """Base class to build machine learning model."""
        self.pipeline_steps = pipeline_steps
        self.use_pipeline: bool = self._use_pipeline(self.pipeline_steps)

    @classmethod
    def build_pipeline_with_transformer(cls,
                                        transformers: list[Tpreprocessor],
                                        model: Tclassifierwithpipeline|Tregressorwithpipeline):
        """
        Set up a column transformers to deal with numerical and categorical values:
        - Encoding of categorical variables
        - Scaling of numerical variables

        Parameter
        ---------
        transformers: a list that is either made of transformers which are applied on the whole
        data or tuples of the form (transformer, columns) with transformer the given transformation
        to apply and columns the data to transform.

        Columns is either got with the make_column_selector class of scikit-learn or by using the
        columns property of a pandas DataFrame (in this cas it's required to convert the
        pandas.Index into a list of string)

        model: the model to apply, either a classifier or a regressor

        output: to transform the transformer output into a DataFrame. By default None.
        """
        return cls(pipeline_steps=[*transformers, model])


    ###############
    # INITIALIZER #
    ###############
    def _print_model_initialization(self, model: Tclassifier|Tregressor):
        """Print model parameter at initialization."""
        print(f"Build a {model.__class__.__name__} model.")

    def _factory_model_initializer(self,
                                   model_class: type[Tclassifier|Tregressor], **kwargs) \
                                   -> Tclassifier|Tregressor:
        """Initialize a T model."""
        model = model_class(**kwargs)
        self._print_model_initialization(model)
        return model

    def _factory_pipeline_initializer(self, *steps) -> Pipeline:
        """
        Initialize a pipeline.

        Parameter
        ---------
        *steps: list of estimator objects
        """
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

    def _use_pipeline(self, pipeline_steps = []):
        """Some models can either be used with or without a pipeline."""
        return True if pipeline_steps else False

    def _factory_transformer_initialize(self, *transformers):
        """
        Initialize a column transformer
        in order to handle categorical and numerical values.

        Parameter
        ---------
        *transformers: tuples of the form (transformer, columns)
        """
        return make_column_transformer(*transformers)


    #############
    # REGRESSOR #
    #############
    @property
    def is_regressor(self) -> bool:
        return isinstance(self, RegressorMixin)


    ###########
    # STARTER #
    ###########
    def start(self,
              x_train: pd.DataFrame = pd.DataFrame(),
              x_test: pd.DataFrame = pd.DataFrame(),
              y_train: pd.Series = pd.Series(),
              y_test: pd.Series = pd.Series()):
        """
        Either perform a machine learning model or start a pipeline to sequentially apply a list
        of transformers to pre-process the data.

        Parameters
        ----------
        x_train: training data

        x_test: testing data

        y_train: training target

        y_test: testing target
        """
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
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model using training data."""
        self.model.fit(x_train, y_train)

    def predict(self,
                x_train: pd.DataFrame,
                x_test: pd.DataFrame,
                y_train: pd.Series,
                y_test: pd.Series) -> None:
        """
        Predict the target and check model performance.

        Parameters
        ----------
        x_train: training data

        x_test: testing data

        y_train: training target

        y_test: testing target
        """
        # Predict target
        y_train_predicted = self.model.predict(x_train)

        if self.is_regressor:
            self._predict_regressor(y_train, y_train_predicted, y_test, self.model.predict(x_test))
        else:
            # Check model performance with training data
            self.print_training_accuracy(y_train, y_train_predicted)

            # Check model performance with testing data
            self.print_testing_accuracy(x_test, y_test)


    def _predict_regressor(self,
                           y_train: pd.Series,
                           y_train_predicted: npt.NDArray[np.generic],
                           y_test: pd.Series,
                           y_test_predicted: npt.NDArray[np.generic]) -> None:
        """Specific preidction for regressor, check RegressorMix for the implementation."""
        pass

    #############
    # PARAMETER #
    #############
    def get_hyperparameters(self) -> dict[str, any]:
        """Get model hyperparameters."""
        return self.model.get_params()

    def set_hyperparameters(self, **parameters) -> None:
        """Set model hyperparameters."""
        self.model.set_params(**parameters)

    def set_output(self, transform: str = "pandas") -> None:
        """To configure  all steps of the pipeline to output DataFrame."""
        self.model.set_output(transform=transform)

    @property
    def pipeline(self) -> Pipeline:
        """Return the model explicitly typed as Pipeline."""
        if not isinstance(self.model, Pipeline):
            raise TypeError("Model should be a Pipeline to be able to call this method.")
        return self.model

    def get_output_features(self):
        """
        Get output feature names for transformation. In order to call this method, training the
        model is required.
        """
        if not isinstance(self.pipeline[0], ColumnTransformer):
            raise TypeError("In order to get output feature names for transformation, a pipeline "
                            "with transformers must be done.")
        return self.pipeline[0].get_feature_names_out()


    ############
    # ACCURACY #
    ############
    def print_training_accuracy(self,
                                y_train: pd.Series,
                                y_predicted: npt.NDArray[np.generic]) -> None:
        """Print model performance with training data."""
        print(f"The train accuracy is {self.get_training_accuracy(y_train, y_predicted):.3f}.")

    def get_training_accuracy(self,
                              y_train: pd.Series,
                              y_predicted: npt.NDArray[np.generic]) -> float:
        """Get model performance with training data."""
        return (y_train == y_predicted).mean()

    def print_testing_accuracy(self, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Print model performance with testing data."""
        print("The test accuracy is "
              f"{self.get_testing_accuracy(x_test, y_test):.3f}")

    def get_testing_accuracy(self, x_test: pd.DataFrame, y_test: pd.Series) -> float:
        """Get model performance with testing data."""
        return self.model.score(x_test, y_test)

    def print_mean_absolute_error(self, x_data: pd.DataFrame, targets: pd.Series) -> None:
        """Print model mean absolute error."""
        print("The mean absolute error of the optimal model is "
              f"{self.get_mean_absolute_error(x_data, targets):.3f}")

    def get_mean_absolute_error(self, x_data: pd.DataFrame, targets: pd.Series) -> float:
        """Get model mean absolute error."""
        return mean_absolute_error(targets, self.model.predict(x_data))

    def print_mean_squared_error(self, x_data: pd.DataFrame, targets: pd.Series) -> None:
        """Print model mean absolute error."""
        print("The mean squared error of the optimal model is "
              f"{self.get_mean_squared_error(x_data, targets):.3f}")

    def get_mean_squared_error(self, x_data: pd.DataFrame, targets: pd.Series) -> float:
        """Get model mean squared error."""
        return mean_squared_error(targets, self.model.predict(x_data))


    ####################
    # CROSS VALIDATION #
    ####################
    def print_cross_validate(self, scores: CvResults) -> None:
        """Print cross validation result."""
        # Testing error
        mean_score, std_score = \
            self.get_cross_validate_mean_and_std(scores['test_score'])
        # print(f"Shuffle split cross-validation with n={n_splits}.\n"
        print("The mean cross-validated testing error is: "
             f"{mean_score:.3f} ± {std_score:.3f}"
             f" with an average fitting time of {scores['fit_time'].mean():.3f}")

        # Training error
        if "train_score" in (res := scores):
            mean_score, std_score = \
                self.get_cross_validate_mean_and_std(res['train_score'])
            print("The mean cross-validated training error is "
                  f"{mean_score:.3f} ± {std_score:.3f}")

    def get_cv_estimator(self, scores: CvResults) -> list[AcceptModelType]:
        """
        The estimator objects for each cv split. This is available only if return_estimator
        parameter of kfold_cross_validate is set to True.
        """
        if "estimator" in (res := scores):
            return res["estimator"]
        raise Exception("In order to get the estimator objects for each cv split, return_estimator"
                        " must be set to True.")

    def revert_negation(self,
                        results: CvResults,
                        scoring: str | None) -> CvResults:
        """Revert the negation apply to the error metric to get the actual error."""
        if scoring and scoring.startswith("neg_"):
            results["test_score"] = -results["test_score"]
            if "train_score" in (res := results):
                results["train_score"] = -res["train_score"]
        return results


    ##########################
    # KFLOD CROSS VALIDATION #
    ##########################
    def kfold_cross_validate(self,
                             x_data: pd.DataFrame,
                             y_data: pd.Series,
                             nb_fold: int = 5,
                             scoring: str | None = None,
                             return_train_score: bool = False,
                             return_estimator: bool = False,
                             **kwargs) -> CvResults:
        """
        To performe a kfold cross-validation strategy.

        Parameter
        ---------
        x_data: the whole dataset

        y_data: the whole targets

        nb_fold: the cross-validation splitting strategy

        scoring: strategy to evaluate the performance of the estimator across cross-validation
        splits.

        **kwargs: by default use the current model to perform the cross-validation. For
        hyperparameter tuning, an outer cross-validation is performed on the whole dataset using
        the tuned model.
        """
        results:CvResults = cross_validate(
            self.model if "model" not in kwargs.keys() else kwargs["model"],
            x_data,
            y_data,
            cv=nb_fold,
            scoring=scoring,
            return_train_score=return_train_score,
            error_score="raise",
            n_jobs=2,
            return_estimator=return_estimator)

        return self.revert_negation(results, scoring)

    def print_kfold_cross_validation_accuracy(self, scores: CvResults) -> None:
        """Print the result (accuracy and fitting time) of a kfold cross-validation strategy."""
        print(f"Kfold cross-validation with K={len(scores['test_score'])}.")
        self.print_cross_validate(scores)


    def get_cross_validate_mean_and_std(self,
                                        scores: npt.NDArray[np.float64]) \
                                        -> tuple[float, float]:
        """Get the mean/std of a cross_validation strategy."""
        return scores.mean(), scores.std()


    ##################################
    # SHUFFLE SPLIT CROSS VALIDATION #
    ##################################
    def shuffle_split_cv_generator(self, n_splits: int, test_size: float = 0.2):
        """Set up a shuffle split cross-validation strategy."""
        return ShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=0
        )

    def shuffle_split_cross_validate(self,
                                     x_data: pd.DataFrame,
                                     y_data: pd.Series,
                                     n_splits: int,
                                     scoring: str | None = None,
                                     test_size: float = 0.2,
                                     return_train_score: bool = False) -> CvResults:
        """
        To performe a shuffle split cross-validation strategy.

        Scikit-learn allow the use of any metric, like 'mean_absolute_error' into a score to be
        used in cross validate. In order to do so, pass a string of the error metric with and
        additional neg_ such as 'neg_mean_absolute_error'.

        Parameter
        ---------
        x_data: the whole dataset

        y_data: the whole targets

        n_splits: number of re-shuffling & splitting iterations.
        """
        # Set the cross-validation strategy and performe it
        results: CvResults = cross_validate(self.model,
                                            x_data,
                                            y_data,
                                            cv=self.shuffle_split_cv_generator(n_splits,
                                                                               test_size),
                                            scoring=scoring,
                                            return_train_score=return_train_score,
                                            error_score="raise",
                                            n_jobs=2)

        return self.revert_negation(results, scoring)

    def print_shuffle_split_cross_validation_accuracy(self, scores: CvResults) -> None:
        """Print the result (accuracy and fitting time) of a shuffle split cv strategy."""
        print(f"Shuffle split cross-validation with n={len(scores['test_score'])}.")
        self.print_cross_validate(scores)


    ################################
    # MANUAL HYPERPARAMETER TUNING #
    ################################
    def _hyperparameter_loop_recurcive(self,
                                       data: pd.DataFrame,
                                       targets: pd.Series,
                                       scores: dict[str, list[float|int]],
                                       index: int,
                                       **hyperparameters):
        """
        Use recursion to build a nested for loop for each hyperparameters in order to perform
        cross-validation and tuned them.
        """
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
            if (res := {
                    k: v for k, v in results.items() if k in ["training_mean", "training_std"]
            }):
                scores["training_mean"].append(res["train_score"].mean())
                scores["training_std"].append(res["train_score"].std())

    def manual_hyperparameter_tuning(self,
                                     data: pd.DataFrame,
                                     targets: pd.Series,
                                     **hyperparameters) -> dict[str, list[float|int]]:
        """Manually tuned hyperparameters."""
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


    ###################################
    # AUTOMATED HYPERPARAMETER TUNING #
    ###################################
    def _grid_search_cv(self,
                       param_grid: dict[str, list[float|int]],
                       cv: int) -> GridSearchCV:
        """
        Hyperparameter tuning by grid-search on the training set.

        Some limitation:
        - does not scale well with the number of hyperparameters to tune
        - the grid imposes a regularity during the search which might miss better parameter
        values between two consecutive values on the grid.
        - hyperparameters value must be specified explicitly
        """
        return GridSearchCV(self.model,
                            param_grid=param_grid,
                            cv=cv,
                            n_jobs=2,
                            return_train_score=True)

    def _randomized_search_cv(self,
                             param_distribution: dict[str, list[float|int]],
                             n_iter: int,
                             cv: int,
                             scoring: str | None = None) -> RandomizedSearchCV:
        """
        Hyperparameter tuning by randomized-search on the training set.

        Randomly generate the parameters candidate
        """
        return RandomizedSearchCV(self.model,
                                  param_distributions=param_distribution,
                                  n_iter=n_iter,
                                  cv=cv,
                                  verbose=1,
                                  return_train_score=True,
                                  scoring=scoring)

    def _save_results_as_dataframe(self, results: dict[str, list[float]], path: Path) -> None:
        """Print scores as a DataFrame."""
        # Build the DataFrame
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

        # Save the DataFrame as a csv
        cv_results.to_csv(path, index=False)

    def _no_missing_parameters(self,
                               search_class: str,
                               expected_param: list[str],
                               **kwargs) -> None:
        """
        In order to perform an automated search to tune hyperparameter some additional paramaters
        are required and are specific to the search class.
        """
        if not set(expected_param).issubset(kwargs.keys()):
            raise Exception(f"In order to perform a {search_class} parameters "
                            f"{', '.join(expected_param)} are expected")

    def automated_search_cross_validation(self,
                                          search_class: type[GridSearchCV|RandomizedSearchCV],
                                          hyperparameters: dict[str, list[float|int]],
                                          x_data: pd.DataFrame,
                                          y_data: pd.Series,
                                          x_train: pd.DataFrame,
                                          y_train: pd.Series,
                                          path: Path | None = None,
                                          **kwargs):
        """
        Perform an automated search to tune hyperparameter.

        Parameter
        ---------
        search_class: which method to apply for hyperparameter tuning. For now, either a
        grid-search or a randomized-search

        hyperparameters: hyperparameter values to try out for model tuning. Either fix values for
        the grid-search or a range of values for the randomized-search.

        x_data: the whole dataset use for the outer cross-validation, i.e. the cv used to evaluate
        the generalization performance of the tuned model

        y_data: the whole targets use for the outer cross-validation

        x_train: training data set use for the inner cross-validation, i.e. the cv used to tune the
        hyperparameters

        y_train: training targets use for the inner cross-validation

        path: automated tuning especially for a randomized-search with a large number of iterations
        is costly. Therefore, at the end result are saved as a csv file.

        kwargs: list of parameters specific to the search class
        """
        scoring: str | None = kwargs["scoring"] if "scoring" in kwargs.keys() else None

        # 1. Build an automated-search cross-validation
        self._no_missing_parameters(
            search_class.__name__, SEARCH_EXPECTED_PARAM[search_class.__name__], **kwargs
        )
        if search_class is GridSearchCV:
            search_model = self._grid_search_cv(hyperparameters, cv=kwargs["cv"])
        elif search_class is RandomizedSearchCV:
            search_model = self._randomized_search_cv(hyperparameters,
                                                      cv=kwargs["cv"],
                                                      n_iter=kwargs["n_iter"],
                                                      scoring=scoring)
        else:
            raise Exception(f"Unsupported search methods to tune hyperparameters.")

        # 2. Train the model with the best set of hyperparameters
        search_model.fit(x_train, y_train)

        # Revert the negation apply by the error metric to get the actual error
        if scoring and scoring.startswith("neg_"):
            search_model.cv_results_["mean_test_score"] *= -1
            search_model.cv_results_["mean_train_score"] *= -1

        # 3. Display the optimum hyperparameters
        print(f"The optimum set of hyperparameters is {search_model.best_params_}")

        # 4. Save the tuning result into a dataframe
        if path:
            self._save_results_as_dataframe(search_model.cv_results_, path)

        # 5. Compute the generalization performance of the model with the score method preovide a
        #    single estimation of the generalization performance. Therefore, it's always preferable
        #    to perform a cross-validation. This pattern is called a nested cross-validation.
        print("Perform an outer cross-validation on the whole dataset to compute the "
              "generalization performance of the model with tuned hyperparameters.")
        scores = self.kfold_cross_validate(
            x_data, y_data, nb_fold=5, scoring=scoring, return_train_score=True, model=search_model
        )
        self.print_kfold_cross_validation_accuracy(scores)


    ####################
    # VALIDATION CURVE #
    ####################
    def compute_validation_curve(self,
                                 x_data: pd.DataFrame,
                                 y_data: pd.Series,
                                 scoring: str,
                                 score_name: str,
                                 negate_score: bool = False,
                                 cv: Tcv | None = None,
                                 **kwargs) -> ValidationCurveDisplay:
        """
        Use validation curve to try out hyperparameters.

        Pass either a classifier or a pipeline (parameter self.model) to
        ValidationCurveDisplay.from_estimator
        """
        return ValidationCurveDisplay.from_estimator(self.model,
                                                     x_data,
                                                     y_data,
                                                     cv=cv,
                                                     scoring=scoring,
                                                     score_name=score_name,
                                                     param_name=kwargs["param_name"],
                                                     param_range=kwargs["param_range"],
                                                     negate_score=negate_score,
                                                     std_display_style="fill_between",
                                                     n_jobs=2)


    ##################
    # LEARNING CURVE #
    ##################
    def compute_learning_curve(self,
                               x_data: pd.DataFrame,
                               y_data: pd.Series,
                               cv: Tcv,
                               scoring: str,
                               score_name: str,
                               negate_score: bool = False,
                               **kwargs) -> LearningCurveDisplay:
        """
        Use learning curve to try out various training set size.

        Pass either a classifier or a pipeline (parameter self.model)
        to ValidationCurveDisplay.from_estimator
        """
        return LearningCurveDisplay.from_estimator(self.model,
                                                   x_data,
                                                   y_data,
                                                   train_sizes=kwargs["train_sizes"],
                                                   cv=cv,
                                                   score_type="both", # both train and test errors
                                                   scoring=scoring,
                                                   score_name=score_name,
                                                   negate_score=negate_score,
                                                   std_display_style="fill_between",
                                                   n_jobs=2)
