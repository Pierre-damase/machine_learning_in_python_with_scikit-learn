from typing import Literal, NamedTuple, NotRequired, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              BaggingRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import (LinearRegression, LogisticRegression, Ridge,
                                  RidgeCV)
from sklearn.metrics._scorer import _Scorer
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     LeaveOneGroupOut, RandomizedSearchCV,
                                     ShuffleSplit, StratifiedKFold,
                                     TimeSeriesSplit)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler,
                                   OneHotEncoder, OrdinalEncoder,
                                   PolynomialFeatures, PowerTransformer,
                                   QuantileTransformer, SplineTransformer,
                                   StandardScaler)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


########
# DATA #
########
class DataSetType(TypedDict):
    """
    x_data: input data (features)

    y_data: targets
    """
    x_data: pd.DataFrame # input data, i.e. the features
    y_data: pd.Series # the targets

class SplitSetType(TypedDict):
    """
    x_train: training data

    x_test: testing data

    y_train: training targets

    y_test: testing targets
    """
    x_train: pd.DataFrame # training data
    x_test: pd.DataFrame  # testing data
    y_train: pd.Series # training targets
    y_test: pd.Series  # testing targets


########
# MODEL #
########
# Linear model
type AcceptLinearEstimatorType = (LinearRegression
                              | LogisticRegression
                              | Ridge
                              | RidgeCV)
Tlinearestimator = TypeVar("Tlinearestimator", bound=AcceptLinearEstimatorType)

Tlinearmodel = TypeVar("Tlinearmodel", bound=AcceptLinearEstimatorType | Pipeline)

# Classifier for classification task
type AcceptClassifierType = (AdaBoostClassifier
                             | BaggingClassifier
                             | DecisionTreeClassifier
                             | DummyClassifier
                             | GradientBoostingClassifier
                             | HistGradientBoostingClassifier
                             | KNeighborsClassifier
                             | LogisticRegression
                             | RandomForestClassifier
                             | Ridge
                             | RidgeCV
                             | SVC)
Tclassifier = TypeVar("Tclassifier", bound=AcceptClassifierType)

# Regressor for regression task
type AcceptRegressorType = (BaggingRegressor
                            | DecisionTreeRegressor
                            | DummyRegressor
                            | GradientBoostingRegressor
                            | KNeighborsRegressor
                            | LinearRegression
                            | RandomForestRegressor
                            | SVR)
Tregressor = TypeVar("Tregressor", bound=AcceptRegressorType)

# Estimator
type AcceptEstimatorType = (AcceptClassifierType | AcceptRegressorType)
Testimator = TypeVar("Testimator", bound=AcceptEstimatorType)

# Model
Tmodel = TypeVar("Tmodel", bound=AcceptEstimatorType | Pipeline)


#################
# PRE-PROCESSOR #
#################
type PreprocessorType = (KBinsDiscretizer
                         | QuantileTransformer
                         | MinMaxScaler
                         | Nystroem
                         | OneHotEncoder
                         | OrdinalEncoder
                         | PolynomialFeatures
                         | PowerTransformer
                         | SplineTransformer
                         | StandardScaler)

# Either a tuple (transformer, columns) with transformer=transformer to apply and
#                                      columns=which to modify with the transfomer or 'passthrough'
# Or just a transformer which is apply on all data
type AcceptPreprocessorType = (PreprocessorType
                               | tuple[PreprocessorType | str, selector])
Tpreprocessor = TypeVar("Tpreprocessor", bound=AcceptPreprocessorType)


####################
# CROSS-VALIDATION #
####################
# Allow cv strategy
type CvStrategy = Literal["GroupKFold", "KFold", "LeaveOneGroupOut", "ShuffleSplit",
                          "StratifiedKFold", "TimeSeriesSplit"]

# Cross-validation type
type CvType = (GroupKFold
               | KFold
               | LeaveOneGroupOut
               | ShuffleSplit
               | StratifiedKFold
               | TimeSeriesSplit)
Tcv = TypeVar("Tcv", bound=CvType | int) # int to specify the nb of folds in a KFold

# Cross-validation result contains at least the test result , test time, train time and may
# contains the train result and estimator for each split.
class CvResults(TypedDict):
    """
    fit_time: the time for fitting the estimator on the train set for each cv split

    score_time: the time for scoring the estimator on the test set for each cv split

    test_score: the score array for test scores on each cv split

    train_score: the score array for train scores on each cv split [Not required]

    estimator: estimator object for each cv split. [Not required]
    """
    fit_time: npt.NDArray[np.float64]
    score_time: npt.NDArray[np.float64]
    test_score: npt.NDArray[np.float64]
    train_score: NotRequired[npt.NDArray[np.float64]]
    estimator: NotRequired[list[AcceptEstimatorType]]

class CvParameters(NamedTuple):
    """
    * k-fold cv *
    nb_splits: the number of folds

    shuffle: whether to shuffle the data before splitting into batches

    random_state: when shuffle is True, random_state affects the ordering of the indices, which
    controls the randomness of each fold

    * Suffle split cv *
    n_splits: the number of re-shuffling & splitting iterations

    test_size: represent the proportion of the dataset to include in the test split
    """
    n_splits: int
    shuffle: bool = False
    random_state: int | None = None
    test_size: float | None = None


#########################
# HYPERPARAMETER TUNING #
#########################
type SearchCvType = (GridSearchCV
                     | RandomizedSearchCV)

type SearchCvHyperparamType = dict[str, list[float|int|None] | npt.NDArray[np.int64]]

class SearchCvParameters(NamedTuple):
    """
    cv: determines the cross-validation splitting strategy. At the moment only an integer to
    specifythe number of folds in a KFold strategy

    n_iter: number of parameter settings that are sampled for a randomized-search cv
    """
    cv: int
    n_iter: int | None = None

class SearchOuterCv(NamedTuple):
    """
    x_data: the whole dataset use by the outer cross-validation, i.e. the cv used to evaluate the
    generalization performance of the tuned model

    y_data: the whole targets use for the outer cross-validation

    cv_strategy: which cross-validation strategy to use

    cv_params: additional parameter of the the cross-validation strategy
    """
    x_data: pd.DataFrame
    y_data: pd.Series
    cv_strategy: CvStrategy = "KFold"
    cv_params: CvParameters = CvParameters(5)


###########
# SCORING #
###########
type ScoringFunctionType = (Literal["balanced_accuracy",
                                   "neg_mean_absolute_error",
                                   "neg_mean_absolute_percentage_error",
                                   "r2"]
                            | _Scorer)

type MetricFunctionType = Literal["median_absolute_error",
                                  "mean_absolute_error",
                                  "mean_absolute_percentage_error",
                                  "mean_squared_error"]
