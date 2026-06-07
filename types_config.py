from typing import NotRequired, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingRegressor, HistGradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import (LinearRegression, LogisticRegression, Ridge,
                                  RidgeCV)
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler,
                                   OneHotEncoder, OrdinalEncoder,
                                   PolynomialFeatures, PowerTransformer,
                                   QuantileTransformer, SplineTransformer,
                                   StandardScaler)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor


########
# DATA #
########
# type DataSetType = tuple[pd.DataFrame, pd.Series]
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
Tlinearestimator = TypeVar('Tlinearestimator', bound=AcceptLinearEstimatorType)

Tlinearmodel = TypeVar('Tlinearmodel', bound=AcceptLinearEstimatorType | Pipeline)

# Classifier
type AcceptClassifierType = (DecisionTreeRegressor
                             | DummyClassifier
                             | HistGradientBoostingClassifier
                             | KNeighborsClassifier
                             | LogisticRegression
                             | Ridge
                             | RidgeCV
                             | SVC)
Tclassifier = TypeVar('Tclassifier', bound=AcceptClassifierType)

# Regressor
type AcceptRegressorType = (BaggingRegressor
                            | DecisionTreeRegressor
                            | KNeighborsRegressor
                            | LinearRegression
                            | SVR)
Tregressor = TypeVar('Tregressor', bound=AcceptRegressorType)

# Estimator
type AcceptEstimatorType = (AcceptClassifierType | AcceptRegressorType)
Testimator = TypeVar('Testimator', bound=AcceptEstimatorType)

# Model
Tmodel = TypeVar('Tmodel', bound=AcceptEstimatorType | Pipeline)


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
Tpreprocessor = TypeVar('Tpreprocessor', bound=AcceptPreprocessorType)


####################
# CROSS-VALIDATION #
####################
# Cross-validation type, either an int to specify the number of folds in a KFold or a CV splitter
type AcceptCvType = (int | ShuffleSplit)
Tcv = TypeVar('Tcv', bound=AcceptCvType)

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
