from typing import NotRequired, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LinearRegression, LogisticRegression
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
type DataSetType = tuple[pd.DataFrame, pd.Series]

################
# LINEAR MODEL #
################
type AcceptLinearModelWithPipelineType = (LinearRegression
                                          | LogisticRegression
                                          | Pipeline)
Tlinearmodel = TypeVar('Tlinearmodel', bound=AcceptLinearModelWithPipelineType)


##############
# CLASSIFIER #
##############
type AcceptClassifierWithPipelineType = (HistGradientBoostingClassifier
                                     | KNeighborsClassifier
                                     | LogisticRegression
                                     | SVC)
Tclassifierwithpipeline = TypeVar('Tclassifierwithpipeline', bound=AcceptClassifierWithPipelineType)

type AcceptClassifierType = (DecisionTreeRegressor
                        | DummyClassifier
                        | Pipeline
                        | AcceptClassifierWithPipelineType)
Tclassifier = TypeVar('Tclassifier', bound=AcceptClassifierType)

#############
# REGRESSOR #
#############
type AcceptRegressorWithPipelineType = (KNeighborsRegressor
                                        | LinearRegression)
Tregressorwithpipeline = TypeVar('Tregressorwithpipeline', bound=AcceptRegressorWithPipelineType)

type AcceptRegressorType = (DecisionTreeRegressor
                            | Pipeline
                            | SVR
                            | AcceptRegressorWithPipelineType)
Tregressor = TypeVar('Tregressor', bound=AcceptRegressorType)


#########
# MODEL #
#########
type AcceptModelType = (AcceptClassifierType | AcceptRegressorType)
Tmodel = TypeVar('Tmodel', bound=AcceptModelType)

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

##################
# PIPELINE STEPS #
##################
type AcceptPipelineType = (OneHotEncoder
                            | OrdinalEncoder
                            | StandardScaler
                            | AcceptClassifierWithPipelineType
                            | AcceptRegressorWithPipelineType
                            | AcceptPreprocessorType)
Tpipelinesteps = TypeVar('Tpipelinesteps', bound=AcceptPipelineType)

####################
# CROSS-VALIDATION #
####################
# Cross-validation type, either an int to specify the number of folds in a KFold or a CV splitter
type AcceptCvType = (int | ShuffleSplit)
Tcv = TypeVar('Tcv', bound=AcceptCvType)

# Cross-validation result contains at least the test result , test time, train time and may
# contains the train result and estimator for each split.
class CvResults(TypedDict):
    fit_time: npt.NDArray[np.float64]
    score_time: npt.NDArray[np.float64]
    test_score: npt.NDArray[np.float64]
    train_score: NotRequired[npt.NDArray[np.float64]]
    estimator: NotRequired[list[AcceptModelType]]
