from typing import TypeVar

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
# str type for passthrough, columns specified with it are added at the right
type AcceptPreprocessorType = (tuple[KBinsDiscretizer
                                     | QuantileTransformer
                                     | MinMaxScaler
                                     | Nystroem
                                     | OneHotEncoder
                                     | OrdinalEncoder
                                     | PolynomialFeatures
                                     | PowerTransformer
                                     | SplineTransformer
                                     | StandardScaler
                                     | str, selector])
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

# Cross-validation type, either an int to specify the number of folds in a KFold or a CV splitter
type AcceptCvType = (int | ShuffleSplit)
Tcv = TypeVar('Tcv', bound=AcceptCvType)
