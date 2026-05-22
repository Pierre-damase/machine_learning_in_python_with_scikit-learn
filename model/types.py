from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from typing import TypeVar


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
type AcceptPreprocessorType = (tuple[QuantileTransformer
                                     | MinMaxScaler
                                     | OneHotEncoder
                                     | OrdinalEncoder
                                     | PowerTransformer
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

# Cross-validation type
Tcv = TypeVar('Tcv', bound=ShuffleSplit)
