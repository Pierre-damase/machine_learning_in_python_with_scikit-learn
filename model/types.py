from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from typing import TypeVar


# Model type
type AcceptModelWithPipelineType = (HistGradientBoostingClassifier
                                     | KNeighborsClassifier
                                     | LogisticRegression
                                     | SVC)
Tmodelwithpipeline = TypeVar(
    'Tmodelwithpipeline', bound=AcceptModelWithPipelineType
)

type AcceptModelType = (DecisionTreeRegressor
                        | DummyClassifier
                        | Pipeline
                        | AcceptModelWithPipelineType)
Tmodel = TypeVar('Tmodel', bound=AcceptModelType)

# Pre-processor type
# str type for passthrough, columns specified with it are added at the right
type AcceptPreprocessorType = (
    tuple[OneHotEncoder|OrdinalEncoder|StandardScaler|str, selector]
)
Tpreprocessor = TypeVar('Tpreprocessor', bound=AcceptPreprocessorType
)

# Pipeline steps type
type AcceptPipelineType = (OneHotEncoder
                            | OrdinalEncoder
                            | StandardScaler
                            | AcceptModelWithPipelineType
                            | AcceptPreprocessorType)
Tpipelinesteps = TypeVar('Tpipelinesteps', bound=AcceptPipelineType)

# Cross-validation type
Tcv = TypeVar('Tcv', bound=ShuffleSplit)
