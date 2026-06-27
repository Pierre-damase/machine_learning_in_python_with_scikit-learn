import data_handler as dh
import pandas as pd
from config import AMES_HOUSING_NUMERICAL_FEATURES, DataPath, TargetColumn
from model import LogisticRegressionModel
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from types_config import CvParameters


def convert_targets(y_data: pd.Series) -> pd.Series:
    """
    Convert the regression target into a classification target to predict
    whether or not an house is expensive.
    """
    return pd.Series(pd.cut(y_data, [0, 200000, 1000000], labels=['cheap', 'expensive']))


def predictive_model_with_numerical_features(x_data: pd.DataFrame, y_data: pd.Series) -> None:
    """Predictive model using only numerical features."""
    # Build logistic regression model
    logistic_regression = LogisticRegressionModel.build_pipeline([StandardScaler()])

    # KFold cross-validation to evaluate generalization performance of the model
    scores = logistic_regression.make_cross_validate(x_data, y_data, cv_params=CvParameters(10))
    logistic_regression.print_cross_validate(scores)

def predictive_model(x_data: pd.DataFrame, y_data: pd.Series) -> None:
    """
    Predictive model using both numerical and categorical features. Therefore, it's required to
    apply a specific transformer for each data type:
      - Perform a one-hot encoding of categorical variables
      - Perform a scaling of numerical variables
    """
    # 1. Machine learning pipeline with column transformer to
    initialized_model = LogisticRegressionModel.build_pipeline(
        transformers=[
            (StandardScaler(), selector(dtype_exclude=object)),
            (OneHotEncoder(handle_unknown="ignore"), selector(dtype_include=object))
        ],
        max_iter=500
    )

    # 2. KFold cross-validation to evaluate generalization performance of the model
    scores = initialized_model.make_cross_validate(x_data, y_data, cv_params=CvParameters(5))
    initialized_model.print_cross_validate(scores)


############
# ANALYSIS #
############
def run_analysis():
    # 1. Load data
    data = dh.load_data_from_file(DataPath.AMES_HOUSING.value, target=TargetColumn.AMES_HOUSING)

    # 2. Convert continuous target into a classification target
    categorical_targets = convert_targets(data["y_data"])

    # 3. Predictive model
    # Using only numerical features
    predictive_model_with_numerical_features(
        dh.get_subset(data["x_data"], AMES_HOUSING_NUMERICAL_FEATURES),
        categorical_targets
    )

    # With numerical and categorical features
    predictive_model(data["x_data"], categorical_targets)

if __name__ == "__main__":
    run_analysis()
