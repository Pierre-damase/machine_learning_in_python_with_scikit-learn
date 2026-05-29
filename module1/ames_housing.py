import data_handler as dh
import pandas as pd
from config import AMES_HOUSING_NUMERICAL_FEATURES, DataPath, TargetColumn
from model import LogisticRegressionModel
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def convert_targets(targets: pd.Series) -> pd.Series:
    """
    Convert the regression target into a classification target to predict
    whether or not an house is expensive.
    """
    return pd.cut(targets, [0, 200000, 1000000], labels=['cheap', 'expensive'])


def predictive_model_with_numerical_features(data: pd.DataFrame, targets: pd.Series) -> None:
    """Predictive model using only numerical features."""
    # Build logistic regression model
    logistic_regression = LogisticRegressionModel(pipeline_steps=[
        StandardScaler(), LogisticRegression()
    ])

    # KFold cross-validation to evaluate generalization performance of the model
    scores = logistic_regression.kfold_cross_validate(data, targets, 10)
    logistic_regression.print_kfold_cross_validation_accuracy(scores)

def predictive_model(data: pd.DataFrame, targets: pd.Series) -> None:
    """
    Predictive model using both numerical and categorical features. Therefore, it's required to
    apply a specific transformer for each data type:
      - Performe a one-hot encoding of categorical variables
      - Performe a scaling of numerical variables
    """
    # 1. Machine learning pipeline with column transformer to
    initialized_model = LogisticRegressionModel.build_pipeline_with_transformer(
        transformers=[
            (StandardScaler(), selector(dtype_exclude=object)),
            (OneHotEncoder(handle_unknown="ignore"), selector(dtype_include=object))
        ],
        model=LogisticRegression(max_iter=500)
    )

    # 2. KFold cross-validation to evaluate generalization performance of the model
    scores = initialized_model.kfold_cross_validate(data, targets, 5)
    initialized_model.print_kfold_cross_validation_accuracy(scores)


############
# ANALYSIS #
############
def run_analysis():
    # 1. Load data
    data, targets = dh.load_data_from_file(DataPath.AMES_HOUSING.value,
                                           TargetColumn.AMES_HOUSING)

    # 2. Convert continuous target into a classification target
    categorical_targets = convert_targets(targets)

    # 3. Predictive model
    # Using only numerical features
    # predictive_model_with_numerical_features(data[NUMERICAL_FEATURES], categorical_targets)

    # With numerical and categorical features
    predictive_model(data[AMES_HOUSING_NUMERICAL_FEATURES], categorical_targets)

if __name__ == "__main__":
    run_analysis()
