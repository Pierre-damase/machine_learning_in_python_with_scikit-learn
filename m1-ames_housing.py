from config import (
    AMES_HOUSING_NUMERICAL_FEATURES,
    DataPath,
    TargetColumn
)
from model import LogisticRegressionModel
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

import data_handler as dh
import pandas as pd


"""
Convert the regression target into a classification target to predict
whether or not an house is expensive.
"""
def convert_targets(targets: pd.Series) -> pd.Series:
    return pd.cut(targets, [0, 200000, 1000000], labels=['cheap', 'expensive'])


"""Predictive model using only numerical features."""
def predictive_model_with_numerical_features(data: pd.DataFrame, targets: pd.Series) -> None:
    # Build logistic regression model
    logistic_regression = LogisticRegressionModel(pipeline_steps=[StandardScaler(), LogisticRegression()])

    # KFold cross-validation to evaluate generalization performance of the model
    scores = logistic_regression.kfold_cross_validate(data, targets, 10)
    logistic_regression.print_kfold_cross_validation_accuracy(scores)

"""Predictive model using both numerical and categorical features."""
def predictive_model(data: pd.DataFrame, targets: pd.Series) -> None:
    # 1. Machine learning pipeline with column transformer to
    #    - Performe a one-hot encoding of categorical variables
    #    - Performe a scaling of numerical variables
    initialized_model = \
        LogisticRegressionModel(pipeline_steps=[
            # Transformers
            *[
                (StandardScaler(), selector(dtype_exclude=object)),
                (OneHotEncoder(handle_unknown="ignore"), selector(dtype_include=object))
            ],
            # Classifier 
            LogisticRegression(max_iter=500)
        ])

    # 2. KFold cross-validation to evaluate generalization performance of the model
    scores = initialized_model.kfold_cross_validate(data, targets, 5)
    initialized_model.print_kfold_cross_validation_accuracy(scores)


if __name__ == "__main__":
    # 1. Load data
    data, targets = dh.load_data_from_csv(DataPath.AMES_HOUSING.value,
                                          TargetColumn.AMES_HOUSING)

    # 2. Convert continuous target into a classification target
    categorical_targets = convert_targets(targets)

    # 3. Predictive model
    # Using only numerical features
    # predictive_model_with_numerical_features(data[NUMERICAL_FEATURES], categorical_targets)

    # With numerical and categorical features
    predictive_model(data[AMES_HOUSING_NUMERICAL_FEATURES], categorical_targets)
