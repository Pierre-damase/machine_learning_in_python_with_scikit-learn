from model import (
    DummyClassifierModel, 
    KNeighborsClassifierModel, 
    LogisticRegressionModel
)
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from visualisation import check_data, scaler_jointplot

import data_handler as dh
import pandas as pd

ADULT_PATH = Path("./data/adult.arff")

"""Load numerical data from adult census."""
def _load_numerical_data() -> tuple[pd.DataFrame, pd.Series]:
    # Load adult census data as DataFrame and extract the target
    adult_census, targets = dh.load_data_from_arff(ADULT_PATH)

    # Filtered out any non-numeric values
    data = dh.get_subset_from_dtypes(adult_census, [int, float])

    return data, targets


"""
Build predictive models on tabular assets with only numerical features.

n.b. the following workflow is simplify for some parts because data are already known
"""
def first_model_with_scikit_learn() -> None:
    # 1. Load adult census data as DataFrame and extract the target
    # adult_census, targets = extract_target(load_data_from_arff(ADULT_PATH), ADULT_TARGET)
    adult_census, targets = dh.load_data_from_arff(ADULT_PATH)

    # 2. Look at the data to get some insight
    # check_data(adult_census, targets)

    # Select some feature (just for an example of a simple model)
    data = dh.get_subset(adult_census, ["age", "capital-gain", "capital-loss", "hours-per-week"])

    # 3. Randomnly split data between train and test set
    train_test_split = dh.manual_train_test_split(data, targets)

    # 4. Build model
    k_neighbors = KNeighborsClassifierModel(5)
    k_neighbors.start(*train_test_split)
    k_neighbors = KNeighborsClassifierModel(50)
    k_neighbors.start(*train_test_split)


"""
Performe moreadvanced operation:

- identifying numerical data in a heterogeneous dataset;
- selecting the subset of columns corresponding to numerical data;
- using a scikit-learn helper to separate data into train-test sets;
- training and evaluating a more complex scikit-learn model.
"""
def working_with_numerical_data() -> None:
    # 1. Load numerical data from adult census data as DataFrame and extract the target
    data, targets = _load_numerical_data()

    # 2. Look at the data to get some insight
    # check_data(adult_census, targets)

    # 3. Randomnly split data between train and test set
    train_test_split = dh.sklearn_train_test_split(data, targets)

    # 4.1 Build logistic regression model
    logistic_regression = LogisticRegressionModel()
    logistic_regression.start(*train_test_split)

    # 4.2 Build dummy classifier model
    high_income = DummyClassifierModel(">50K")
    high_income.start(*train_test_split) # catastrophic (maybe highlight a class inbalance)

    low_income = DummyClassifierModel("<=50K")
    low_income.start(*train_test_split) # ok


"""Preprocessing of numerical features with standard scaler."""
def preprocessing_for_numerical_features() -> None:
    use_pipeline = True

    # 1. Load numerical data from adult census data as DataFrame and extract the target
    data, targets = _load_numerical_data()

    # 2. Look at the data to get some insight
    # check_data(adult_census, targets)

    # 3. Randomnly split data between train and test set
    train_test_split = dh.sklearn_train_test_split(data, targets)

    if not use_pipeline:
        # 4. Transform input data before training a model
        x_train_scaled = dh.standard_scaler(train_test_split[0])

        # 4bis. Check scaler effect on data
        print(x_train_scaled.describe())
        scaler_jointplot(train_test_split[0], x_train_scaled, x_axis="age", y_axis="hours-per-week")

        # 5. Build logistic regression model
        logistic_regression = LogisticRegressionModel()
        logistic_regression.start(*train_test_split)
    else:
        # 4&5. Start a simple pipeline to scale the data + build a logistic regression model
        logistic_regression = LogisticRegressionModel(pipeline_steps=[StandardScaler(), LogisticRegression()])
        logistic_regression.start(*train_test_split)


"""Use cross evaluation to evaluate generalization performance of the model."""
def model_evaluation_using_cross_validation():
    # 1. Load numerical data from adult census data as DataFrame and extract the target
    data, targets = _load_numerical_data()

    # 2. Look at the data to get some insight
    # check_data(adult_census, targets)

    # 3. Build logistic regression model
    logistic_regression = LogisticRegressionModel(pipeline_steps=[StandardScaler(), LogisticRegression()])
    
    # 4. KFold cross-validation to evaluate generalization performance of the model
    scores = logistic_regression.kfold_cross_validate(data, targets, 5)
    logistic_regression.print_kfold_cross_validation_accuracy(scores)

    scores = logistic_regression.kfold_cross_validate(data, targets, 10)
    logistic_regression.print_kfold_cross_validation_accuracy(scores)


if __name__ == "__main__":
    # first_model_with_scikit_learn()
    working_with_numerical_data()
    # preprocessing_for_numerical_features()
    model_evaluation_using_cross_validation()