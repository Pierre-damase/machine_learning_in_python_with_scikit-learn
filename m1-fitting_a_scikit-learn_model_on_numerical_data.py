import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import data_handler as dh
from config import DataPath, TargetColumn
from model import (DummyClassifierModel, KNeighborsClassifierModel,
                   LogisticRegressionModel)
from visualisation import check_data, scaler_jointplot


def _load_numerical_data(data: pd.DataFrame) -> pd.DataFrame:
    """Load numerical data from adult census."""
    return dh.get_subset(data, dtypes=[int, float])


def first_model_with_scikit_learn(data: pd.DataFrame,
                                  targets: pd.Series) -> None:
    """
    Build predictive models on tabular assets with only numerical features.

    n.b. the following workflow is simplify for some parts because data are already known
    """
    # 1. Look at the data to get some insight
    # check_data(data, targets)

    # 2. Select some feature (just for an example of a simple model)
    data = dh.get_subset(data, columns=["age", "capital-gain", "capital-loss", "hours-per-week"])

    # 3. Randomnly split data between train and test set
    train_test_split = dh.manual_train_test_split(data, targets)

    # 4. Build model
    k_neighbors = KNeighborsClassifierModel(5)
    k_neighbors.start(*train_test_split)
    k_neighbors = KNeighborsClassifierModel(50)
    k_neighbors.start(*train_test_split)


def working_with_numerical_data(data: pd.DataFrame,
                                targets: pd.Series) -> None:
    """
    Performe moreadvanced operation:

    - identifying numerical data in a heterogeneous dataset;
    - selecting the subset of columns corresponding to numerical data;
    - using a scikit-learn helper to separate data into train-test sets;
    - training and evaluating a more complex scikit-learn model.
    """
    # 1. Look at the data to get some insight
    # check_data(data, targets)

    # 2. Randomnly split data between train and test set
    train_test_split = dh.sklearn_train_test_split(data, targets)

    # 3.1 Build logistic regression model
    logistic_regression = LogisticRegressionModel()
    logistic_regression.start(*train_test_split)

    # 4. Build dummy classifier model
    high_income = DummyClassifierModel(strategy="constant", constant=">50K")
    high_income.start(*train_test_split) # catastrophic (maybe highlight a class inbalance)

    low_income = DummyClassifierModel(strategy="constant", constant="<=50K")
    low_income.start(*train_test_split) # ok


def preprocessing_for_numerical_features(data: pd.DataFrame,
                                         targets: pd.Series) -> None:
    """Preprocessing of numerical features with standard scaler."""
    use_pipeline = True

    # 1. Look at the data to get some insight
    # check_data(data, targets)

    # 2. Randomnly split data between train and test set
    train_test_split = dh.sklearn_train_test_split(data, targets)

    if not use_pipeline:
        # 3. Transform input data before training a model
        x_train_scaled = dh.standard_scaler(train_test_split[0])

        # 3bis. Check scaler effect on data
        print(x_train_scaled.describe())
        scaler_jointplot(train_test_split[0],
                         x_train_scaled,
                         x_axis="age",
                         y_axis="hours-per-week")

        # 4. Build logistic regression model
        logistic_regression = LogisticRegressionModel()
        logistic_regression.start(*train_test_split)
    else:
        # 3&4. Start a simple pipeline to scale the data + build a logistic regression model
        logistic_regression = LogisticRegressionModel(
            pipeline_steps=[StandardScaler(), LogisticRegression()]
        )
        logistic_regression.start(*train_test_split)


def model_evaluation_using_cross_validation(data: pd.DataFrame,
                                            targets: pd.Series) -> None:
    """Use cross evaluation to evaluate generalization performance of the model."""
    # 1. Look at the data to get some insight
    # check_data(data, targets)

    # 2. Build logistic regression model
    logistic_regression = LogisticRegressionModel(
        pipeline_steps=[StandardScaler(), LogisticRegression()]
    )

    # 3. KFold cross-validation to evaluate generalization performance of the model
    scores = logistic_regression.kfold_cross_validate(data, targets, 5)
    logistic_regression.print_kfold_cross_validation_accuracy(scores)

    scores = logistic_regression.kfold_cross_validate(data, targets, 10)
    logistic_regression.print_kfold_cross_validation_accuracy(scores)


if __name__ == "__main__":
    # Load adult census data as DataFrame and extract the target
    adult_census = dh.load_data_from_arff(DataPath.ADULT_CENSUS.value,
                                          TargetColumn.ADULT_CENSUS)

    first_model_with_scikit_learn(*adult_census)


    # Load numerical data from adult census data as DataFrame and extract the target
    data, targets = _load_numerical_data(adult_census[0]), adult_census[1]

    working_with_numerical_data(data, targets)
    preprocessing_for_numerical_features(data, targets)
    model_evaluation_using_cross_validation(data, targets)
