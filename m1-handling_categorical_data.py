from config import (
    DataPath,
    TargetColumn
)
from model import (
    GradientBoostingClassifierModel,
    LogisticRegressionModel
)
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder
)
from typing import TypeVar

import data_handler as dh
import pandas as pd


Tmodel = TypeVar('Tmodel',
                 GradientBoostingClassifierModel,
                 LogisticRegressionModel)


###########
# ENCODER #
###########
def _ordinal_encoder(data: pd.DataFrame, targets: pd.Series) -> None:
    """Try out ordinal encoder."""
    print("\nOrdinal encoding")

    # 1. Machine learning pipeline with a one-hot encoding of the data
    logistic_regression = LogisticRegressionModel(pipeline_steps=[
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        LogisticRegression(max_iter=500)]
        )

    # 2. KFold cross-validation to evaluate generalization performance of the model
    scores = logistic_regression.kfold_cross_validate(data, targets, 5)
    logistic_regression.print_kfold_cross_validation_accuracy(scores)

def _onehot_encoder_handle_unknown(data: pd.DataFrame, targets: pd.Series) -> None:
    """Try out one hot encoder with handle_unknown set to 'ignore'."""
    print("\nOne-hot encoding with handle_unknown set to 'ignore'")

    # 1. Machine learning pipeline with a one-hot encoding of the data
    logistic_regression = LogisticRegressionModel(pipeline_steps=[
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=500)]
        )

    # 2. KFold cross-validation to evaluate generalization performance of the model
    scores = logistic_regression.kfold_cross_validate(data, targets, 5)
    logistic_regression.print_kfold_cross_validation_accuracy(scores)

def _onehot_encoder_categories(data: pd.DataFrame, targets: pd.Series) -> None:
    """Try out one hot encoder by passing all the possible categories."""
    print("\nOne-hot encoding with passing all the possible categories.")

    # 1. Machine learning pipeline with a one-hot encoding of the data
    logistic_regression = LogisticRegressionModel(pipeline_steps=[
        OneHotEncoder(categories=dh.get_all_categories(data)),
        LogisticRegression(max_iter=500)]
        )

    # 2. KFold cross-validation to evaluate generalization performance of the model
    scores = logistic_regression.kfold_cross_validate(data, targets, 5)
    logistic_regression.print_kfold_cross_validation_accuracy(scores)

def _onehot_encoder_min_frequencies(data: pd.DataFrame, targets: pd.Series) -> None:
    """Try out one hot encoder by adjusting the min frequency parameter."""
    min_frequencies:float = [None, 0.01, 0.05, 0.5, 1]

    # Try out multiple min_frequency values to find a optimum one
    print("\nOne-hot encoding with various min_frequency value.")
    for min_frequency in min_frequencies:
        print(f"min_frequency={min_frequency}.")

        # 1. Machine learning pipeline with a one-hot encoding of the data
        logistic_regression = LogisticRegressionModel(pipeline_steps=[
            OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=min_frequency),
            LogisticRegression(max_iter=500)]
            )

        # 2. KFold cross-validation to evaluate generalization performance of the model
        scores = logistic_regression.kfold_cross_validate(data, targets, 5)
        logistic_regression.print_kfold_cross_validation_accuracy(scores)


####################
# CROSS-VALIDATION #
####################
def kfold_cross_validation(model: Tmodel,
                           data: pd.DataFrame,
                           targets: pd.Series,
                           nb_fold: int = 5):
    """Kfold cross validation."""
    scores = model.kfold_cross_validate(data, targets, nb_fold)
    model.print_kfold_cross_validation_accuracy(scores)


###########
# SECTION #
###########
def encoding_of_categorical_variables(data: pd.DataFrame,
                                      targets: pd.Series) -> None:
    """Encoding of categorical variables."""
    # Filtered out any non-numeric values
    data = dh.get_subset(data, dtypes=[str])

    one_hot_encoding_test = False
    if one_hot_encoding_test:
        # Apply a one-hot encoding
        data_encoded = dh.one_hot_encoder(data, sparse_output=False)

        # Display result of the encoding
        print(data.head())
        print(data_encoded.head())

    # Try out ordinal encoder
    _ordinal_encoder(data, targets) # kinda bad

    # Try out one hot encoder
    #   - With handle_unknown set to 'ignore'
    _onehot_encoder_handle_unknown(data, targets) # pretty good
    #   - By passing all the possible categories
    _onehot_encoder_categories(data, targets) # pretty good
    #   - By adjusting the min frequency parameter
    _onehot_encoder_min_frequencies(data, targets) # various performance depending of min_frequency


def linear_model_with_heterogeneously_data_type(data: pd.DataFrame,
                                                targets: pd.Series) -> None:
    """
    Using numerical and categorical variables together to train a linear model.

    For linear model it's required to:
    - Encode catogerical variables (with one-hot encoding for example)
    - Scale numerical variables
    """
    # Build a pipeline with a column transformer in order to deal with
    # numerical and categorical variables
    model = LogisticRegressionModel.build_pipeline_with_transformer(
        transformers = [
            (OneHotEncoder(handle_unknown="ignore"), selector(dtype_include=str)),
            (StandardScaler(), selector(dtype_exclude=str))
        ],
        model = LogisticRegression(max_iter=500)
    )

    # KFold cross-validation to evaluate generalization performance of the model
    kfold_cross_validation(model, data, targets)

def treebased_model_with_heterogeneously_data_type(data: pd.DataFrame,
                                                   targets: pd.Series) -> None:
    """
    Using numerical and categorical variables together to train a tree-based model.

    For tree-based model, it's only required to encode categorical variables.
    Therefore, use special string 'passthrough' to indicate to pass the numerical columns
    through untransformed.

    Be aware that the current implementation of HistGradientBoostingClassifier is still
    incomplete. For example, does not yet support sparse input data.
    """
    # Build a pipeline with a column transformer in order to deal with
    # numerical and categorical variables
    print("\nReference pipeline with no numerical scaling and integer-coded categories")
    model = GradientBoostingClassifierModel.build_pipeline_with_transformer(
        transformers = [
            (
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                selector(dtype_include=str)
            ),
            ("passthrough", selector(dtype_exclude=str))
        ],
        model = HistGradientBoostingClassifier()
    )
    kfold_cross_validation(model, data, targets)

    # Just to prove that perform numerical scaling with based-tree model is not required
    # Do not improve the accuracy and time difference is not significant
    print("\nPipeline with numerical scaling and integer-coded categories")
    model = GradientBoostingClassifierModel.build_pipeline_with_transformer(
        transformers = [
            (
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                selector(dtype_include=str)
            ),
            (StandardScaler(), selector(dtype_exclude=str))
        ],
        model = HistGradientBoostingClassifier()
    )
    kfold_cross_validation(model, data, targets)

    # Just to prove that using a one-hot encoding with based-tree model is NOT REQUIRED AND KINDA
    # BAD. Does not affect the accuracy and improve the fit duration probably due to
    # sparce_output=False as a workaround because HistGradientBoostingClassifier does not yet
    # support sparse input data
    print("\nPipeline with no numerical scaling and one-hot encoded categories")
    model = GradientBoostingClassifierModel.build_pipeline_with_transformer(
        transformers = [
            (
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                selector(dtype_include=str)
            ),
            ("passthrough", selector(dtype_exclude=str))
        ],
        model = HistGradientBoostingClassifier()
    )
    kfold_cross_validation(model, data, targets)


def treebased_model_with_mix_encoder(data: pd.DataFrame,
                                     targets: pd.Series) -> None:
    """
    Using numerical and categorical variables together to train a tree-based model with
    a target encoder which is well suited for nominal, categorical features with high cardinality.
    This encoding scheme is useful with categorical features with high cardinality, where one-hot
    encoding would inflate the feature space making it more expensive for a downstream model to
    process. A classical example of high cardinality categories are location based such as zip code
    or region.
    """
    # Get categorical features with high cardinality
    high_cardinality, low_cardinality = \
        dh.get_cardinality_features(dh.get_subset(data, dtypes=[str]))

    # Build a pipeline with a column transformer in order to deal with
    # numerical and categorical variables (use different encoder depending
    # of data cardinality)
    model = GradientBoostingClassifierModel.build_pipeline_with_transformer(
        transformers = [
            (TargetEncoder(target_type="auto"), high_cardinality.columns.to_list()),
            (
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                low_cardinality.columns.to_list()
            ),
            ("passthrough", selector(dtype_exclude=str))
        ],
        model = HistGradientBoostingClassifier()
    )
    kfold_cross_validation(model, data, targets)


if __name__ == "__main__":
    # Load data
    adult_census = dh.load_data_from_arff(DataPath.ADULT_CENSUS.value,
                                          TargetColumn.ADULT_CENSUS)

    encoding_of_categorical_variables(*adult_census)
    linear_model_with_heterogeneously_data_type(*adult_census)
    treebased_model_with_heterogeneously_data_type(*adult_census)
    treebased_model_with_mix_encoder(*adult_census)
