import data_handler as dh
import pandas as pd
from config import DataPath, TargetColumn
from model import GradientBoostingClassifierModel, LogisticRegressionModel
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from visualisation import show_errorbars_for_hyperparameter_tuning


#######################
# LOGISTIC REGRESSION #
#######################
def manual_logistic_regression_tuning(data: pd.DataFrame,
                                      targets: pd.Series) -> None:
    """Manual tuning of hyperparameter C of a logistic regression."""
    # 1. Remove categorical features
    data = dh.get_subset(data, dtypes=[int, float])

    # 2. Build the model
    model = LogisticRegressionModel.build_pipeline([StandardScaler()], max_iter=500)

    # 3. Tuning of hyperparameter C
    hyperparameters = {
        "logisticregression__C": [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]
    }
    scores = model.manual_hyperparameter_tuning(data, targets, **hyperparameters)

    # 4. Show scores
    show_errorbars_for_hyperparameter_tuning(
        {"mean": scores["testing_mean"], "std": scores["testing_std"]},
        hyperparameters["logisticregression__C"],
        xlabel="C",
        train_scores={"mean": scores["training_mean"], "std": scores["training_std"]}
    )


################################
# GRADIENT BOOSTING CLASSIFIER #
################################
def _build_gradient_boosting(learning_rate:float = 0.1,
                             max_leaf_nodes: int = 31) -> GradientBoostingClassifierModel:
    """
    Build a gradient boosting classifier with a transformer to deal with numerical and
    categorical features.
    """
    return GradientBoostingClassifierModel.build_pipeline(
        transformers=[
            (
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                selector(dtype_include=str)
            ),
            ("passthrough", selector(dtype_exclude=str))
        ],
        random_state=42,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes
    )


def _display_errors(scores: dict[str, list[float]]):
    """Display manual tuning results."""
    data = pd.DataFrame({k.rsplit('__')[-1]: v for k, v in scores.items()})\
             .sort_values("testing_mean", ascending=False)
    print(data)

def manual_gradient_boosting_tuning(data: pd.DataFrame,
                                    targets: pd.Series,
                                    train_size: float) -> None:
    """
    Manual tuning of hyperparameters learning_rate and max_leaf_nodes of
    a gradient boosting classifier.

    - learning_rate: control the ability of a new tree to correct the error of the
                   previous sequence of trees
    - max_leaf_nodes: control the depth of each tree
    """
    # 1. Split data. At first to work on a smaller subset then to tuned the
    #    hyperparameter on the training set
    x_train, _, y_train, _ = \
        dh.sklearn_train_test_split(data, targets, test_size=1-train_size)

    # 2. Build model
    model = _build_gradient_boosting()

    # 3. Hyperparameters tuning
    hyperparameters = {
        "histgradientboostingclassifier__learning_rate": [0.01, 0.1, 1, 10],
        "histgradientboostingclassifier__max_leaf_nodes": [3, 10, 30]
    }
    scores = model.manual_hyperparameter_tuning(x_train, y_train, **hyperparameters)

    _display_errors(scores)


def optimum_gradient_boosting(data: pd.DataFrame,
                              targets: pd.Series,
                              learning_rate: float,
                              max_leaf_nodes: int) -> None:
    """Gradient boosting with best hyperparameters."""
    # 1. Build model with default hyperparameters
    model = _build_gradient_boosting(learning_rate=learning_rate,
                                     max_leaf_nodes=max_leaf_nodes)

    # 2. Cross-validation
    scores = model.kfold_cross_validate(data, targets, 5, return_train_score=True)
    model.print_kfold_cross_validation_accuracy(scores)


############
# ANALYSIS #
############
def run_analysis():
    adult_census = dh.load_data_from_file(DataPath.ADULT_CENSUS.value, TargetColumn.ADULT_CENSUS)

    # Logistic regression manual tuning
    manual_logistic_regression_tuning(*adult_census)

    # Gradient boosting classifier manual tuning
    #manual_gradient_boosting_tuning(*adult_census, train_size=0.2)
    #manual_gradient_boosting_tuning(*adult_census, train_size=0.8)

    # print("\nGradient boosting with optimum hyperparameters.")
    #optimum_gradient_boosting(*adult_census, learning_rate=0.1, max_leaf_nodes=30)

if __name__ == "__main__":
    run_analysis()
