import data_handler as dh
import pandas as pd
from config import DataPath, TargetColumn
from model import (DecisionTreeClassifierModel, LinearRegressionModel,
                   LogisticRegressionModel)
from sklearn.compose import make_column_selector as selector
from sklearn.metrics import make_scorer, precision_score
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from types_config import (CvParameters, DataSetType, ScoringFunctionType,
                          Tpreprocessor)

type ModelClassTypes = (DecisionTreeClassifierModel
                        | LinearRegressionModel
                        | LogisticRegressionModel)

########
# DATA #
########
def load_blood_dataset() -> DataSetType:
    """Load blood dataset."""
    return dh.load_data_from_file(DataPath.BLOOD_TRANSFUSION.value, TargetColumn.BLOOD_TRANSFUSION)

def load_housing_dataset() -> DataSetType:
    """Load ames housing dataset."""
    data = dh.load_data_from_file(DataPath.AMES_HOUSING.value,
                                  TargetColumn.AMES_HOUSING,
                                  drop_na=True)
    data["y_data"] /= 1000
    return data


#########
# MODEL #
#########
def build_model(model_class: type[ModelClassTypes],
                x_train: pd.DataFrame,
                y_train: pd.Series,
                transformers: list[Tpreprocessor] | None = None) -> ModelClassTypes:
    """Build a model, train it and predict the testing data."""
    # Build model
    if transformers is None:
        model: ModelClassTypes = model_class.build()
    else:
        model: ModelClassTypes = model_class.build_pipeline(transformers=transformers)

    # Train the model
    model.start(x_train=x_train, y_train=y_train)

    return model


############
# ANALYSIS #
############
def run_classification_metrics():
    """Try out various metrics for classification task."""
    # Load data
    blood = load_blood_dataset()
    split_data = dh.sklearn_train_test_split(**blood, test_size=0.5, random_state=0)

    # Build model
    classifier = build_model(LogisticRegressionModel, split_data["x_train"], split_data["y_train"])

    # Evaluate the generalization performance of the model with various metrics
    classifier.evaluate_generalization_performance(split_data["x_test"],
                                                   split_data["y_test"],
                                                   pos_label="donated")

    # Build logistic regression
    tree = build_model(DecisionTreeClassifierModel, split_data["x_train"], split_data["y_train"])
    for scoring in ["accuracy", # default metric, thus passing None is equivalent
                    "balanced_accuracy",
                    make_scorer(precision_score, pos_label="donated")]:
        scores = tree.make_cross_validate(**blood,
                                          cv_strategy="StratifiedKFold",
                                          cv_params=CvParameters(10),
                                          scoring=scoring)
        tree.print_cross_validate(scores)

def run_regression_metrics():
    """
    Try out various metrics for regression task.

    Performing target transformation, such as quantile transformation to numerical values, is often
    disapproved for linear regression by statisticians. It's more suitable to use model such as
    PoissonRegressor or TweddieRegressor.
    """
    # Load data
    housing = load_housing_dataset()
    split_data = dh.sklearn_train_test_split(**housing, test_size=0.25, random_state=0)

    # 1st transfomer. Ignore sring targets and no transformation.
    # 2nd transformer. Ignore string targets and use quantile transformer for numerical value.
    # 3rd transformer. Add a one hot encoder to string value.
    transformers = [
        [
            ("passthrough", selector(dtype_exclude="str"))
        ],
        [
            (
                QuantileTransformer(n_quantiles=900, output_distribution="normal"),
                selector(dtype_exclude="str")
            )
        ],
        [
            (OneHotEncoder(handle_unknown="ignore"), selector(dtype_include="str")),
            (
                QuantileTransformer(n_quantiles=900, output_distribution="normal"),
                selector(dtype_exclude="str")
            )
        ]
    ]

    # Scoring for cross-validation
    scoring: list[ScoringFunctionType] = ["r2", "neg_mean_absolute_error"]
    for transformer in transformers:
        # Build model
        regressor = build_model(LinearRegressionModel,
                                split_data["x_train"],
                                split_data["y_train"],
                                transformers=transformer)

        # Cross-validation
        for i in scoring:
            scores = regressor.make_cross_validate(**housing, cv_params=CvParameters(10), scoring=i)
            regressor.print_cross_validate(scores)

        # Evaluate the generalization performance of the model with various metrics. The model
        # tends to under-estimate the price of the house.
        regressor.evaluate_generalization_performance(split_data["x_test"],
                                                      split_data["y_test"],
                                                      x_train=split_data["x_train"],
                                                      y_train=split_data["y_train"])


def run_analysis():
    """Various metrics help to evaluate the generalization performance of a model."""
    run_classification_metrics()
    run_regression_metrics()

if __name__ == "__main__":
    pass
