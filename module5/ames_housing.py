import data_handler as dh
import numpy as np
import pandas as pd
from config import AMES_HOUSING_NUMERICAL_FEATURES, DataPath, TargetColumn
from model import DecisionTreeRegressorModel, LinearRegressionModel
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from types_config import DataSetType, Tpreprocessor

type ClassModelTypes = (DecisionTreeRegressorModel | LinearRegressionModel)

########
# DATA #
########
def load_data(columns: list[str] |None = None) -> DataSetType:
    """
    Load ames housing dataset.

    Parameters
    ----------
    columns: list of features to select
    """
    return dh.load_data_from_file(DataPath.AMES_HOUSING.value,
                                  target=TargetColumn.AMES_HOUSING,
                                  columns=columns,
                                  drop_na=True)


#########
# MODEL #
#########
def linear_regression(x_data: pd.DataFrame,
                      y_data: pd.Series,
                      transformers: list[Tpreprocessor]):
    """
    Perform a linear regression.

    Parameters
    ----------
    x_data: the whole features

    y_data: the whole targets
    """
    # Build model
    regression: LinearRegressionModel = LinearRegressionModel.build_pipeline(transformers)

    # Cross-validation
    scores = regression.make_cross_validate(x_data, y_data, nb_fold=10)
    regression.print_cross_validate(scores, verbose=True)

def decision_tree(x_data: pd.DataFrame,
                  y_data: pd.Series,
                  random_state: int = 0,
                  max_depth: int | None = None,
                  tune_maxdepth: bool = False,
                  transformers: list[Tpreprocessor] | None = None):
    """
    Perform a decision tree.

    Parameters
    ----------
    x_data: the whole features

    y_data: the whole targets
    """
    # Build model
    if transformers is None:
        regression: DecisionTreeRegressorModel = DecisionTreeRegressorModel.build(
            random_state=random_state, max_depth=max_depth
        )
    else:
        regression: DecisionTreeRegressorModel = DecisionTreeRegressorModel.build_pipeline(
            transformers=transformers, random_state=random_state, max_depth=max_depth
        )

    # Default cross-validation
    scores = regression.make_cross_validate(x_data, y_data, nb_fold=10)
    regression.print_cross_validate(scores, verbose=True)

    if tune_maxdepth:
        # Split data
        split_data = dh.sklearn_train_test_split(x_data, y_data)

        # Grid-search cv to tune max_depth parameter
        regression.automated_search_cross_validation(GridSearchCV,
                                                     {"max_depth": np.arange(1, 15, 1)},
                                                     x_data,
                                                     y_data,
                                                     split_data["x_train"],
                                                     split_data["y_train"],
                                                     cv=10)


############
# ANALYSIS #
############
def run_analysis_with_numerical_only():
    """
    Run analysis using numerical data only. In this case, the linear model outperformed the
    decision tree most of the time even with a tuned max_depth.
    """
    # Laod data
    housing = load_data(columns=AMES_HOUSING_NUMERICAL_FEATURES)

    # Linear regression
    linear_regression(**housing, transformers=[StandardScaler()])

    # Decision tree with defaut hyperparameter value and tune max_depth
    decision_tree(**housing, tune_maxdepth=True)

    # Decision tree with tuned hyperparameter
    decision_tree(**housing, max_depth=6)
    decision_tree(**housing, max_depth=6, random_state=1)
    decision_tree(**housing, max_depth=6, random_state=2)

def run_analysis_with_all():
    """Run analysis using numerical and categorical data."""
    # Laod data
    housing = load_data()

    # Linear regression
    linear_regression(
        **housing,
        transformers=[
            (StandardScaler(), selector(dtype_exclude=str)),
            (
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                selector(dtype_include=str)
            )
        ]
    )

    # Decision tree
    decision_tree(
        **housing,
        transformers=[
            ("passthrough", selector(dtype_exclude=str)),
            (
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                selector(dtype_include=str)
            )
        ],
        max_depth=7,
        random_state=2
    )

def run_analysis():
    """Use decision tree with ames housing dataset."""
    run_analysis_with_numerical_only()
    run_analysis_with_all()

if __name__ == "__main__":
    run_analysis()
