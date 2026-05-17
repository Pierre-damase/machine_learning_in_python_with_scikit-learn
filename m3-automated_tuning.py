from config import (
    DataPath,
    TargetColumn
)
from model import GradientBoostingClassifierModel
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import loguniform

import data_handler as dh
import pandas as pd


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


#########
# MODEL #
#########
"""Build a gradient boosting classifier."""
def build_gradient_boosting_classifier() -> GradientBoostingClassifierModel:
    return GradientBoostingClassifierModel.build_pipeline_with_transformer(
        transformers = [
            (
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                selector(dtype_include=str)
            ),
            ("passthrough", selector(dtype_exclude=str))
        ],
        classifier = HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4)
    )


#########################
# HYPERPARAMETER TUNING #
#########################
"""
Hyperparameter tuning by grid-search.

For tree-based models, ordinal encoder avoids having high-dimensional representations.

  - learning_rate: control the ability of a new tree to correct the error of the
                   previous sequence of trees
  - max_leaf_nodes: control the depth of each tree
"""
def grid_search_tuning(model: GradientBoostingClassifierModel,
                       data: list[pd.DataFrame|pd.Series]) -> None:

    # 1. Set up the parameter grid used by the grid-search algorithm
    #    Explicitly specified hyperparameter values to try out
    param_grid = {
        "histgradientboostingclassifier__learning_rate": (0.01, 0.1, 1, 10),
        "histgradientboostingclassifier__max_leaf_nodes": (3, 10, 30)
    }

    # 2. Tune hyperparameters
    model.automated_search_cross_validation(GridSearchCV, param_grid, *data, verbose=True, cv=2)

"""
Hyperparameter tuning by randomized-search.

  - learning_rate: control the ability of a new tree to correct the error of the
                   previous sequence of trees
  - max_leaf_nodes: control the depth of each tree
  - l2_regularization: control the strength of the regulation
  - min_sample_leaf: control the minimum number of samples required in a leaf
  - max_bins: control the maximum number of bins to construct the histograms
"""
def randomized_search_tuning(model: GradientBoostingClassifierModel,
                             data: list[pd.DataFrame|pd.Series]) -> None:
    # 1. Set up the parameter distribution used by the randomized-search algorithm
    #    Specified a range of hyperparameter values to try out
    param_distribution = {
        "histgradientboostingclassifier__l2_regularization": loguniform(1e-6, 1e3),
        "histgradientboostingclassifier__learning_rate": loguniform(0.001, 10),
        "histgradientboostingclassifier__max_leaf_nodes": loguniform_int(2, 256),
        "histgradientboostingclassifier__min_samples_leaf": loguniform_int(1, 100),
        "histgradientboostingclassifier__max_bins": loguniform_int(2, 255)
    }

    # 2. Tune hyperparameters. For performance issue only run 10 iterations.
    #    In order to make a decent analysis, at least 500 iterations would be best.
    model.automated_search_cross_validation(
        RandomizedSearchCV, param_distribution, *data, verbose=True, cv=5, n_iter=10
    )


if __name__ == "__main__":
    # 1. Load data
    adult_census = dh.load_data_from_arff(DataPath.ADULT_CENSUS.value,
                                          TargetColumn.ADULT_CENSUS)

    # 2. Split data into random train and test subsets
    adult_census_split = dh.sklearn_train_test_split(*adult_census, test_size=0.8)

    # 3. Build the model
    model = build_gradient_boosting_classifier()

    # 4. Tuning using various method
    # grid_search_tuning(model, adult_census_split)
    randomized_search_tuning(model, adult_census_split)
