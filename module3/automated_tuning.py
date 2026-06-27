from pathlib import Path

import data_handler as dh
import pandas as pd
from config import DataPath, TargetColumn
from model import HistGradientBoostingClassifierModel
from scipy.stats import loguniform
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder
from types_config import SearchCvParameters, SearchOuterCv
from visualisation import show_parallel_coordinates_for_hyperparameter_tuning


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
def build_gradient_boosting_classifier() -> HistGradientBoostingClassifierModel:
    """Build a gradient boosting classifier."""
    return HistGradientBoostingClassifierModel.build_pipeline(
        transformers=[
            (
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                selector(dtype_include=str)
            ),
            ("passthrough", selector(dtype_exclude=str))
        ],
        random_state=42,
        max_leaf_nodes=4
    )


#########################
# HYPERPARAMETER TUNING #
#########################
def grid_search_tuning(model: HistGradientBoostingClassifierModel,
                       x_data: pd.DataFrame,
                       y_data: pd.Series,
                       x_train: pd.DataFrame,
                       y_train: pd.Series,
                       path: Path) -> None:
    """
    Hyperparameter tuning by grid-search.

    For tree-based models, ordinal encoder avoids having high-dimensional representations.

    - learning_rate: control the ability of a new tree to correct the error of the previous
    sequence of trees.
      . A gradient boosting model with large learning rate will tend to overfit. It is due to the
        fact that the sequence of added trees will rapidly correct the residuals and thus will
        fit noisy samples. A learning rate larger than 1 can even make the optimization problem
        diverge.
      . On an other hand, setting a low learning rate will pervent the model to miminimize the loss
        even on the training set and therefore will cause underfitting.

    - max_leaf_nodes: control the depth of each tree
    """
    # 1. Set up the parameter grid used by the grid-search algorithm
    #    Explicitly specified hyperparameter values to try out
    param_grid = {
        "histgradientboostingclassifier__learning_rate": (0.01, 0.1, 1, 10),
        "histgradientboostingclassifier__max_leaf_nodes": (3, 10, 30)
    }

    # 2. Tune hyperparameters
    model.automated_search_cv(search_class=GridSearchCV,
                              search_params=SearchCvParameters(2),
                              parameters=param_grid,
                              x_train=x_train,
                              y_train=y_train,
                              path=path,
                              search_outer_cv=SearchOuterCv(x_data, y_data))

def randomized_search_tuning(model: HistGradientBoostingClassifierModel,
                             x_data: pd.DataFrame,
                             y_data: pd.Series,
                             x_train: pd.DataFrame,
                             y_train: pd.Series,
                             path: Path) -> None:
    """
    Hyperparameter tuning by randomized-search.

    - learning_rate: control the ability of a new tree to correct the error of the
                   previous sequence of trees
    - max_leaf_nodes: control the depth of each tree
    - l2_regularization: control the strength of the regulation
    - min_sample_leaf: control the minimum number of samples required in a leaf
    - max_bins: control the maximum number of bins to construct the histograms
    """
    # 1. Set up the parameter distribution used by the randomized-search algorithm
    #    Specified a range of hyperparameter values to try out
    param_dist = {
        "histgradientboostingclassifier__l2_regularization": loguniform(1e-6, 1e3),
        "histgradientboostingclassifier__learning_rate": loguniform(0.001, 10),
        "histgradientboostingclassifier__max_leaf_nodes": loguniform_int(2, 256),
        "histgradientboostingclassifier__min_samples_leaf": loguniform_int(1, 100),
        "histgradientboostingclassifier__max_bins": loguniform_int(2, 255)
    }

    # 2. Tune hyperparameters. For performance issue only run 10 iterations.
    #    In order to make a decent analysis, at least 500 iterations would be best.
    model.automated_search_cv(search_class=RandomizedSearchCV,
                              search_params=SearchCvParameters(5, n_iter=20),
                              parameters=param_dist,
                              x_train=x_train,
                              y_train=y_train,
                              path=path,
                              search_outer_cv=SearchOuterCv(x_data, y_data))


############
# ANALYSIS #
############
def run_analysis():
    # 1. Load data
    adult_census = dh.load_data_from_file(DataPath.ADULT_CENSUS.value,
                                          TargetColumn.ADULT_CENSUS)

    # 2. Split data into random train and test subsets
    train_data = dh.get_train_split(dh.sklearn_train_test_split(**adult_census, test_size=0.8))

    # 3. Build the classifier model
    model = build_gradient_boosting_classifier()

    # 4. Tuning using various method
    path = Path(*DataPath.HYPERPARAMETER_TUNING.value.parts + ("grid_search.csv",))
    # grid_search_tuning(model, *adult_census, x_train, y_train, path)
    # show_parallel_coordinates_for_hyperparameter_tuning(pd.read_csv(path))

    path = Path(*DataPath.HYPERPARAMETER_TUNING.value.parts + ("randomized_search.csv",))
    randomized_search_tuning(model,
                             adult_census["x_data"],
                             adult_census["y_data"],
                             *train_data,
                             path)
    show_parallel_coordinates_for_hyperparameter_tuning(pd.read_csv(path))

if __name__ == "__main__":
    run_analysis()
