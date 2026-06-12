from os import path
from pathlib import Path
from typing import Literal

import data_handler as dh
import pandas as pd
from config import DataPath
from model import (GradientBoostingRegressorModel,
                   HistGradientBoostingRegressorModel,
                   RandomForestRegressorModel)
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from types_config import (CvParameters, DataSetType, SearchCvHyperparamType,
                          SearchCvParameters, SearchCvType, SplitSetType)

SCORING = "neg_mean_absolute_error"

########
# DATA #
########
def load_californian_housing() -> DataSetType:
    """Load californian housing dataset."""
    return dh.load_california_dataset()


#########
# MODEL #
#########
def randomized_search_cv(regressor: GradientBoostingRegressorModel | RandomForestRegressorModel,
                         search_class: type[SearchCvType],
                         search_params: SearchCvParameters,
                         parameters: SearchCvHyperparamType,
                         x_train: pd.DataFrame,
                         x_test: pd.DataFrame,
                         y_train: pd.Series,
                         y_test: pd.Series,
                         path: Path | None = None) -> SearchCvType:
    """Randomized search cv."""
    # Hyperparameters tuning
    tuned_regressor = regressor.automated_search_cv(search_class=search_class,
                                                    search_params=search_params,
                                                    parameters=parameters,
                                                    x_train=x_train,
                                                    y_train=y_train,
                                                    scoring=SCORING,
                                                    path=path)

    # Predict the testing data using the tuned model
    regressor.predict_tuned_model(tuned_regressor, x_test, y_test, scoring=SCORING)

    return tuned_regressor


############
# ANALYSIS #
############
def run_hist_gradient_boosting_tuning(split_data: SplitSetType,
                                      search_type: Literal["grid-search", "randomized-search"],
                                      x_data: pd.DataFrame,
                                      y_data: pd.Series):
    """
    Tune histogram gradient-boosting hyperparameters.

    On of the most important hyperparameters for histogram gradient-boosting is max_iter, which
    controls the number of trees in the estimator, and the stopping criteria, which stops the
    algorithms when there is no improvement for some time. Therefore, the final number of trees
    can be lower than the number set by the user (contrary to random-forest).

    Hyperparameters that control the tree structure
    - max_depth: a low depth up to 8 levels at most
    - max_leaf_nodes: few leaves up to 256
    The boosting algorithm fit the error of the previous tree in the ensemble. Thus, fitting a
    fully grown trees would be detrimental. For instance, start the algorithm with a fully grown
    tree will cause the first tree of the ensemble to perfectly fit the data and thus no subsequent
    tree would be required, since residuals are really low or even zero.

    learning_rate controls how much each correction contributes to the final prediction. A smaller
    learning-rate means the corrections of a new tree result in small adjustments to the model
    prediction. On the other hand, a higher learning rate makes larger adjustement with each tree.
    """
    # Buid model
    regressor = HistGradientBoostingRegressorModel.build()

    # Hyperparameters tuning
    match search_type:
        case "grid-search":
            # 1\ Set paramter grid
            param_grid = {"max_depth": [3, 8, 16],
                          "max_leaf_nodes": [15, 31, 62],
                          "learning_rate": [0.1, 0.5, 1]}

            # 2\ Grid-search cross-validation.
            file_name = "hist_boosting_gradient.csv"
            path = Path(*DataPath.HYPERPARAMETER_TUNING.value.parts + (file_name,))
            tuned_regressor = randomized_search_cv(
                regressor, GridSearchCV, SearchCvParameters(5), param_grid, **split_data, path=path
            )

            # 3\ Cross-validation with the tuned model
            #cv = KFold(n_splits=5, shuffle=True, random_state=0)
            #cross_validate(tuned_regressor, x_data, y_data, cv=cv, return_estimator=True)
            #scores = regressor.make_cross_validate(x_data,
            #                                       y_data,
            #                                       "KFold",
            #                                       CvParameters(5, shuffle=True, random_state=0),
            #                                       scoring=SCORING,
            #                                       return_estimator=True,
            #                                       model=tuned_regressor)
            #regressor.print_cross_validate(scores, verbose=True)

        case "randomized-search":
            # 1\ Set parameter distributions. It's better to fix a large max_iter and tune
            # early_stopping.
            param_dist = {"max_iter": [3, 10, 30, 100, 300, 1000],
                          "max_leaf_nodes": [2, 5, 10, 20, 50, 100],
                          "learning_rate": loguniform(0.01, 1),}

            # 2\ Randomized search cross-validation. The best set of parameters are max_iter=100,
            # max_leaf_nodes=20 and learning_rate=0.197 with a testing error of 31.3 k$. 20
            randomized_search_cv(regressor,
                                 RandomizedSearchCV,
                                 SearchCvParameters(5, n_iter=20),
                                 param_dist,
                                 **split_data)

def run_random_forest_tuning(split_data: SplitSetType):
    """
    Tune random-forest hyperparameters.

    The most important parameter to tune is max_features, which controls the randomness:
    - Too much randomness in the trees lead to underfitted base models and can be detrimental for
    the ensemble as a whole.
    - Too few randomness in the trees leads to more correlation of the prediction errors and as a
    result reduce the benefits of the averaging step in terms of overfitting control.
    - A randomness set to None means that the only source of randomness in the random-forest is the
    bagging procedure.

    Hyperparameters that control the tree structure
    - max_depth enforces growing symmetric trees
    - max_leaf_nodes

    min_samples_leaf controls the minimum number of samples required to be at a leaf node. In other
    words, a split point at any depth is only done if it leaves enough training samples in each
    branches (left and right):
    - Too low value may lead to isolate sample in deep tree, promoting overfitting
    - Too high value would prevent deep trees, which can lead to underfitting

    More trees improve the generalization performance but slow down the fitting / prediction time.
    Tuning n_estimators generally result in a waste of computer power. We just need to ensure that
    it's large enough so that doubling its value does not lead to a significant improvement of the
    validation error.

    10 iterations is a good start to quickly inspect the hyperparameters combinations without
    spending too much computational ressources.

    In this example, the most important parameter is max_leaf_nodes that must be high and thus
    means deep trees in the forest.
    """
    # 1\ Set parameter distributions
    param_dist = {"max_features": [1, 2, 3, 5, None],
                  "max_leaf_nodes": [10, 100, 1000, None],
                  "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100]}

    # 2\ Randomized search cross-validation. The best set of parameters are max_features=5,
    # max_leaf_nodes=None and min_samples_leaf=2 with a testing error of 32.5 k$.
    regressor = RandomForestRegressorModel.build(n_jobs=2)
    randomized_search_cv(
        regressor, GridSearchCV, SearchCvParameters(5, n_iter=10), param_dist, **split_data
    )

def run_analysis():
    """Tune hyperparameter for ensemble model."""
    # Load data
    housing = load_californian_housing()

    # Split data
    split_data = dh.sklearn_train_test_split(**housing)

    # Random-forest tuning
    # run_random_forest_tuning(split_data)

    # Histogram gradient-boosting tuning
    # run_hist_gradient_boosting_tuning(split_data, search_type="randomized-search", **housing)
    run_hist_gradient_boosting_tuning(split_data, search_type="grid-search", **housing)

if __name__ == "__main__":
    run_analysis()
