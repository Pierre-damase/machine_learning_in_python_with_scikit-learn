from typing import TypeVar

import data_handler as dh
import numpy as np
import pandas as pd
from config import DataPath, TargetColumn
from model import (LinearRegressionModel, LogisticRegressionModel,
                   RidgeRegressionModel, RidgeRegressionModelCV)
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from types_config import CvResults, DataSetType
from visualisation import (plot_coefficients_of_linear_model,
                           show_errorbars_for_hyperparameter_tuning)

FEATURES = ["LotFrontage", "LotArea", "PoolArea", "YearBuilt", "YrSold"]

type ClassModelTypes = (LinearRegressionModel | LogisticRegressionModel)
Tclassmodel = TypeVar('Tclassmodel', bound=ClassModelTypes)


########
# DATA #
########
def load_data() -> DataSetType:
    """Load ames housing dataset."""
    data, targets = dh.load_data_from_file(DataPath.AMES_HOUSING.value, TargetColumn.AMES_HOUSING)
    return dh.get_subset(data, FEATURES), targets


####################
# CROSS-VALIDATION #
####################
def cross_validation(model: ClassModelTypes,
                     x_data: pd.DataFrame,
                     y_data: pd.Series,
                     nb_fold: int | None = None,
                     n_splits: int | None = None,
                     test_size: float | None = None) -> CvResults:
    scores = model.make_cross_validate(x_data,
                                       y_data,
                                       nb_fold=nb_fold,
                                       n_splits=n_splits,
                                       test_size=test_size,
                                       scoring="neg_mean_squared_error",
                                       return_estimator=True,
                                       return_train_score=True)
    model.print_cross_validate(scores)
    return scores


############
# ANALYSIS #
############
def run_simple_linear_regression(data: pd.DataFrame, targets: pd.Series) -> None:
    """
    Build a simple linear regression.

    By default, the training error is in average one order of magnitude lower than the testing
    error, which indicates overfitting. Some coefficients are extremly large while others are
    extremly small, yet non-zero. Furthermore, the coefficient values can be very unstable across
    cv folds.
    """
    regression = LinearRegressionModel.build_pipeline(
        [PolynomialFeatures(degree=2, include_bias=False)]
    )
    scores = cross_validation(regression, data, targets, nb_fold=10)
    plot_coefficients_of_linear_model(regression.get_coefficients(scores))

def run_simple_ridge_regression(data: pd.DataFrame, targets: pd.Series, solver: str) -> None:
    """
    Use ridge regression to force the linear regression model to consider all features in a more
    homogeneous manner. This process is called regularization.

    Using ridge regression may display lot of warning depending of the solver because the features
    included both extremly large and extremly small values, which are causing numerical problems
    when training the predictive model.

    The training and testing error get closer tehrefore the model is less overfitting but still.
    The overall magnitudes of the weights are shrunk, yet non-zero. But, the weights' values remain
    unstable from one to another.

    Keep in mind that scalind the data and tune the regularization parameter is imporant in order
    to get better result.
    """
    ridge = RidgeRegressionModel.build_pipeline([PolynomialFeatures(degree=2, include_bias=False)],
                                                alpha=100,
                                                solver=solver)
    scores = cross_validation(ridge, data, targets, nb_fold=10)
    plot_coefficients_of_linear_model(ridge.get_coefficients(scores))

def run_ridge_regression_with_scaling(data: pd.DataFrame, targets: pd.Series, solver: str) -> None:
    """
    Perform ridge regression after data scaling. Data scaling may help regularization to stay
    neutral and treat approximately equally each feature.

    Compared to the previous model (ridge regression without data scaling), most weight magnitudes
    have a similar order of magnitude, i.e. they are equally contributing. And the number of
    unstable weights also decreased as well.
    """
    ridge = RidgeRegressionModel.build_pipeline(
        [
            MinMaxScaler(),
            PolynomialFeatures(degree=2, include_bias=False)
        ],
        alpha=10,
        solver=solver
    )
    scores = cross_validation(ridge, data, targets, nb_fold=20)
    plot_coefficients_of_linear_model(ridge.get_coefficients(scores))

def run_ridge_regression_with_tuning(data: pd.DataFrame, targets: pd.Series) -> None:
    """
    Hyperparameter tuning must be done on each dataset. Therefore, for ridge regression it's
    required to tune the parameter alpha.

    Tuning alpha help to drastically reduce the gap between the training and testing error, which
    indicates that the model is not overfitting anymore or less.
    """
    # Set alpha parameter
    alphas = np.logspace(-7, 5, num=100)

    # Build model
    ridge: RidgeRegressionModelCV = RidgeRegressionModelCV.build_pipeline(
        [
            MinMaxScaler(),
            PolynomialFeatures(degree=2, include_bias=False)
        ],
        alphas=alphas,
        store_cv_results=True
    )

    # Cross-validation + display the testing error of the inner cv
    scores = cross_validation(ridge, data, targets, n_splits=50, test_size=0.2)
    ridge.print_best_alpha_from_cv_estimator(scores)
    show_errorbars_for_hyperparameter_tuning(ridge.get_mean_cv_results(scores, alphas),
                                             alphas,
                                             xlabel="Alpha",
                                             xscale="log",
                                             yscale="log")

def run_analysis():
    # Load data
    housing = load_data()

    # Linear regression
    # run_simple_linear_regression(*housing)

    # Ridge regression
    # run_simple_ridge_regression(*housing, "cholesky")
    # run_simple_ridge_regression(*housing, "saga")
    # run_simple_ridge_regression(*housing, "lsqr")

    # Ridge regression with data scaling
    # run_ridge_regression_with_scaling(*housing, "cholesky")
    # run_ridge_regression_with_scaling(*housing, "saga")
    # run_ridge_regression_with_scaling(*housing, "lsqr")

    # Ridge regression with data scaling and alpha tuning
    run_ridge_regression_with_tuning(*housing)

if __name__ == "__main__":
    run_analysis()
