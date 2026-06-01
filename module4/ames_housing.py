import data_handler as dh
import numpy as np
import pandas as pd
from config import DataPath, TargetColumn
from model import RidgeRegressionModel, RidgeRegressionModelCV
from sklearn.compose import make_column_selector as selector
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import (OneHotEncoder, SplineTransformer,
                                   StandardScaler)
from types_config import Tpreprocessor
from visualisation import plot_coefficients_of_linear_model

numerical_features = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]


########
# DATA #
########
def load_numerical_data():
    """Load ames housing dataset with numerical features only."""
    data = dh.load_data_from_file(DataPath.AMES_HOUSING.value)
    return dh.get_subset(data, columns=numerical_features), \
        pd.Series(data[TargetColumn.AMES_HOUSING])

def load_data():
    """Load ames housing dataset."""
    return dh.load_data_from_file(DataPath.AMES_HOUSING.value, TargetColumn.AMES_HOUSING)


#####################
# RIDGE REGRESSSION #
#####################
def ridge_regression(x_data: pd.DataFrame, y_data: pd.Series, alpha: float):
    # Ridge regression without regularization
    regression: RidgeRegressionModel = RidgeRegressionModel.build_pipeline([StandardScaler()],
                                                                           alpha=alpha)

    # Cross-validation
    scores = regression.make_cross_validate(x_data, y_data, nb_fold=10, return_estimator=True)
    regression.print_cross_validate(scores)

    # Largest absolute value of coefficient
    coef = regression.get_coefficients(scores)
    regression.print_nlargest_coefficient(coef, 5)
    plot_coefficients_of_linear_model(coef)

def ridge_cv_regression(x_data: pd.DataFrame,
                        y_data: pd.Series,
                        transformers: list[Tpreprocessor]):
    # RidgeCV regression
    alphas = np.logspace(-3, 3, num=101)
    regression: RidgeRegressionModelCV = RidgeRegressionModelCV.build_pipeline(
        transformers=transformers,
        alphas=alphas,
        store_cv_results=True
    )

    # Cross-validation
    scores = regression.make_cross_validate(x_data,
                                            y_data,
                                            n_splits=50,
                                            test_size=0.2,
                                            return_estimator=True)
    regression.print_cross_validate(scores)

    # Alpha
    alpha_range = np.unique(np.array(regression.get_best_alpha_from_cv_estimator(scores)) > 99,
                            return_counts=True)
    print(f"Alpha range {alpha_range}")
    regression.print_best_alpha_from_cv_estimator(scores)


############
# ANALYSIS #
############
def run_analysis_with_numerical_data():
    """Perform ridge regression using only numerical data."""
    # Load data
    data, targets = load_numerical_data()

    # Ridge regression
    ridge_regression(data, targets, alpha=0)
    ridge_regression(data, targets, alpha=1)

    # Ridge regression with removing GarageArea features, which help to reduce the standard
    # deviation of GarageCars coefficient. Because, GarageArea and GarageCars are strongly
    # correlated and this kind of features typically cause unstable estimation of the matching
    # linear model coefficient. Correlation between two features could be checked by computing a
    # correlation coefficent such as the Pearson, Spearman or Kendall correlation.
    data = data.drop("GarageArea", axis=1)
    ridge_regression(data, targets, alpha=1)

    # RidgeCV regression
    ridge_cv_regression(data, targets, transformers=[StandardScaler()])

def run_analysis_with_all_data():
    """Use linear model with all data."""
    # Load data
    data, targets = load_data()

    # RidgeCV regression with StandardScaler. Using both numerical and categorical features lead to
    # a lower error than using only numerical and therefore a better model.
    ridge_cv_regression(data,
                        targets,
                        transformers=[
                            (StandardScaler(), selector(dtype_exclude=str)),
                            (OneHotEncoder(handle_unknown="ignore"), selector(dtype_include=str))
                        ])

    # RidgeCV with SplineTransformer and feature engineering. Using a non-linear pipeline lead to a
    # lower error than omitting feature engineering and therefore to a better model.
    ridge_cv_regression(data,
                        targets,
                        transformers=[
                            (SplineTransformer(), selector(dtype_exclude=str)),
                            (OneHotEncoder(handle_unknown="ignore"), selector(dtype_include=str)),
                            Nystroem(kernel="poly", degree=2, n_components=300)
                        ])

def run_analysis():
    # run_analysis_with_numerical_data()
    run_analysis_with_all_data()

if __name__ == "__main__":
    run_analysis()
