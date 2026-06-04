from typing import TypeVar

import data_handler as dh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DataPath, TargetColumn
from model import (LinearRegressionModel, LogisticRegressionModel,
                   RidgeRegressionModel, RidgeRegressionModelCV)
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import (MinMaxScaler, PolynomialFeatures,
                                   StandardScaler)
from types_config import CvResults, DataSetType, Tpreprocessor
from visualisation import (plot_coefficients_of_linear_model,
                           show_errorbars_for_hyperparameter_tuning)

HOUSING_FEATURES = ["LotFrontage", "LotArea", "PoolArea", "YearBuilt", "YrSold"]

PENGUIN_FEATURES = ["Culmen Length (mm)", "Culmen Depth (mm)"]

type ClassModelTypes = (LinearRegressionModel | LogisticRegressionModel)
Tclassmodel = TypeVar('Tclassmodel', bound=ClassModelTypes)


########
# DATA #
########
def load_ames_housing() -> DataSetType:
    """Load ames housing dataset."""
    data, targets = dh.load_data_from_file(DataPath.AMES_HOUSING.value, TargetColumn.AMES_HOUSING)
    return dh.get_subset(data, HOUSING_FEATURES), targets

def load_penguin() -> DataSetType:
    """Load penguin dataset."""
    data = dh.get_subset(dh.load_data_from_file(DataPath.PENGUIN.value),
                         columns=PENGUIN_FEATURES + [TargetColumn.PENGUIN]).dropna()
    return dh.get_subset(data, PENGUIN_FEATURES), pd.Series(data[TargetColumn.PENGUIN])

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

#################
# VISUALISATION #
#################
def barplot(weights: list[pd.Series], params: list[float]) -> None:
    """
    To display the impact of the regularization on the coefficients. As expected, a small C (strong
    regularization) shrinks the coefficients toward zero.
    """
    pd.concat(weights, axis=1, keys=params).plot.barh()
    plt.title("Logistic regression weights depending of C")
    plt.tight_layout()
    plt.show()


#####################
# LINEAR REGRESSION #
#####################
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

    The training and testing error get closer therefore the model is less overfitting but still.
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

def run_regularization_for_regression_task():
    """
    Regularization for regression task using Ridge regression. Use the alpha parameter to control
    the strength of the regularization:

    - A low alpha mean weak regularization
    - A high alpha mean strong regularization
    """
    # Load data
    housing = load_ames_housing()

    # Linear regression
    run_simple_linear_regression(*housing)

    # Ridge regression
    run_simple_ridge_regression(*housing, "cholesky")
    run_simple_ridge_regression(*housing, "saga")
    run_simple_ridge_regression(*housing, "lsqr")

    # Ridge regression with data scaling
    run_ridge_regression_with_scaling(*housing, "cholesky")
    run_ridge_regression_with_scaling(*housing, "saga")
    run_ridge_regression_with_scaling(*housing, "lsqr")

    # Ridge regression with data scaling and alpha tuning
    run_ridge_regression_with_tuning(*housing)


#######################
# LOGISTIC REGRESSION #
#######################
def run_regularization_for_classification_task(transformers: list[Tpreprocessor], **kwargs):
    """
    Regularization for classification task by tuning the C parameter:

    - A low C parameter mean strong regularization: in this case, the classifier is less confident
    in its prediction. We are enforcing a spread sigmoid.

    - A high C parameter mean weak regularization. By default, C is 1.0, which means almost no
    regularization: in this case, the classifier is more confident. We are enforcing a steep
    sigmoid.

    Similar observation with feature engineering. Except that the decision boundary is no longer a
    straight line because the linear model is classifying in the 100-dimensional (n_components=100)
    feature space created by the Nystroem transformer. As a result, the decision boundary induced
    by the overall pipeline is now expressive enough to wrap around the minority class.
    """
    # Load data
    penguin = load_penguin()

    # Split data
    x_train, x_test, y_train, y_test = dh.sklearn_train_test_split(*penguin, test_size=0.4)

    # C value to try out
    params = [1e-6, 0.01, 0.1, 1, 10, 100, 1e6]

    # Set up a logistic regression pipeline and try out various C parameters
    _, axs = plt.subplots(ncols=len(params), constrained_layout=True)
    weights = []
    for i in range(len(params)):
        # Build pipeline and train the model
        regression: LogisticRegressionModel = LogisticRegressionModel.build_pipeline(transformers,
                                                                                     C=params[i],
                                                                                     **kwargs)
        regression.start(x_train, x_test, y_train, y_test)

        # Get the associated weights of each feature. Skip with feature engineering.
        coefs = regression.get_weights()
        if len(coefs) == 2:
            weights.append(pd.Series(coefs, index=PENGUIN_FEATURES))

        # Boundary decision
        regression.decision_boundary_display(pd.concat([x_train, y_train], axis=1),
                                             x_train,
                                             response_method="predict_proba",
                                             multiclass_colors=["blue", "red", "green"],
                                             palette=["tab:blue", "tab:red", "tab:green"],
                                             hue=TargetColumn.PENGUIN.value,
                                             plot_method="pcolormesh",
                                             draw_contour_lines=True,
                                             ax=axs[i],
                                             alpha=0.8,
                                             vmin=0.0,
                                             vmax=1.0)
        axs[i].set(title=f"Decision boundary with C={params[i]}",
                   xlabel=x_train.columns[0],
                   ylabel=x_train.columns[1] if i == 0 else None)
    plt.show()

    # Impact of the regularization on the coefficient, i.e. the weight associated to each feature.
    # Skip with feature engineering.
    if weights:
        barplot(weights, params)


############
# ANALYSIS #
############
def run_analysis():
    """
    Regularization for regression and classification task in order to avoid overfitting for linear
    models.
    """
    # Regularization for regression task using ridge regression
    #run_regularization_for_regression_task()

    # Regularization for classification task by using various C parameter
    #run_regularization_for_classification_task(transformers=[StandardScaler()])

    # Use Nystroem to experiment impact of the regularization with non-linear feature engineering
    run_regularization_for_classification_task(transformers=[
        StandardScaler(),
        Nystroem(kernel="rbf", gamma=1.0, n_components=100, random_state=0)
    ],
                                               max_iter=1000)

if __name__ == "__main__":
    run_analysis()
