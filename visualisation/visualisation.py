import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import (LearningCurveDisplay,
                                     ValidationCurveDisplay)


def _show():
    """To show a graphic."""
    plt.tight_layout()
    plt.show()

####################
# DATA OBSERVATION #
####################
def check_data(data: pd.DataFrame, targets: pd.Series) -> None:
    """Simple workflow to look at the data."""
    display_shape, display_hist, display_count, display_relationship, display_pairplot = \
        True, True, True, True, True
    str_columns = data.select_dtypes([str]).columns

    # Display data shape
    if display_shape:
        print(f"The dataset contains {data.shape[0]} samples and {data.shape[1]} features with\n"
              "- numerical features: "
              f"{', '.join([col for col in data.columns if col not in str_columns])}\n"
              f"- categorical features: {', '.join(str_columns)}")

    # Histogram visualisation of numerical values
    if display_hist:
        _histogram(data)

    # Count categorical values - could be useful to detect class imbalance
    if display_count:
        for col in str_columns:
            print(data[col].value_counts())

    # Look at the relationship between two variables
    if display_relationship:
        _crosstab(data, "education", "education-num")

    if display_pairplot:
        _pairplot(data, targets, ["age", "education-num", "hours-per-week"])

def _histogram(data: pd.DataFrame) -> None:
    """Simple visualisation of numerical values as histrogram."""
    _ = data.hist(figsize=(20, 14))
    plt.title('Histogram for numerical values')
    _show()

def _crosstab(data: pd.DataFrame, x: str, y: str) -> None:
    """
    Crosstab virusalisation to look at the relationship between two variables.

    To detect redundant (or highly correlated) variables.
    """
    tab = pd.crosstab(index=data[x], columns=data[y])
    print(tab)

def _pairplot(data: pd.DataFrame, target: pd.Series, columns: list[str]) -> None:
    """
    Pairplot virusalisation to show how each variable differs according to the target.

    Can reveal interaction between variables.
    """
    # Only plot a subset of the data for performance issue and keep the plot readable
    n_samples = 5000

    _ = sns.pairplot(
        data=pd.concat([data[:n_samples], target[:n_samples]], sort=False, axis=1),
        vars=columns,
        hue=target.name, # target is a pandas series and Series.name return his name
        plot_kws={"alpha": 0.2},
        height=3,
        diag_kind="hist",
        diag_kws={"bins": 30},
    )
    _show()


################
# DATA SCALING #
################
def scaler_jointplot(x_train: pd.DataFrame,
                     x_train_scaled: pd.DataFrame,
                     x_axis: str,
                     y_axis: str) -> None:
    """
    Use jointplot to display histograms of the distribution and a scatterplot of any pair of
    numerical features. In order to observe than scaler does not change the structure of the data.
    Only the axes get shifted and scaled.
    """
    # Limit points to plot for clearer result
    limit = 300

    _jointplot(x_train[:limit],
               x_axis,
               y_axis, f"Jointplot of {x_axis} and {y_axis} before data scaling")
    _jointplot(x_train_scaled[:limit],
               x_axis,
               y_axis,
               f"Jointplot of {x_axis} and {y_axis} after data scaling")
    _show()

def _jointplot(data: pd.DataFrame, x_axis: str, y_axis: str, title: str) -> None:
    """Jointplot."""
    sns.jointplot(
        data=data,
        x=x_axis,
        y=y_axis,
        marginal_kws=dict(bins=15)
    )
    plt.suptitle(title)


##########################
# TRAINING/TESTING ERROR #
##########################
def error_distribution(scores: dict[str, npt.NDArray[np.float64]]):
    """Display the training and testing error distribution."""
    try:
        errors = pd.DataFrame(
            {"Train score": scores["train_score"], "Test score": scores["test_score"]}
        )
        errors.plot.hist(bins=50, edgecolor="black")
        plt.xlabel("Mean absolute error")
        plt.title("Training and testing error distribution via cross-validation")
        _show()
    except Exception as e:
        print(e)


####################
# VALIDATION CURVE #
####################
def show_validation_curve(curve: ValidationCurveDisplay, xlabel: str) -> None:
    """Show validation curve."""
    curve.ax_.set(
        xlabel=xlabel,
        title="Validation curve"
    )
    _show()


##################
# LEARNING CURVE #
##################
def show_learning_curve(curve: LearningCurveDisplay) -> None:
    """Show learning"""
    curve.ax_.set(
        xscale="log",
        xlabel="Number of samples in the training set",
        title="Learning curve"
    )
    _show()


############
# ERRORBAR #
############
def _errorbar(means: list[float],
              stds: list[float],
              x: list[float],
              color: str,
              label: str) -> None:
    """Plot y versus x as line and markers with attached errorbars."""
    plt.errorbar(x,
                 means,
                 yerr=stds,
                 fmt="-o",
                 capsize=5,
                 ecolor=color,
                 color=color,
                 elinewidth=1.5,
                 label=label)

def show_errorbars_for_hyperparameter_tuning(train_scores: dict[str, list[float]],
                                             test_scores: dict[str, list[float]],
                                             x: list[float],
                                             xlabel: str) -> None:
    """Display errorbar for manual tuning of hyperparameter."""
    _errorbar(train_scores["mean"], train_scores["std"], x, "blue", "Train")
    _errorbar(test_scores["mean"], test_scores["std"], x, "orange", "Test")
    plt.xlabel(xlabel)
    plt.ylabel("Scores")
    plt.legend()
    _show()


########################
# PARALLEL COORDINATES #
########################
def show_parallel_coordinates_for_hyperparameter_tuning(data: pd.DataFrame) -> None:
    """Build a parallel coordinates graphic."""
    # Ignore some columns
    columns_to_ignore = ["std_test_score", "rank_test_score", "mean_train_score",
                         "std_train_score"]
    data.drop(columns=columns_to_ignore, inplace=True)

    # The parallel coordinate plot only take numerical data as entry
    # therefore convert bool column into int
    bool_data = data.select_dtypes(bool)
    if not bool_data.empty:
        bool_data = bool_data.astype(int)
        data = data.drop(columns=bool_data.columns)
        data = pd.concat([data, bool_data], axis=1)

    # Plot
    fig = px.parallel_coordinates(data,
                                  color="mean_test_score",
                                  dimensions=data.columns,
                                  color_continuous_scale=px.colors.sequential.Viridis)
    fig.show()


###############
# COEFFICIENT #
###############
def plot_coefficients_of_logistic_regression(coef: dict[str, list[float]]):
    """
    Plot coefficients of logistic regression.

    Parameter
    ---------
    coef: dictionary with features as keys and coefficients as values
    """
    # Set up dataframe. Filtered out to keep only the most important features (top 15)
    data = pd.DataFrame.from_dict(coef)
    if len(coef) > 15:
        top_features = data.median().sort_values(ascending=False).head(15).index
        data = data[top_features]

    # Plot
    _, ax = plt.subplots()
    _ = data.abs().plot.box(color={"whiskers": "black", "medians": "black", "caps": "black"},
                            vert=False,
                            ax=ax)
    _show()


####################
# CROSS-VALIDATION #
####################
def plot_cross_validation_scores(test_scores: list[npt.NDArray[np.float64]], labels: list[str]):
    """Plot cross-validation scores."""
    indices = np.arange(len(test_scores[0]))
    for i in range(len(test_scores)):
        plt.scatter(indices, test_scores[i], label=labels[i])
    plt.ylim((0, 1))
    plt.xlabel("Cross-validation iteration")
    plt.ylabel("Accuracy")
    _ = plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    _show()
