import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.typing as npt
import seaborn as sns

from sklearn.model_selection import (
    LearningCurveDisplay,
    ValidationCurveDisplay
)


####################
# DATA OBSERVATION #
####################
"""Simple workflow to look at the data."""
def check_data(data: pd.DataFrame, targets: pd.Series) -> None:
    display_shape, display_hist, display_count, display_relationship, display_pairplot = \
        True, True, True, True, True
    str_columns = data.select_dtypes([str]).columns

    # Display data shape
    if display_shape:
        print(f"The dataset contains {data.shape[0]} samples and {data.shape[1]} features with\n"
              f"- numerical features: {', '.join([col for col in data.columns if col not in str_columns])}\n"
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

"""Simple visualisation of numerical values as histrogram."""
def _histogram(data: pd.DataFrame) -> None:
    _ = data.hist(figsize=(20, 14))
    plt.title('Histogram for numerical values')
    plt.show()

"""
Crosstab virusalisation to look at the relationship between two variables.

To detect redundant (or highly correlated) variables.
"""
def _crosstab(data: pd.DataFrame, x: str, y: str) -> None:
    tab = pd.crosstab(index=data[x], columns=data[y])
    print(tab)

"""
Pairplot virusalisation to show how each variable differs according to the target.

Can reveal interaction between variables.
"""
def _pairplot(data: pd.DataFrame, target: pd.Series, columns: list[str]) -> None:
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
    plt.show()


################
# DATA SCALING #
################
"""
Use jointplot to display histograms of the distribution and a
scatterplot of any pair of numerical features. In order to observe
than scaler does not change the structure of the data. 
Only the axes get shifted and scaled.
"""
def scaler_jointplot(x_train: pd.DataFrame, x_train_scaled: pd.DataFrame, x_axis: str, y_axis: str) -> None:
    # Limit points to plot for clearer result
    limit = 300

    _jointplot(x_train[:limit], x_axis, y_axis, f"Jointplot of {x_axis} and {y_axis} before data scaling")
    _jointplot(x_train_scaled[:limit], x_axis, y_axis, f"Jointplot of {x_axis} and {y_axis} after data scaling")
    plt.show()

"""Jointplot."""
def _jointplot(data: pd.DataFrame, x_axis: str, y_axis: str, title: str) -> None:
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
"""Display the training and testing error distribution."""
def error_distribution(scores: dict[str, npt.NDArray[np.float64]]):
    try:
        errors = pd.DataFrame({"Train score": scores["train_score"], "Test score": scores["test_score"]})
        errors.plot.hist(bins=50, edgecolor="black")
        plt.xlabel("Mean absolute error")
        plt.title("Training and testing error distribution via cross-validation")
        plt.show()
    except Exception as e:
        print(e)


####################
# VALIDATION CURVE #
####################
"""Show validation curve."""
def show_validation_curve(curve: ValidationCurveDisplay, xlabel: str) -> None:
    curve.ax_.set(
        xlabel=xlabel,
        title="Validation curve"
    )
    plt.show()


##################
# LEARNING CURVE #
##################
"""Show learning"""
def show_learning_curve(curve: LearningCurveDisplay) -> None:
    curve.ax_.set(
        xscale="log",
        xlabel="Number of samples in the training set",
        title="Learning curve"
    )
    plt.show()


############
# ERRORBAR #
############
"""Plot y versus x as line and markers with attached errorbars."""
def _errorbar(means: list[float],
              stds: list[float],
              x: list[float],
              color: str,
              label: str) -> None:
    plt.errorbar(x,
                 means,
                 yerr=stds,
                 fmt="-o",
                 capsize=5,
                 ecolor=color,
                 color=color,
                 elinewidth=1.5,
                 label=label)

"""Display errorbar for manual tuning of hyperparameter."""
def show_errorbars_for_hyperparameter_tuning(train_scores: dict[str, list[float]],
                                             test_scores: dict[str, list[float]],
                                             x: list[float],
                                             xlabel: str):
    _errorbar(train_scores["mean"], train_scores["std"], x, "blue", "Train")
    _errorbar(test_scores["mean"], test_scores["std"], x, "orange", "Test")
    plt.xlabel(xlabel)
    plt.ylabel("Scores")
    plt.legend()
    plt.show()
