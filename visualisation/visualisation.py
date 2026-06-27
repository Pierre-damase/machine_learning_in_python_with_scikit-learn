import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             PredictionErrorDisplay, RocCurveDisplay,
                             precision_score, recall_score)
from sklearn.model_selection import (LearningCurveDisplay,
                                     ValidationCurveDisplay)
from types_config import CvResults, DataSetType


def _show():
    """To show a graphic."""
    plt.tight_layout()
    plt.show()

####################
# DATA OBSERVATION #
####################
def check_data(x_data: pd.DataFrame, y_data: pd.Series) -> None:
    """Simple workflow to look at the data."""
    display_shape, display_hist, display_count, display_relationship, display_pairplot = \
        True, True, True, True, True
    str_columns = x_data.select_dtypes([str]).columns

    # Display data shape
    if display_shape:
        print(f"The dataset contains {x_data.shape[0]} samples and {x_data.shape[1]} features "
              "with\n- numerical features: "
              f"{', '.join([col for col in x_data.columns if col not in str_columns])}\n"
              f"- categorical features: {', '.join(str_columns)}")

    # Histogram visualisation of numerical values
    if display_hist:
        _histogram(x_data)

    # Count categorical values - could be useful to detect class imbalance
    if display_count:
        for col in str_columns:
            print(x_data[col].value_counts())

    # Look at the relationship between two variables
    if display_relationship:
        _crosstab(x_data, "education", "education-num")

    if display_pairplot:
        _pairplot(x_data, y_data, ["age", "education-num", "hours-per-week"])

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

def _pairplot(x_data: pd.DataFrame, y_data: pd.Series, columns: list[str]) -> None:
    """
    Pairplot virusalisation to show how each variable differs according to the target.

    Can reveal interaction between variables.
    """
    # Only plot a subset of the data for performance issue and keep the plot readable
    n_samples = 5000

    _ = sns.pairplot(
        data=pd.concat([x_data[:n_samples], y_data[:n_samples]], sort=False, axis=1),
        vars=columns,
        hue=y_data.name, # target is a pandas series and Series.name return his name
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
def error_distribution(scores: CvResults):
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
def show_validation_curve(curve: ValidationCurveDisplay,
                          xlabel: str,
                          point_of_interests: list[float|int] | None = None) -> None:
    """Show validation curve."""
    # Ax
    curve.ax_.set(xlabel=xlabel, title="Validation curve")

    # Figure
    curve.figure_.set_size_inches((20, 12))

    # Add vline
    if point_of_interests is not None:
        for x in point_of_interests:
            plt.axvline(x=x, color="red", linestyle="--", alpha=0.5, label=x)
        plt.legend()

    # Display
    _show()


##################
# LEARNING CURVE #
##################
def show_learning_curve(curve: LearningCurveDisplay) -> None:
    """Show learning curve."""
    curve.ax_.set(
        xscale="log",
        xlabel="Number of samples in the training set",
        title="Learning curve"
    )
    _show()


####################
# CONFUSION MATRIX #
####################
def show_confusion_matrix(curve: ConfusionMatrixDisplay,
                          ax: Axes | None = None) -> None:
    """
    Show confusion matrix in order to display:
    - True positive (TP): the probability of a positive test result, conditioned on the individual
    truly being positive. e.g. blood donors correctly identified as such

    - True negative (TN): the probability of a negative test result, conditioned on the individual
    truly being negative: e.g. not blood donors correctly identified as such

    - False negative (FN): e.g. blood donors incorrectly identified as not

    - False positive (FP): e.g. not blood donors incorrectly identified as one
    """
    curve.ax_.set_title("Confusion matrix")
    if ax is None:
        _show()


############################
# PRECISION RECALL DISPLAY #
############################
def show_pr_curve(curve: PrecisionRecallDisplay,
                  y_test: pd.Series,
                  y_predicted: npt.NDArray[np.str_],
                  pos_label: str,
                  ax: Axes | None = None) -> None:
    """
    Show precision recall curve. A perfect classifier would have a precision of 1 for all recall
    values and an average precision (AP) of 1.
    """
    # Precision refers to the test's ablity to accurately detect genuine blood donors (TP), taking
    # into account everyone who has been considered a donor (TP + FP). Precision = TP / (TP + FP)
    # In other words, precision measure the ability of the model to not make mistake among the
    # samples actually classified as positive.
    precision = precision_score(y_test, y_predicted, pos_label=pos_label)
    print(f"\nPrecision score is {precision:.3f}")

    # Recall (sensitivity) refers to the test's ability to correctly detect blood donors (TP) out
    # of those who did give blood (TP + FN). Recall = TP / (TP + FN)
    # In other words, recall is the ability of the model to find all the samples that should have
    # been classified as positive.
    recall = recall_score(y_test, y_predicted, pos_label=pos_label)
    print(f"Recall score is {recall:.3f}")

    # Label
    curve.ax_.set_xlabel("Recall (sensitivity)")
    curve.ax_.set_ylabel("Precision (PPV)")

    # Limit
    curve.ax_.set_xlim(0, 1)
    curve.ax_.set_ylim(0, 1)

    curve.ax_.legend()
    curve.ax_.set_title("Precision-recall curve")
    if ax is None:
        _show()


#####################
# ROC CURVE DISPLAY #
#####################
def show_roc_curve(curve: RocCurveDisplay) -> None:
    """
    Show Receiver Operating Characteristic (ROC) curve. It's a common plot for sensitivity (recall)
    and specificity. Specificity measures the proportion of correctly classified samples in the
    negative class. Specificity = TN / (TN + FP).
    """
    # Label
    curve.ax_.set_xlabel("False positive rate")
    curve.ax_.set_ylabel("True positive rate (sensitivity/recall)")

    # Limit
    curve.ax_.set_xlim(0, 1)
    curve.ax_.set_ylim(0, 1)

    curve.ax_.legend()
    curve.ax_.set_title("Receiver Operating Characteristics curve")
    _show()


############################
# PREDICTION ERROR DISPLAY #
############################
def show_prediction_error_curve(curve: PredictionErrorDisplay) -> None:
    """
    Plot the residuals, which represent the difference between the actual and predicted values,
    against he predicted values.
    """
    pass


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

def show_errorbars_for_hyperparameter_tuning(test_scores: dict[str, list[float]],
                                             x: list[float],
                                             xlabel: str,
                                             xscale: str = "linear",
                                             yscale: str = "linear",
                                             train_scores: dict[str, list[float]] | None = None) \
                                             -> None:
    """Display errorbar for manual tuning of hyperparameter."""
    _errorbar(test_scores["mean"], test_scores["std"], x, "orange", "Test")
    if train_scores:
        _errorbar(train_scores["mean"], train_scores["std"], x, "blue", "Train")
    plt.xlabel(xlabel)
    plt.xscale(xscale)
    plt.ylabel("Scores")
    plt.yscale(yscale)
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
def plot_coefficients_of_linear_model(coef: dict[str, list[float]]):
    """
    Plot coefficients of linear model.

    Parameters
    ----------
    coef: dictionary with features as keys and coefficients as values
    """
    # Set up dataframe. Filtered out to keep only the most important features (top 15)
    data = pd.DataFrame.from_dict(coef)
    if len(coef) > 20:
        top_features = data.median().sort_values(ascending=False).head(15).index
        data = data[top_features]

    # Plot
    titles = ["Linear model coefficients", "Linear model coefficients with log scaling"]
    _, axs = plt.subplots(ncols=len(titles), figsize=(14, 10), constrained_layout=True)
    for i in range(len(axs)):
        data.abs().plot.box(color={"whiskers": "black", "medians": "black", "caps": "black"},
                            vert=False,
                            ax=axs[i])
        axs[i].set(title=titles[i])
    axs[1].set(xscale="symlog")
    _show()


####################
# CROSS-VALIDATION #
####################
def plot_cross_validation_scores(scores: list[CvResults],
                                 names: list[str],
                                 score_name: str | None = None,
                                 is_regression: bool = False) -> None:
    """
    Plot cross-validation scores.

    Parameters
    ----------
    scores: list of cv results

    names: label of each cv perform

    score_names: for regression problem it could be useful to plot the scoring function name

    is_regression: the plot strategy differ between  calssification and regression model
    """
    # Set bins
    if is_regression:
        bins = np.linspace(start=min([min(ele["test_score"]) for ele in  scores])-20,
                           stop=max([max(ele["test_score"]) for ele in  scores])+20,
                           num=100)
    else:
        bins = np.linspace(start=0, stop=1.0, num=100)

    # Plot testing errors
    data = pd.concat(
        [pd.Series(scores[i]["test_score"], name=names[i]) for i in range(len(scores))],
        axis=1
    )
    data.plot.hist(bins=bins, edgecolor="black")

    # For classification it's required to limit the graph between 0 and 1.
    if not is_regression:
        plt.xlim((0, 1))

    plt.xlabel("Accuracy (%)" if score_name is None else score_name)
    plt.ylabel("Frequency")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    _show()


#############
# BOOTSTRAP #
#############
def plot_bootstrap_samples(data: DataSetType,
                           bootstrap_samples: list[DataSetType],
                           feature_name: str) -> None:
    """
    Plot resampled data with bootstrap strategy. Selected data for the boostrap samples appear
    with a darker blue circle. Because the boostrap strategy use a selection with replacement,
    the same data point can be selected several times.

    Parameter
    ---------
    data: original data (features, targets)

    bootstrap_samples: list of (features, target)

    feature_name: display x as a function of y with x the given feature and y the target
    """
    nb_samples = len(bootstrap_samples)

    # Plot
    fig, axs = plt.subplots(ncols=nb_samples, figsize=(14, 10), constrained_layout=True)
    for i in range(nb_samples):
        axs[i].scatter(x=bootstrap_samples[i]["x_data"][feature_name],
                       y=bootstrap_samples[i]["y_data"],
                       color="tab:blue",
                       facecolors="none",
                       alpha=0.5,
                       label="Resampled data",
                       s=180,
                       linewidths=5)
        axs[i].scatter(data["x_data"][feature_name],
                       data["y_data"],
                       color="black",
                       s=60,
                       alpha=1,
                       label="Original data")
        axs[i].set_title(f"Resampled data #{i}")
        axs[i].legend()
    fig.suptitle("Boostrap data resampling")
    _show()


###############
# TIME SERIES #
###############
def plot_time_series_detection(y_trains: list[pd.Series],
                               y_tests: list[pd.Series],
                               y_predicted: list[pd.Series]):
    """
    Plot the result of time series detection.

    Parameters
    ----------
    y_trains: list of targets use for training

    y_test: list of targets use for testing

    y_predicted: list of predicted targets
    """
    # Set subplots
    _, axs = plt.subplots(ncols=len(y_trains), figsize=(14, 10), constrained_layout=True)

    # Plot targets
    titles = ["Predictions using shuffled data", "Prediction using unshuffled data"]
    for i in range(len(y_trains)):
        y_trains[i].plot(ax=axs[i], label="Training", color="b")
        y_tests[i].plot(ax=axs[i], label="Testing", color="g")
        y_predicted[i].plot(ax=axs[i], label="Predicted", color="r")
        axs[i].set_title(titles[i])

    plt.legend()
    _show()

