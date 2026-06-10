import data_handler as dh
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from config import SYNTHETIC_DATASET_FEATURE, DataPath, TargetColumn
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from model import (AdaBoostClassifierModel, DecisionTreeClassifierModel,
                   DecisionTreeRegressorModel, GradientBoostingRegressorModel,
                   RandomForestRegressorModel)
from sklearn.tree import DecisionTreeClassifier
from types_config import DataSetType
from visualisation.visualisation import show_validation_curve

type ClassModelTypes = (AdaBoostClassifierModel
                        | DecisionTreeClassifierModel
                        | DecisionTreeRegressorModel
                        | GradientBoostingRegressorModel
                        | RandomForestRegressorModel)

SYNTHETIC_DATASET_FEATURE = "Feature"
SYNTHETIC_DATASET_TARGET = "Target"

PENGUIN_FEATURES = ["Culmen Length (mm)", "Culmen Depth (mm)"]

########
# DATA #
########
def load_penguins() -> DataSetType:
    """Load penguins dataset."""
    return dh.load_data_from_file(DataPath.PENGUIN.value,
                                  target=TargetColumn.PENGUIN,
                                  columns=PENGUIN_FEATURES,
                                  drop_na=True)

def load_californian_housing() -> DataSetType:
    """Load californian housing dataset."""
    return dh.load_california_dataset()


#################
# VISUALISATION #
#################
def plot_synthetic_data(x_train: pd.DataFrame,
                        x_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_train_predicted: pd.Series,
                        y_test_predicted: pd.Series,
                        ax: Axes,
                        title: str) -> None:
    """
    Plot synthetic dataset.

    Parameters
    ----------
    x_train: training data

    x_test: testing data, i.e. data not use to train the model

    y_train: original training target for the first decision tree else residuals of the previous
    model

    y_train_predicted: prediction of the target for the training data

    y_test_predicted: prediction of the target for the testing data
    """
    # Plot the data
    sns.scatterplot(x=x_train[SYNTHETIC_DATASET_FEATURE],
                    y=y_train,
                    color="black",
                    alpha=0.5,
                    ax=ax)

    # Predict the testing data and plot the prediction
    ax.plot(x_test[SYNTHETIC_DATASET_FEATURE],
            y_test_predicted,
            "-",
            label="Decision function")

    # Predict the training data and plot the residuals
    residual_lines: list[Line2D] = []
    for i in range(len(x_train)):
        x_train_value = x_train[SYNTHETIC_DATASET_FEATURE][i]
        residual_lines = ax.plot([x_train_value, x_train_value],
                                 [y_train[i], y_train_predicted[i]],
                                 color="red")
    residual_lines[0].set_label("Residuals")

    ax.legend()
    ax.set_title(title)

def validation_curve(model: ClassModelTypes,
                     x_data: pd.DataFrame,
                     y_data: pd.Series,
                     xlabel: str):
    """Validation curve to try out various n_estimators value."""
    # Parameters to try out
    param_range = np.array([1, 2, 5, 10, 20, 50, 100, 200])

    # Compute and plot the validation curve
    curve = model.compute_validation_curve(x_data,
                                           y_data,
                                           param_name="n_estimators",
                                           param_range=param_range,
                                           scoring="neg_mean_absolute_error",
                                           score_name="Mean absolute error",
                                           negate_score=True)
    show_validation_curve(curve, xlabel)


#########
# MODEL #
#########
def plot_decision_boundary(model: ClassModelTypes,
                           x_data: pd.DataFrame,
                           y_data: pd.Series,
                           misclassified: pd.DataFrame | None = None,
                           previously_misclassified: pd.DataFrame | None = None):
    """
    Plot the decision boundary of a fitted model.

    Parameters
    ----------
    model: machine learning model

    x_data: input data (features)

    y_data: targets

    misclassified: misclassified samples
    """
    model.decision_boundary_display(pd.concat([x_data, y_data], axis=1),
                                    x_data,
                                    response_method="predict",
                                    cmap="tab10",
                                    norm=getattr(mpl, "colors").Normalize(vmin=-0.5, vmax=8.5),
                                    hue=TargetColumn.PENGUIN.value,
                                    multiclass_colors=["blue", "orange", "green"],
                                    palette=["tab:blue", "tab:orange", "tab:green"],
                                    title="Decision tree predictions",
                                    misclassified=misclassified,
                                    previously_misclassified=previously_misclassified,
                                    alpha=0.5)

def plot_decision_boundary_for_estimators(model: AdaBoostClassifierModel,
                                          x_data: pd.DataFrame,
                                          y_data: pd.Series) -> None:
    """Plot the decision boundary of each fitted sub-model."""
    model.decision_boundary_display_for_estimators(pd.concat([x_data, y_data], axis=1),
                                                   x_data,
                                                   response_method="predict",
                                                   hue=TargetColumn.PENGUIN.value,
                                                   cmap="tab10",
                                                   norm=getattr(mpl, "colors").Normalize(vmin=-0.5,
                                                                                         vmax=8.5),
                                                   alpha=0.5)

def build_model(model_class: type[ClassModelTypes],
                x_data: pd.DataFrame,
                y_data: pd.Series,
                scoring: str | None = None,
                skip_cv: bool = False,
                **kwargs) -> ClassModelTypes:
    """
    Build a model.

    Parameters
    ----------
    model_class: which model to initialise

    x_data: input data

    y_data: targets

    **kwargs: additional parameters of the model
    """
    # Build model
    model: ClassModelTypes = model_class.build(**kwargs)

    # Cross-validation
    if not skip_cv:
        scores = model.make_cross_validate(x_data, y_data, nb_fold=10, scoring=scoring)
        model.print_cross_validate(scores)

    return model

def train_model(model: ClassModelTypes,
                x_data: pd.DataFrame,
                y_data: pd.Series,
                sample_weight: npt.NDArray[np.int8] | list[float] | None = None) -> None:
    """Train a model."""
    model.train(x_data, y_data, sample_weight=sample_weight)
    model.print_testing_accuracy(x_data, y_data)


#################
# DECISION TREE #
#################
def decision_tree(penguins: DataSetType) -> None:
    """Run decision tree model."""
    # 1\ Shallow decision tree classifier. The testing error is quite good (~0.93) but some samples
    # have been misclassified by the model.
    classifier = build_model(DecisionTreeClassifierModel, **penguins, max_depth=2, random_state=0)

    # Train the model and get misclassified samples
    train_model(classifier, **penguins)
    misclassified = classifier.get_misclassified(**penguins)
    plot_decision_boundary(classifier, **penguins, misclassified=misclassified)

    # 2\ Train the model by setting the sample_weight parameter at training to give more importance
    # to previously misclassified samples. Which will drastically change the decision function.
    # And add error to previous well classified samples. The testing is worse (~ 0.7).
    train_model(classifier,
                **penguins,
                sample_weight=classifier.get_sample_weight(penguins["y_data"], misclassified))
    plot_decision_boundary(classifier, **penguins, previously_misclassified=misclassified)
    new_misclassified = classifier.get_misclassified(**penguins)

    # 3\ The previous sample weighting is not optimal and add new mistake. Moreover, the testing
    # error is slightly worse. Therefore, we should trust the 1st classifier slightly more than the
    # 2nd. This is the idea behing boosting, a reweighting of the original dataset to help each
    # model (here we just retrain the same model for the example) to focus on specific samples.
    ensemble_weight = classifier.get_ensemble_weight(penguins["y_data"],
                                                     [misclassified, new_misclassified])
    print(ensemble_weight)


#############
# ADA BOOST #
#############
def ada_boost(penguins: DataSetType) -> None:
    """Run AdaBoost model."""
    # AdaBoost with decision tree as estimator
    estimator = DecisionTreeClassifier(max_depth=3, random_state=0)
    classifier: AdaBoostClassifierModel = build_model(AdaBoostClassifierModel,
                                                      **penguins,
                                                      estimator=estimator,
                                                      n_estimators=3,
                                                      random_state=0)

    # Fit the model
    train_model(classifier, **penguins)

    # Fitted sub-estimators information
    classifier.print_estimator_weights()
    classifier.print_estimator_error()
    plot_decision_boundary_for_estimators(classifier, **penguins)

    # Final decision boundary
    plot_decision_boundary(classifier, **penguins)


#####################
# GRADIENT BOOSTING #
#####################
def manual_gradient_boosting():
    """
    Run manual gradient-boosting on generated data.

    The idea is to create a first shallow decision tree that will underfit and is not expressiv
    enough to handle the complexity of the data. Then create a second decision tree that use the
    same input data as training but fits the residuals and tries to predict it instead of the
    original target. The residuals are the prediction errors made by the first tree on the training
    data, instead of the target.
    """
    # Load data
    x_train, x_test, y_train = dh.make_synthetic_dataset(x_min=-1.4,
                                                         x_max=1.4,
                                                         seed=0,
                                                         noise_shift=0.3,
                                                         y_train_shift=0)

    # Set up the max depth of each decision tree. The first one in particular is really shallow on
    # purpose to create an underfit model. Therefore, the resting error as well as the training
    # error are quite low.
    max_depths = [3, 5]

    # Set up the plot
    _, axs = plt.subplots(ncols=len(max_depths), figsize=(20, 12))

    # Manual gradient-boosting procedure
    trees: list[DecisionTreeRegressorModel] = []
    y_residuals: pd.Series = pd.Series()
    for i in range(len(max_depths)):
        # Build a decision tree
        trees.append(build_model(DecisionTreeRegressorModel,
                                 x_train,
                                 y_train,
                                 max_depth=max_depths[i],
                                 random_state=0))

        # Fit the model. The first one use the original target, the following the residuals of the
        # previous model.
        trees[i].train(x_train, y_train if y_residuals.empty else y_residuals)

        # Predictions
        y_train_predicted = trees[i].model.predict(x_train)
        y_test_predicted = trees[i].model.predict(x_test)

        # Get the residuals, i.e. the prediction errors made on the training data.
        y_residuals = y_train - y_train_predicted

        # Plot
        plot_synthetic_data(x_train,
                            x_test,
                            y_train if y_residuals.empty else y_residuals,
                            y_train_predicted,
                            y_test_predicted,
                            axs[i],
                            title=f"Decision tree #{i}")

    plt.title("Synthetic regression dataset")
    plt.show()

    # Analysis of a single sample of interest for which only two trees were enough to make the
    # perfect prediction. However, in practice hundreds of trees may be required to successfully
    # correct the error.
    sample_analysis(trees, x_train, y_train)

def sample_analysis(trees: list[DecisionTreeRegressorModel],
                    x_train: pd.DataFrame,
                    y_train: pd.Series) -> None:
    """Analysis of a sample of interest."""
    # Get sample of interest as a DataFrame (for fitting part)
    sample = x_train.iloc[[-7]]

    # Get x and y value
    x_sample, y_sample = sample[SYNTHETIC_DATASET_FEATURE].iloc[0], y_train.iloc[-7]

    # The error of the first tree as expected is high and the error of next one is really small.
    y_predicted: list[float] = []
    print(f"\nTrue value to predict for f(x={x_sample:.3f}) = {y_sample:.3f}")
    for i in range(len(trees)):
        y_predicted.append(trees[i].model.predict(sample)[0])
        print(f"Prediction of tree #{i} for x={x_sample:.3f} is y={y_predicted[i]:.3f}")
        print(f"The error is {y_sample - y_predicted[i]:.3f}")

    # The prediction of x can be predict as the sum of the prediction of all the trees in the
    # ensemble
    y_ensemble_predicted = sum(y_predicted)
    print(f"\nThe ensemble prediction of x={x_sample:.3f} is y={y_ensemble_predicted:.3f}")
    print(f"The ensemble error is {y_sample - y_ensemble_predicted:.3f}")

def gradient_boosting():
    """
    Run gradient-boosting procedure and perform a comparison with a random-forest.

    Both algorithms lead to very good and close results. However the graient boosting is overall
    faster than the random forest.
    """
    housing = load_californian_housing()

    scoring = "neg_mean_absolute_error"

    # Gradient-boosting (testing error of 44±9k$) and random-forest (testing error of 45±10k$)
    build_model(GradientBoostingRegressorModel, **housing, scoring=scoring, n_estimators=200)
    build_model(RandomForestRegressorModel, **housing, scoring=scoring, n_estimators=200, n_jobs=2)

    # Split data
    split_data = dh.sklearn_train_test_split(**housing, test_size=0.5)
    x_train, y_train = split_data["x_train"], split_data["y_train"]

    # Gradient-boosting. When the number of trees is too large, gradient-boosting model tends to
    # overfit. And at some point their is no improvement anymore. To avoid adding a new unnecessary
    # tree, GradientBoosting offers an early-stopping option. Basicaly, if there is no improvement
    # of the generalization performance after some addition of trees, the algorithm stop.
    model = build_model(GradientBoostingRegressorModel,
                        x_train,
                        y_train,
                        scoring=scoring,
                        skip_cv=True,
                        max_depth=5,
                        learning_rate=0.5)
    validation_curve(model, x_train, y_train, "Number of boosting stages")

    # Random-forest. The model improve when increasing the number of tress in the ensemble.
    # However, the scores reach a plateau where adding new tress just make fitting and scoring
    # slower.
    model = build_model(RandomForestRegressorModel,
                        x_train,
                        y_train,
                        scoring=scoring,
                        skip_cv=True,
                        max_depth=None)
    validation_curve(model, x_train, y_train, "Number of trees in the forest")

    # Gradient-boosting using the early stopping. The average testing error is quite good (44±9k$).
    tree: GradientBoostingRegressorModel = build_model(GradientBoostingRegressorModel,
                                                       **housing,
                                                       scoring=scoring,
                                                       n_estimators=1000,
                                                       n_iter_no_change=5)

    # Fit the model. The testing error of the final model is really good (37k$).
    tree.train(x_train, y_train)
    print(f"{getattr(tree.model, 'n_estimators_')} boosting stages.")
    tree.print_testing_error(split_data["y_test"], tree.model.predict(split_data["x_test"]))

def hist_boosting_gradient():
    """
    Histogram boosting-gradient is a modified version of the gradient-boosting that uses a reduced
    number of splits when building the different trees. Resulting in a faster algorithm that is
    highly recommended for large datasets.
    """
    pass

############
# ANALYSIS #
############
def run_analysis() -> None:
    """Boosting with adaptative boosting (AdaBoost) and grandient-boosting decision tree."""
    # Load data
    penguins = load_penguins()

    # Decision tree and AdaBoost
    #decision_tree(penguins)
    #ada_boost(penguins)

    # Gradient-boosting
    #manual_gradient_boosting()
    #gradient_boosting()
    hist_boosting_gradient()


if __name__ == "__main__":
    run_analysis()
