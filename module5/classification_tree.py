import data_handler as dh
import matplotlib as mpl
import numpy as np
import pandas as pd
from config import DataPath, TargetColumn
from model import DecisionTreeClassifierModel, LogisticRegressionModel
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from types_config import DataSetType

PENGUIN_FEATURES = ["Culmen Length (mm)", "Culmen Depth (mm)"]

type ClassModelTypes = (DecisionTreeClassifierModel | LogisticRegressionModel)

########
# DATA #
########
def load_data() -> DataSetType:
    data = dh.get_subset(dh.load_data_from_file(DataPath.PENGUIN.value),
                         columns=PENGUIN_FEATURES + [TargetColumn.PENGUIN]).dropna()
    return dh.get_subset(data, PENGUIN_FEATURES), pd.Series(data[TargetColumn.PENGUIN])


########
# PLOT #
########
def plot_decision_boundary(model: ClassModelTypes,
                           data: pd.DataFrame, x_data: pd.DataFrame) -> None:
    """Plot decision boundary."""
    model.decision_boundary_display(data,
                                    x_data,
                                    response_method="predict",
                                    hue=TargetColumn.PENGUIN.value,
                                    cmap="tab10",
                                    norm=getattr(mpl, "colors").Normalize(vmin=-0.5, vmax=8.5),
                                    multiclass_colors=["blue", "orange", "green"],
                                    palette=["tab:blue", "tab:orange", "tab:green"],
                                    alpha=0.5)


##############
# PREDICTION #
##############
def prediction(model: DecisionTreeClassifier, culmen_length: float, culmen_depth: float) -> None:
    """
    Perform some simple prediction to assess the model.

    predict_proba can be used to predict the class probabilities of the input samples
    """
    data = pd.DataFrame({
        "Culmen Length (mm)": [culmen_length], "Culmen Depth (mm)": [culmen_depth]
    })
    print(f"A penguin with a culmen length of {culmen_length} mm and a culmen depth of "
          f"{culmen_depth} is most likely to be {model.predict(data)[0]} with a probability of "
          f"{model.predict_proba(data)[0].max():.3f}")


#########
# MODEL #
#########
def classifier_model(model_class: type[ClassModelTypes],
                     penguins: DataSetType,
                     test_size: float = 0.25,
                     param_grid: dict[str, list[int]] | None = None,
                     **kwargs) -> ClassModelTypes:
    """
    Perform either a logistic regression or a classification tree.

    Paramater
    ---------
    model_class: the model to apply

    penguins: a tuple of data + targets

    test_size: portion of the dataset to include in the test split

    param_grid: parameter grid to perform hyperparameter tuning with grid-search cv

    **kwargs: additional parameters for the model such as max_depth for decision tree
    """
    # Split data
    x_train, x_test, y_train, y_test = dh.sklearn_train_test_split(*penguins, test_size=test_size)

    # Build
    regression = model_class.build(**kwargs)

    if param_grid is None:
        # Train model
        regression.start(x_train, x_test, y_train, y_test)

        # Decision function boundary
        plot_decision_boundary(regression, pd.concat([x_train, y_train], axis=1), x_train)

        # For multi-class problem, it could be useful to use this plot instead of the default one
        # called decision_boundary_display
        regression.multi_decision_boundary_display(x_test,
                                                   y_test,
                                                   response_method="predict_proba",
                                                   cmap="Blues")
    else:
        regression.automated_search_cross_validation(GridSearchCV,
                                                     param_grid,
                                                     penguins[0],
                                                     penguins[1],
                                                     x_train,
                                                     y_train,
                                                     cv=10)

    return regression

def logistic_regression(penguins: DataSetType) -> None:
    """
    In a 2-dimensional feature space, a linear classifier finds the oblique line that best
    separate the classes. In a multi-dimensional features space, a linear classifier perform
    the same task except that more than one line is fitted.
    """
    classifier_model(LogisticRegressionModel, penguins)

def classification_tree(penguins: DataSetType,
                        max_depth: int | None = None,
                        param_grid: dict[str, list[int]] | None = None,
                        test_size: float = 0.25) -> None:
    """
    Try out various training test size in order to visualize the impact of such variable on the
    decision rule.

    A model with a maximum depth of 1 is not powerful enough to separate the three species, which
    explain the low accurarcy compared to the linear model. Moreover, with a maximum depth of 1 the
    model can only predict the two more represented classes out of the three.

    A model with a maximum depth of 2 is way more powerful and is able to predict with a pretty
    good accuracy the 3 classes. Moreover, the model use both features culmen length and culmen
    depth in order to create spliting rules.
    """
    # Build model
    tree = classifier_model(DecisionTreeClassifierModel,
                            penguins,
                            test_size=test_size,
                            max_depth=max_depth,
                            param_grid=param_grid)

    if max_depth is not None:
        # Decision tree
        if max_depth < 10:
            tree.plot_decision_tree(PENGUIN_FEATURES)

            # Try various prediction for max_depth=1
            if max_depth == 1:
                prediction(tree.model, culmen_length=41, culmen_depth=0)
                prediction(tree.model, culmen_length=43, culmen_depth=0)


############
# ANALYSIS #
############
def run_classifier_model(penguins: DataSetType):
    """
    Run classifier model to display the difference between a logistic regression and a decision
    tree.
    """
    # Logistic regression
    logistic_regression(penguins)

    # Decision tree with a single decision node
    classification_tree(penguins, max_depth=1)
    classification_tree(penguins, max_depth=2)

def run_hyperparameter_tuning(penguins: DataSetType):
    """Tune hyperparameter max_depth for decision tree."""
    # Display the effect of the max_depth parameter
    # classification_tree(penguins, max_depth=3)
    # classification_tree(penguins, max_depth=30)

    # Perform a grid-search cv to tune the max_depth
    classification_tree(penguins, param_grid={"max_depth": np.arange(2, 10, 1)})

    # Run a decision tree with the optimum max_depth parameter
    classification_tree(penguins, max_depth=5)

def run_analysis():
    """Use decision tree to build machine learning model with classification task."""
    # Load data
    penguins = load_data()

    # Run classifier model, either logistic regression or classification tree
    run_classifier_model(penguins)

    # Hyperparameter tuning
    run_hyperparameter_tuning(penguins)

if __name__ == "__main__":
    run_analysis()
