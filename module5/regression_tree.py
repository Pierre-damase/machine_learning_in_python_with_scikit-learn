import data_handler as dh
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from config import GENERATED_DATASET_FEATURES, DataPath, TargetColumn
from model import DecisionTreeRegressorModel, LinearRegressionModel
from sklearn.model_selection import GridSearchCV
from types_config import (DataSetType, SearchCvHyperparamType,
                          SearchCvParameters, SearchOuterCv)

PENGUIN_FEATURE = "Flipper Length (mm)"
PENGUIN_TARGET = "Body Mass (g)"

type ClassModelTypes = (DecisionTreeRegressorModel | LinearRegressionModel)

########
# DATA #
########
def load_data() -> DataSetType:
    """Load penguin dataset."""
    return dh.load_data_from_file(DataPath.PENGUIN.value,
                                  target=PENGUIN_TARGET,
                                  columns=[PENGUIN_FEATURE],
                                  drop_na=True)

def generate_test_data(x_train: pd.DataFrame, offset: int = 0) -> pd.DataFrame:
    """
    Generate some 'test' data to illustrate how decision trees predict in regression setting. We
    speak about test data because such data were not used to train the model.

    Keep in mind that computing an evaluation metric on such synthetic test set would be
    meaningless since the synthetic dataset does not follow the same distribution as the real world
    data.
    """
    return pd.DataFrame(
        np.arange(x_train[PENGUIN_FEATURE].min() - offset,
                  x_train[PENGUIN_FEATURE].max() + offset),
        columns=[PENGUIN_FEATURE]
    )

def generate_blobs_data() -> pd.DataFrame:
    """
    Build a dataset to illustrate asymmetry in decision tree. The dataset is composed of 2 subsets:
      - A 1st subset where a clear symmetrical separation should be found between the data
      - A 2nd subset where both samples from both classes are mixed
    Therefore, it implies that a decision tree will need more splits to classify properly samples
    from the second subset than from the first one.
    """
    # Generate both interlaced and separated data
    interlaced_data = dh.make_blobs_dataset(300, [[0, 0], [-1, -1]])
    separated_data = dh.make_blobs_dataset(300, [[3, 6], [7, 0]])

    # Concatenate features / targets from both dataset
    x_data = np.concat([interlaced_data["x_data"], separated_data["x_data"]], axis=0)
    y_data = np.concat([interlaced_data["y_data"], separated_data["y_data"]])

    # Build dataframe. Both dataframe share the same name for features and target.
    data = pd.DataFrame(np.concatenate([x_data, y_data[:, np.newaxis]], axis=1),
                        columns=GENERATED_DATASET_FEATURES + [interlaced_data["y_data"].name])

    # Set type
    data[interlaced_data["y_data"].name] = data[interlaced_data["y_data"].name].astype(np.int32)

    return data


#################
# VISUALISATION #
#################
def scatterplot(data: pd.DataFrame,
                x: str,
                y: str,
                title: str,
                x_test: pd.DataFrame | None = None,
                y_predicted: npt.NDArray[np.float64] | None = None,
                label: str | None = None) -> None:
    """Scatterplot visualisation of the data."""
    # Plot
    sns.scatterplot(data=data, x=x, y=y, color="black", alpha=0.5)

    # Add prediction function
    if x_test is not None and y_predicted is not None:
        plt.plot(x_test, y_predicted, label=label)
        plt.legend()

    plt.title(title)
    plt.show()


#########
# MODEL #
#########
def regressor_model(model_class: type[ClassModelTypes],
                    data: pd.DataFrame,
                    x_train: pd.DataFrame,
                    x_test: pd.DataFrame,
                    y_train: pd.Series,
                    param_grid: SearchCvHyperparamType | None = None,
                    **kwargs) -> None:
    """
    Perform either a linear regression or a regression tree.

    Parameters
    ----------
    model_class: the model to apply

    data: the whole dataset

    x_train: training data

    x_test: generated testing data

    y_train: training target

    **kawargs: additional parameters for the model such as max_depth for decision treee
    """
    # Build
    regression = model_class.build(**kwargs)

    if param_grid is None:
        # Train model
        regression.start(x_train=x_train, y_train=y_train)

        # Predict test data
        y_predicted = regression.model.predict(x_test)

        # Visualize the linear approximation
        model_name = regression._estimator_class.__name__
        scatterplot(data,
                    x=PENGUIN_FEATURE,
                    y=PENGUIN_TARGET,
                    title=f"Prediction function using a {model_name}",
                    x_test=x_test[PENGUIN_FEATURE],
                    y_predicted=y_predicted,
                    label=model_name)

        # For decision tree, plot the corresponding tree
        if model_name == "DecisionTreeRegressor" and kwargs["max_depth"] < 10:
            regression.plot_decision_tree([PENGUIN_FEATURE])
    else:
        search_outer_cv = SearchOuterCv(pd.DataFrame(data[PENGUIN_FEATURE]),
                                        pd.Series(data[PENGUIN_TARGET]))
        regression.automated_search_cv(search_class=GridSearchCV,
                                       search_params=SearchCvParameters(10),
                                       parameters=param_grid,
                                       x_train=x_train,
                                       y_train=y_train,
                                       search_outer_cv=search_outer_cv)


############
# ANALYSIS #
############
def run_regressor_model(penguin: pd.DataFrame,
                        x_train: pd.DataFrame,
                        x_test: pd.DataFrame,
                        y_train: pd.Series):
    """
    Run regressor model to display the difference between a linear regression and a decision tree.
    """
    # Linear regression
    regressor_model(LinearRegressionModel, penguin, x_train, x_test, y_train)

    # Regression tree. The decision tree does not have a priori distribution for the data and we do
    # not end-up with a straight line. Increasing the depth of the tree increases the number of
    # partitions and thus the number of constant values that the tree is capable of predicting.
    regressor_model(DecisionTreeRegressorModel, penguin, x_train, x_test, y_train, max_depth=1)
    regressor_model(DecisionTreeRegressorModel, penguin, x_train, x_test, y_train, max_depth=3)

    # Generate some test data with and offset value to check the extrapolation capabilities of each
    # model.
    # 1\ Linear model can extrapolate the mass of penguin without any issue.
    # 2\ Decision trees are non-parametric models and they cannot extrapolate. For flipper
    # lenghts below the minimum (above the maximum), the predicted mass of the penguin will always
    # be the one of the penguin with the shorter (higher) flipper length in the training data.
    x_test = generate_test_data(x_train, offset=30)
    regressor_model(LinearRegressionModel, penguin, x_train, x_test, y_train)
    regressor_model(DecisionTreeRegressorModel, penguin, x_train, x_test, y_train, max_depth=3)

def run_hyperparameter_tuning(penguin: pd.DataFrame,
                              x_train: pd.DataFrame,
                              x_test: pd.DataFrame,
                              y_train: pd.Series):
    """Tune hyperparameter max_depth for decision tree."""
    # Display the effect of the max_depth parameter
    # regressor_model(DecisionTreeRegressorModel, penguin, x_train, x_test, y_train, max_depth=2)
    # regressor_model(DecisionTreeRegressorModel, penguin, x_train, x_test, y_train, max_depth=30)

    # Perform a grid-search cv to tune the max_depth
    regressor_model(DecisionTreeRegressorModel,
                    penguin,
                    x_train,
                    x_test,
                    y_train,
                    param_grid={"max_depth": np.arange(2, 10, 1)})

    # Run a decision tree with the optimum max_depth parameter
    regressor_model(DecisionTreeRegressorModel, penguin, x_train, x_test, y_train, max_depth=3)

def run_decision_tree_with_asymmetrical_data():
    """
    For a given dataset, optimal generalization performance could be reached by growing some of the
    branches deeper than some others. To illustrate this, generate two dataset, one with separated
    data and one with interlaced data. Then run some decision tree model.
    """
    # Generate data
    blobs_data = generate_blobs_data()
    split_data = dh.sklearn_train_test_split(
        dh.get_subset(blobs_data, GENERATED_DATASET_FEATURES),
        pd.Series(blobs_data[TargetColumn.GENERATED_DATASET])
    )

    # Train decision tree with various max_depth and min_samples_leaf to check the effect of these
    # hyperparameters
    for max_depth in [2, 6]:
        for min_samples_leaf in [1, 60]:
            tree: DecisionTreeRegressorModel = DecisionTreeRegressorModel.build(
                max_depth=max_depth, min_samples_leaf=min_samples_leaf
            )
            tree.start(x_train=split_data["x_train"], y_train=split_data["y_train"])
            tree.decision_boundary_display(
                pd.concat([split_data["x_train"], split_data["y_train"]], axis=1),
                split_data["x_train"],
                response_method="predict",
                hue=TargetColumn.GENERATED_DATASET.value,
                cmap="RdBu",
                alpha=0.5)
            tree.plot_decision_tree(GENERATED_DATASET_FEATURES)

def run_analysis():
    """Use decision tree to build machine learning model with regression task."""
    # Load data
    data_train = load_data()
    x_test = generate_test_data(data_train["x_data"])
    penguin = pd.concat([data_train["x_data"], data_train["y_data"]], axis=1)

    # Run regressor model, either linear regression or regressor tree
    run_regressor_model(penguin, data_train["x_data"], x_test, data_train["y_data"])

    # Hyperparameter tuning
    run_hyperparameter_tuning(penguin, data_train["x_data"], x_test, data_train["y_data"])

    # Decision tree with asymmetrical data
    run_decision_tree_with_asymmetrical_data()

if __name__ == "__main__":
    run_analysis()
