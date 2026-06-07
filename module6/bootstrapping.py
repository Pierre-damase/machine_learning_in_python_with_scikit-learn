import data_handler as dh
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from model import BaggingRegressorModel, DecisionTreeRegressorModel
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from types_config import DataSetType
from visualisation import plot_bootstrap_samples

type ClassModelTypes = (BaggingRegressorModel | DecisionTreeRegressorModel)

GENERATED_DATASET_FEATURE = "Feature"
GENERATED_DATASET_TARGET = "Target"

########
# DATA #
########
def load_californian_housing() -> DataSetType:
    """Load californian housing dataset."""
    return dh.load_california_dataset()

def load_generated_data(n_samples=30) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Generate simple synthetic dataset to better understand bagging. The target is a non-linear
    function.

    Return
    ------
    x_train: training features

    x_test: testing features, i.e. generated data not use to train the model

    y_train: training targets
    """
    x_min, x_max = -3, 3

    # Create a random number generator
    rng = np.random.default_rng(1)

    # Features
    x_train = rng.uniform(x_min, x_max, size=n_samples)
    x_test = np.linspace(x_max, x_min, num=300)

    # Target
    noise = 4.0 * rng.normal(size=(n_samples,))
    y_train = x_train**3 - 0.5 * (x_train + 1) ** 2 + noise
    y_train /= y_train.std()

    return pd.DataFrame(x_train, columns=[GENERATED_DATASET_FEATURE]), \
        pd.DataFrame(x_test, columns=[GENERATED_DATASET_FEATURE]), \
        pd.Series(y_train, name=GENERATED_DATASET_TARGET)


#################
# VISUALISATION #
#################
def scatterplot(x_train: pd.DataFrame,
                x_test: pd.DataFrame,
                y_train: pd.Series,
                y_predicted: npt.NDArray[np.float64],
                title: str):
    """Scatterplot."""
    sns.scatterplot(x=x_train[GENERATED_DATASET_FEATURE], y=y_train, color="black", alpha=0.5)
    plt.plot(x_test[GENERATED_DATASET_FEATURE], y_predicted, label="Decision function")
    plt.title(title)
    plt.show()


#########
# MODEL #
#########
def cross_validation(model_class: ClassModelTypes,
                     x_data: pd.DataFrame,
                     y_data: pd.Series,
                     nb_fold: int = 10) -> None:
    """Perfom cross-validation."""
    scores = model_class.make_cross_validate(x_data, y_data, nb_fold=nb_fold)
    model_class.print_cross_validate(scores)

def predict(model_class: ClassModelTypes,
            x_test: pd.DataFrame):
    return model_class.model.predict(x_test)

def decision_tree(x_data: pd.DataFrame,
                  y_data: pd.Series,
                  param_grid: dict[str, list[float|int|None]] | None = None,
                  **kwargs) -> DecisionTreeRegressorModel:
    """
    Build decision tree model.

    Parameters
    ----------
    x_data: the whole features

    y_data: the whole targets

    **kwargs: additional parameter for the model
    """
    # Build model
    regression: DecisionTreeRegressorModel = DecisionTreeRegressorModel.build(random_state=0,
                                                                              **kwargs)

    if param_grid is not None:
        # Split data
        split_data = dh.sklearn_train_test_split(x_data, y_data)

        # Grid-search cv
        regression.automated_search_cross_validation(GridSearchCV,
                                                     param_grid,
                                                     x_data,
                                                     y_data,
                                                     split_data["x_train"],
                                                     split_data["y_train"],
                                                     cv=10)
    else:
        # Cross-validation
        cross_validation(regression, x_data, y_data)

    # Train the model
    regression.start(x_train=x_data, y_train=y_data)

    return regression

def bagging_regressor(x_data: pd.DataFrame, y_data: pd.Series):
    """
    Build a bagging regressor model using decision tree as estimator.

    Parameters
    ----------
    x_data: the whole features

    y_data: the whole targets
    """
    # Build model
    estimator = DecisionTreeRegressor(random_state=0)
    regression: BaggingRegressorModel = BaggingRegressorModel.build(estimator=estimator,
                                                                    n_estimators=20,
                                                                    random_state=0)

    # Cross-validation
    cross_validation(regression, x_data, y_data)


############
# ANALYSIS #
############
def run_ensemble_models_example():
    """Use californian housing dataset as an introductory example to ensemble models."""
    # Load data
    housing = load_californian_housing()

    # Decision tree with default parameter. Really low testing score (~0.24), hence the model is
    # not good at all.
    decision_tree(**housing)

    # Use a grid-search cv to find the optimal set of paramaters. In this case,
    # {"max_depth": None, "min_samples_leaf": 1, "min_samples_split": 50} and the generalization
    # performance of the tuned model is much better but still low (~0.51)
    decision_tree(**housing, param_grid={
        "max_depth": [5, 8, None],
        "min_samples_split": [2, 10, 30, 50],
        "min_samples_leaf": [0.01, 0.05, 0.1, 1]
    })

    # Bagging regressor with decision tree. Without tunin hyperparameter, the model is already
    # better than the tuned decision tree (~0.6)
    bagging_regressor(**housing)

def run_bagging():
    """
    Use generated data to better understand bagging, which stand for Bootstrap aggregating. Bagging
    uses bootstrap resampling (random sampling with replacement) to learn several models on random
    variations of the training set
    """
    # Load data
    x_train, x_test, y_train = load_generated_data()

    # Decision tree
    regression = decision_tree(x_train, y_train, max_depth=3)
    scatterplot(x_train, x_test, y_train, predict(regression, x_test),
                title="Prediction by a single decision tree")

    # Generate 3 bootstrap samples from the data
    bootstrap_samples = dh.make_bootstrap_samples(x_train, y_train, 3)
    plot_bootstrap_samples({"x_data": x_train, "y_data": y_train},
                           bootstrap_samples,
                           GENERATED_DATASET_FEATURE)

def run_random_forest():
    pass

def run_analysis():
    """Bootsrapping with bagging algorithm and random forest."""
    # Use californian housing as an introductory example to ensemble models
    # run_ensemble_models_example()

    # Use generated data to better understand bagging
    run_bagging()

    run_random_forest()


if __name__ == "__main__":
    run_analysis()
