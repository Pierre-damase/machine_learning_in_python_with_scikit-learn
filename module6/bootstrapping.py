import data_handler as dh
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from config import DataPath, TargetColumn
from matplotlib.axes import Axes
from model import (BaggingClassifierModel, BaggingRegressorModel,
                   DecisionTreeClassifierModel, DecisionTreeRegressorModel,
                   RandomForestClassifierModel, RandomForestRegressorModel,
                   RegressorMixin, RidgeRegressionModel)
from scipy.stats import randint
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import (MinMaxScaler, OrdinalEncoder,
                                   PolynomialFeatures)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from types_config import DataSetType, Tmodel, Tpreprocessor
from visualisation import plot_bootstrap_samples

type ClassModelTypes = (BaggingClassifierModel
                        | BaggingRegressorModel
                        | DecisionTreeClassifierModel
                        | DecisionTreeRegressorModel
                        | RandomForestClassifierModel
                        | RandomForestRegressorModel)

GENERATED_DATASET_FEATURE = "Feature"
GENERATED_DATASET_TARGET = "Target"

PENGUIN_FEATURE = "Flipper Length (mm)"
PENGUIN_TARGET = "Body Mass (g)"

########
# DATA #
########
def load_californian_housing() -> DataSetType:
    """Load californian housing dataset."""
    return dh.load_california_dataset()

def load_adult_census() -> DataSetType:
    """Load adult census dataset."""
    return dh.load_data_from_file(DataPath.ADULT_CENSUS.value, target=TargetColumn.ADULT_CENSUS)

def load_penguins() -> DataSetType:
    """Load penguin dataset."""
    return dh.load_data_from_file(DataPath.PENGUIN.value,
                                  target=PENGUIN_TARGET,
                                  columns=[PENGUIN_FEATURE],
                                  drop_na=True)

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
                title: str | None = None,
                label: str = "Decision function",
                column: str = GENERATED_DATASET_FEATURE,
                ax: Axes | None = None):
    """Scatterplot."""
    ax = sns.scatterplot(x=x_train[column], y=y_train, color="black", alpha=0.5, ax=ax)
    ax.plot(x_test[column], y_predicted, label=label)
    ax.legend()
    if title is not None:
        plt.title(title)
        plt.show()

def scatterplot_for_manual_bagging(x_train: pd.DataFrame,
                                   x_test: pd.DataFrame,
                                   y_train: pd.Series,
                                   y_predicted: list[npt.NDArray[np.float64]],
                                   bag_prediction: npt.NDArray[np.float64],
                                   title: str):
    """
    Scatterplot for manual bagging in order to plot the prediction of each bootstrap sample as well
    as the final prediction.
    """
    sns.scatterplot(x=x_train[GENERATED_DATASET_FEATURE], y=y_train, color="black", alpha=0.5)

    # Prediction of each decision tree model
    for i in range(len(y_predicted)):
        plt.plot(x_test[GENERATED_DATASET_FEATURE],
                 y_predicted[i],
                 linestyle="--",
                 alpha=0.8,
                 label=f"Decision function #{i}")

    # Final prediction
    plt.plot(x_test[GENERATED_DATASET_FEATURE],
             bag_prediction,
             label="Average prediction",
             linestyle="-")

    plt.legend()
    plt.title(title)
    plt.show()

def scatterplot_for_sklearn_bagging(regression: BaggingRegressorModel,
                                    x_train: pd.DataFrame,
                                    x_test: pd.DataFrame,
                                    y_train: pd.Series,
                                    bag_predictions: npt.NDArray[np.float64],
                                    title: str):
    """Scatterplot for bagging using scikit-learn implemenations."""
    # Convert testing data into numpy array to avoid a warning in scikit-learn
    x_test_np = x_test.to_numpy()

    # Get estimators
    estimators = getattr(regression.model, "estimators_")

    # Set up plot
    nb_features = len(x_train.columns)

    _, axs = plt.subplots(ncols=nb_features, figsize=(20, 12))
    for i in range(nb_features):
        column = x_train.columns.values[i]
        if nb_features == 1:
            # For generated data, only one features, hence subplots return a single Axes object
            axs: list[Axes] = [axs]

        # Predict testing target and plot it only for small dataset
        if len(x_train) < 100:
            for j in range(len(estimators)):
                y_predicted = estimators[j].predict(x_test_np)
                axs[i].plot(x_test[column],
                            y_predicted,
                            linestyle="--",
                            alpha=0.1,
                            color="tab:gray",
                            label="Predictions of individual trees" if j == 0 else None)

        # Scatterplot
        scatterplot(x_train,
                    x_test,
                    y_train,
                    bag_predictions,
                    label="Prediction of ensemble",
                    column=column,
                    ax=axs[i])
    plt.title(title)
    plt.show()


#########
# MODEL #
#########
def cross_validation(model_class: ClassModelTypes,
                     x_data: pd.DataFrame,
                     y_data: pd.Series,
                     nb_fold: int = 10,
                     scoring: str | None = None) -> None:
    """Perfom cross-validation in order to evaluate the generalization performance."""
    scores = model_class.make_cross_validate(x_data, y_data, nb_fold=nb_fold, scoring=scoring)
    model_class.print_cross_validate(scores)

def predict(model_class: ClassModelTypes,
            x_test: pd.DataFrame):
    return model_class.model.predict(x_test)

def decision_tree(model_class: type[DecisionTreeClassifierModel | DecisionTreeRegressorModel],
                  x_data: pd.DataFrame,
                  y_data: pd.Series,
                  param_grid: dict[str, list[float|int|None]] | None = None,
                  transformers: list[Tpreprocessor] | None = None,
                  **kwargs) -> DecisionTreeRegressorModel:
    """
    Build decision tree model.

    Parameters
    ----------
    model_class: either a decision tree classifier for classification task or a decision tree
    regressor for a regression task

    x_data: the whole features

    y_data: the whole targets

    transformers: if set initialize a decision tree within a pipeline

    **kwargs: additional parameter for the model
    """
    # Build model
    if transformers is None:
        regression = model_class.build(random_state=0, **kwargs)
    else:
        regression = model_class.build_pipeline(transformers, random_state=0, **kwargs)

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

def bagging_regressor(estimator: type[Tmodel],
                      x_data: pd.DataFrame,
                      y_data: pd.Series,
                      x_test: pd.DataFrame = pd.DataFrame(),
                      y_test: pd.Series = pd.Series(),
                      param_distribution: dict[str, list[float|int|None]] | None = None,
                      n_estimators: int = 20,
                      scoring: str | None = None) -> BaggingRegressorModel:
    """
    Build a bagging ensemble for regression task.

    Parameters
    ----------
    estimator: the base estimator of the bagging procedure

    x_data: the whole features

    y_data: the whole targets

    x_test: testing data, use by the tuning part to test the performance of the best model

    y_test: testing targets, use by the tuning part to test the performance of the nest model

    param_distribution: hyperparameter distribution for tuning

    n_estimators: the number of base estimators in the ensemble

    **kwargs: additional parameters for the base estimator
    """
    # Build model
    regression: BaggingRegressorModel = BaggingRegressorModel.build(estimator=estimator,
                                                                    n_estimators=n_estimators,
                                                                    random_state=0)

    if param_distribution is not None and not x_test.empty and not y_test.empty:
        # Grid-search cv
        regression.automated_search_cross_validation(RandomizedSearchCV,
                                                     param_distribution,
                                                     x_data,
                                                     y_data,
                                                     x_test,
                                                     y_test,
                                                     cv=10,
                                                     n_iter=100)

    else:
        # Cross-validation
        cross_validation(regression, x_data, y_data, scoring=scoring)

    return regression

def manual_bagging_regressor(x_train: pd.DataFrame,
                             x_test: pd.DataFrame,
                             y_train: pd.Series) -> None:
    """Manual bagging algorithm implementation for a regression task."""
    # 1. Bootstrapping part. Generate many variations of the original training set
    bootstrap_samples = dh.make_bootstrap_samples(x_train, y_train, 3)
    plot_bootstrap_samples({"x_data": x_train, "y_data": y_train},
                           bootstrap_samples,
                           GENERATED_DATASET_FEATURE)
    dh.analyse_bootstrap_samples(bootstrap_samples)

    # 2. Fitting part. Each bootstrap sample is fit using a decision tree model for example
    y_predicted = []
    for sample in bootstrap_samples:
        regression = decision_tree(DecisionTreeRegressorModel,
                                   sample["x_data"],
                                   sample["y_data"],
                                   max_depth=3)
        y_predicted.append(regression.model.predict(x_test))

    # 3. Aggregation part. The final prediction will consider the aggregated prediction of each
    # "bootstrap" model, which is an average prediction of each bootstrap sample in regression.
    bag_predictions = np.mean(y_predicted, axis=0)

    # 4. Display the final prediction as well as the prediction of each model
    scatterplot_for_manual_bagging(x_train,
                                   x_test,
                                   y_train,
                                   y_predicted,
                                   bag_predictions,
                                   title="Prediction by trees trained on various bootstraps")

def sklearn_bagging_regressor(estimator: type[Tmodel],
                              x_data: pd.DataFrame,
                              y_data: pd.Series,
                              x_train: pd.DataFrame = pd.DataFrame(),
                              x_test: pd.DataFrame = pd.DataFrame(),
                              y_train: pd.Series = pd.Series(),
                              y_test: pd.Series = pd.Series(),
                              n_estimators: int = 100,
                              param_distribution: dict[str, list[float|int|None]] | None = None,
                              scoring: str | None = None) -> None:
    """
    Bagging in scikit-learn for a regression task.

    The bagging procedure is implements as a meta-estimator: it takes a base model (estimator) that
    is cloned several times and trained independently on each bootstrap sample.

    Parameters
    ----------
    estimator: the base estimator of the bagging procedure. In this case either a decision tree or
    a polynomial pipeline

    x_data: the whole features

    y_data: the whole targets

    x_train: training data

    x_test: testing data, use by the tuning part to test the performance of the best model

    y_train: training target

    y_test: testing targets, use by the tuning part to test the performance of the nest model

    param_distribution: to tune bagging regressor hyperparameters by randomized-search cv

    scoring: the scoring function use either by the cv or the randomized-search cv
    """
    # For generated data, use the whole dataset as the training and ploting data.
    x_train = x_train if not x_train.empty else x_data
    y_train = y_train if not y_train.empty else y_data

    # 1. Bootstrapping part. Build a bagging ensemble with n_estimators=100 to get a stronger
    # smoothing effect. x_test and y_test only use by the tuning part.
    regression = bagging_regressor(estimator,
                                   x_data,
                                   y_data,
                                   x_test,
                                   y_test,
                                   n_estimators=n_estimators,
                                   param_distribution=param_distribution,
                                   scoring=scoring)

    # 2. Fitting part.
    regression.start(x_train, x_test, y_train, y_test)

    # 3. Aggregation part.
    x_test_ = x_test
    if len(x_test) > 100 and not y_test.empty:
        x_test_, _ = dh.dataframe_sample(x_test, y_test, size=100)
    bag_predictions = regression.model.predict(x_test_)

    # Visualisation
    if len(x_train) < 100:
        scatterplot_for_sklearn_bagging(regression,
                                        x_train ,
                                        x_test_,
                                        y_train,
                                        bag_predictions,
                                        "Prediction by bagging regressor")

def build_model(model_class: type[BaggingClassifierModel
                                       | RandomForestClassifierModel
                                       | RandomForestRegressorModel],
                x_data: pd.DataFrame,
                y_data: pd.Series,
                transformers: list[Tpreprocessor] | None = None,
                scoring: str | None = None,
                **kwargs):
    """
    Build a bagging ensemble for classification task or a random forest.

    Parameters
    ----------
    transformers: a list of preprocessor to deal with categorical and numerical features

    x_data: the whole input data (features)

    y_data: the whole targets

    **kwargs: additional parameter for the model
    """
    # Build model
    if transformers is not None:
        model = model_class.build_pipeline(transformers, **kwargs)
    else:
        model = model_class.build(**kwargs)

    # Cross-validation
    cross_validation(model, x_data, y_data, scoring=scoring)

    return model


############
# ANALYSIS #
############
def run_ensemble_models_example() -> None:
    """Use californian housing dataset as an introductory example to ensemble models."""
    # Load data
    housing = load_californian_housing()

    # Decision tree with default parameter. Really low testing score (~0.24), hence the model is
    # not good at all.
    decision_tree(DecisionTreeRegressorModel, **housing)

    # Use a grid-search cv to find the optimal set of paramaters. In this case,
    # {"max_depth": None, "min_samples_leaf": 1, "min_samples_split": 50} and the generalization
    # performance of the tuned model is much better but still low (~0.51)
    param_grid = {
        "max_depth": [5, 8, None],
        "min_samples_split": [2, 10, 30, 50],
        "min_samples_leaf": [0.01, 0.05, 0.1, 1]
    }
    decision_tree(DecisionTreeRegressorModel, **housing, param_grid=param_grid)

    # Bagging regressor with decision tree. Without tunin hyperparameter, the model is already
    # better than the tuned decision tree (~0.6)
    bagging_regressor(DecisionTreeRegressor(random_state=0, max_depth=3), **housing)

def run_sklearn_bagging_on_generated_data(x_train: pd.DataFrame,
                                          x_test: pd.DataFrame,
                                          y_train: pd.Series) -> None:
    """Bagging in scikit-learn with generated data."""
    # Scikit-learn implementation of the bagging procedure with a simple decision tree as base
    # estimator
    estimator = DecisionTreeRegressor(random_state=0, max_depth=3)
    sklearn_bagging_regressor(estimator, x_data=x_train, y_data=y_train, x_test=x_test)

    # Scikit-learn implementation of the bagging procedure with a complex pipelines as base
    # estimator. Data are (i) scale to the 0-1 range, (ii) then some data engineering with a
    # PolynomialFeatures and (iii) the pipeline feeds tge resulting non-linear features to a
    # regularized linear regression model for the final prediction of the target.
    estimator = RidgeRegressionModel.build_pipeline(
        transformers=[
            MinMaxScaler(),
            PolynomialFeatures(degree=4, include_bias=False)
        ],
        alpha=1e-10
    )
    sklearn_bagging_regressor(estimator.model, x_data=x_train, y_data=y_train, x_test=x_test)


def run_bagging_on_generated_data() -> None:
    """
    Use generated data to better understand bagging, which stand for Bootstrap aggregating. Bagging
    uses bootstrap resampling (random sampling with replacement) to learn several models on random
    variations of the training set
    """
    # Load data
    x_train, x_test, y_train = load_generated_data()

    # Decision tree
    regression = decision_tree(DecisionTreeRegressorModel, x_train, y_train, max_depth=3)
    scatterplot(x_train, x_test, y_train, predict(regression, x_test),
                title="Prediction by a single decision tree")

    # Manual implementation of the bagging algorithm
    manual_bagging_regressor(x_train, x_test, y_train)

    # Scikit-learn implementation of the bagging procedure
    run_sklearn_bagging_on_generated_data(x_train, x_test, y_train)

def run_bagging() -> None:
    """Use real data this time, the californian dataset, to run run bagging procedure."""
    # Load data
    housing = load_californian_housing()
    split_data = dh.sklearn_train_test_split(**housing, test_size=0.5)

    # Evaluate the generalization performance of the model using the mean absolute error (~45 k$)
    estimator = DecisionTreeRegressor()
    sklearn_bagging_regressor(estimator,
                              **housing,
                              **split_data,
                              scoring="neg_mean_absolute_error")

    # Use a randomized-search cv in order to tune the bagging regressor hyperparameters (~35 k$)
    #param_distribution = {
    #    "n_estimators": randint(10, 30), # 26
    #    "max_samples": [0.5, 0.8, 1.0], # 1.0
    #    "max_features": [0.5, 0.8, 1.0], # 0.8
    #    "estimator__max_depth": randint(3, 10) # 9
    #}
    #sklearn_bagging_regressor(estimator,
    #                **housing,
    #                **split_data,
    #                param_distribution=param_distribution,
    #                scoring="neg_mean_absolute_error")

def run_random_forest_classifier() -> None:
    """Random forest model for classification task."""
    # Load data
    adult_census = load_adult_census()

    # Set up transformers to deal with categorical and numerical features + additional param
    transformers = [
        (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            selector(dtype_include=str)
        ),
        ("passthrough", selector(dtype_exclude=str))
    ]
    model_parameters = {"n_estimators": 50, "n_jobs": 2, "random_state": 0}

    # Decision tree. The testing error is already quite good (~0.81).
    decision_tree(DecisionTreeClassifierModel, **adult_census, transformers=transformers)

    # Bagging classifier. There is already some improvement of the testing error (~0.854)
    build_model(BaggingClassifierModel,
                **adult_census,
                transformers=transformers,
                estimator=DecisionTreeClassifier(random_state=0),
                **model_parameters)

    # Random forest classifier. There is a slight improvement of the testing error (~0.857)
    build_model(RandomForestClassifierModel,
                **adult_census,
                transformers=transformers,
                **model_parameters)

def run_random_forest_regressor() -> None:
    """Random forest for regression task."""
    # Load data
    penguins = load_penguins()
    split_data = dh.sklearn_train_test_split(**penguins)

    # Random forest regressor. Mean absolute error is ~350.
    regression: RandomForestRegressorModel = build_model(RandomForestRegressorModel,
                                                         **penguins,
                                                         scoring="neg_mean_absolute_error",
                                                         n_estimators=3)

    # Train model
    regression.start(x_train=split_data["x_train"], y_train=split_data["y_train"])

    # Create an evenly spaced values dataset and use it as the testing set for the plotting
    data_range = pd.DataFrame(np.linspace(170, 235, num=len(split_data["y_train"])),
                              columns=penguins["x_data"].columns)

    # Plot prediction of each tree
    regression.plot_decision_tree_predictions(x_train=split_data["x_train"],
                                              x_test=data_range,
                                              y_train=split_data["y_train"],
                                              predictions=regression.model.predict(data_range),
                                              title="Random forest regression")


def run_random_forest() -> None:
    """
    Base estimator of random forest is always a decision tree. Morever, when training a tree, the
    search for the best split is only done on a subset of the original features, which is taken
    randomly and differs for each split node.
    """
    # Classification task
    run_random_forest_classifier()

    # Regression task
    run_random_forest_regressor()

def run_analysis() -> None:
    """Bootsrapping with bagging algorithm and random forest."""
    # Use californian housing as an introductory example to ensemble models
    run_ensemble_models_example()

    # Use generated data to better understand bagging
    run_bagging_on_generated_data()

    # Use californian housing as an example for bagging and hyperparameter tuning
    run_bagging()

    # Use random forest algorithm, for both classification and regression task.
    run_random_forest()


if __name__ == "__main__":
    run_analysis()
