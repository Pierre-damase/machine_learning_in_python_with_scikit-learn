import data_handler as dh
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from config import DataPath
from model import LinearRegressionModel
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, PolynomialFeatures,
                                   SplineTransformer)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from types_config import DataSetType
from visualisation import show_validation_curve

GENERATED_DATASET_FEATURE = "Feature"
GENERATED_DATASET_TARGET = "Target"

PENGUIN_FEATURES = ["Flipper Length (mm)", "Culmen Length (mm)", "Culmen Depth (mm)"]
PENGUIN_TARGET = "Body Mass (g)"

########
# DATA #
########
def generate_data(n_sample: int = 100, min: float = -1.4, max: float = 1.4) -> DataSetType:
    """
    Build a custom dataset consisting of a single feature. The target is built as a cubic
    polynomial on said feature. Some random fluctuations are added to the the target.
    """
    random = np.random.RandomState(0)
    len_data =  max - min

    data = np.sort(random.rand(n_sample) * len_data - len_data / 2)

    return {
        "x_data": pd.DataFrame(data, columns=[GENERATED_DATASET_FEATURE]),
        "y_data": pd.Series(data**3 - 0.5 * data**2 + random.randn(n_sample) * 0.3,
                            name=GENERATED_DATASET_TARGET)
    }

def load_penguins() -> DataSetType:
    """Load penguin dataset, extract numerical features of interest and drop na."""
    return dh.load_data_from_file(DataPath.PENGUIN.value,
                                  target=PENGUIN_TARGET,
                                  columns=PENGUIN_FEATURES,
                                  drop_na=True)


#################
# VISUALISATION #
#################
def scatterplot(data: DataSetType,
                x_data: npt.NDArray[np.float64] | None = None,
                y_predicted: npt.NDArray[np.float64] | None = None,
                title: str = "") -> None:
    """Plot scatterplot."""
    ax = sns.scatterplot(data=pd.concat([data["x_data"], data["y_data"]], axis=1),
                         x=GENERATED_DATASET_FEATURE,
                         y=GENERATED_DATASET_TARGET,
                         color="black",
                         alpha=0.5)
    if x_data is not None and y_predicted is not None:
        ax.plot(x_data, y_predicted)
        ax.set_title(title)
    plt.show()


#########
# MODEL #
#########
def build_pipeline(transformer: type[KBinsDiscretizer
                                     | Nystroem
                                     | PolynomialFeatures
                                     | SplineTransformer],
                   columns: list[str],
                   **kwargs):
    """Transform the input features."""
    return LinearRegressionModel.build_pipeline(transformers = [(transformer(**kwargs), columns)])

def run_model(model_class: type[DecisionTreeRegressor
                                | LinearRegression
                                | Pipeline
                                | SVR],
              data: DataSetType,
              x_data: npt.NDArray[np.float64],
              y_data: pd.Series,
              title: str,
              print_coef: bool = False,
              **kwargs) -> None:
    """
    Build and train a model.

    Parameter
    ---------
    generated_data: the whole dataset use for the plot

    x_data: the whole feature use to train the model and predict

    y_data: the target
    """
    # Build and train model
    model = model_class(**kwargs).fit(x_data, y_data)

    # Predict train data
    y_predicted = model.predict(x_data)

    # Plot the regression
    fit_score_plot_regression(model, data, x_data, y_data, y_predicted, title, print_coef)

def run_model_with_pipeline(model: Pipeline,
                            data: DataSetType,
                            x_data: npt.NDArray[np.float64],
                            y_data: pd.Series,
                            title: str) -> None:
    """
    Build and train a model within a pipeline.

    Parameter
    ---------
    data: the whole dataset use for the plot

    x_data: the whole feature use to train the model and predict

    y_data: the target
    """
    # Convert x_data into a DataFrame
    x_data_converted = pd.DataFrame({"feature": x_data[:, 0]})

    # Train the model
    model.fit(x_data_converted, y_data)

    #Predict the training data
    y_predicted = model.predict(x_data_converted)

    # Plot the regression
    fit_score_plot_regression(model, data, x_data, y_data, y_predicted, title)

def fit_score_plot_regression(model: DecisionTreeRegressor
                              | LinearRegression
                              | SVR,
                              data: DataSetType,
                              x_data: npt.NDArray[np.float64],
                              y_data: pd.Series,
                              y_predicted: npt.NDArray[np.float64],
                              title: str,
                              print_coef: bool = False) -> None:
    """
    Predict the data use to train the model then plot the regression along with the fit score.
    """
    # Coefficient to display for linear model
    if print_coef:
        print(f"Weight {getattr(model, 'coef_')[0]:.3f} and "
              f"intercept {getattr(model, 'intercept_'):.3f}")

    # Training error
    mse = mean_squared_error(y_data, y_predicted)
    scatterplot(data, x_data, y_predicted, title=f"{title} (mse={mse:.3f})")

def cross_validation(model: LinearRegressionModel,
                     x_data: pd.DataFrame,
                     y_data: pd.Series):
    """Perform a cross-validation and display the testing error."""
    scores = model.kfold_cross_validate(
        x_data, y_data, nb_fold=10, scoring="neg_mean_absolute_error"
    )
    model.print_cross_validate(scores)

############################
# SIMPLE LINEAR REGRESSION #
############################
def run_linear_regression(data: DataSetType,
                          x_data: npt.NDArray[np.float64],
                          y_data: pd.Series) -> None:
    """
    Build linear regression.

    Scikit-learn except a 2D matrix of shape (n_samples, n_features). Therefore, a reshape into a
    matrix with a single column is needed for 1D vector.
    """
    title = "Simple linear regression"
    run_model(LinearRegression, data, x_data, y_data, title, print_coef=True)


############################
# DECISION TREE REGRESSION #
############################
def run_decision_tree_regression(data: DataSetType,
                                 x_data: npt.NDArray[np.float64],
                                 y_data: pd.Series) -> None:
    """
    First solution to deal with non-linearity is to use a model that can natively deal with it such
    as decision tree.
    """
    title = "Decision tree regressor"
    run_model(DecisionTreeRegressor, data, x_data, y_data, title, max_depth=3)


####################
# DATA ENGINEERING #
####################
def exploration(x_data: npt.NDArray[np.float64]):
    """Usefull to perform so data exploration."""
    # Either manually perform the data engineering
    x_data_expanded = np.concatenate([x_data, x_data**2, x_data**3], axis=1)

    # Or used the PolynomialFeatures of scikit-learn with include bias set to false in order to
    # avoid the creation of a constant feature perfectly correlated to the intercept_
    polynomial_extension = PolynomialFeatures(degree=3, include_bias=False)

    # Just to demonstrate that the features generated by both methods are equivalent
    print(np.abs(polynomial_extension.fit_transform(x_data) - x_data_expanded).max())

def run_linear_regression_with_data_engineering(data: DataSetType,
                                                x_data: npt.NDArray[np.float64],
                                                y_data: pd.Series) -> None:
    """
    Second solution to deal with non-linearity is to modify the data. For example, by creating new
    features, derived from the original, using expert knowledge.

    In this case, we generated the data and know that we have a cubic and squared relationship
    between the features (x_data) and target (tagets)
    """
    # exploration(x_data)

    # Transform the input features by feature expansion using Polynomial regression
    title = "Polynomial regression"
    linear_regression = build_pipeline(PolynomialFeatures,
                                       ["feature"],
                                       degree=3,
                                       include_bias=False)
    run_model_with_pipeline(linear_regression.model, data, x_data, y_data, title)

    # k-bins discretizer
    title = "Binned regression"
    linear_regression = build_pipeline(KBinsDiscretizer, ["feature"], n_bins=8)
    run_model_with_pipeline(linear_regression.model, data, x_data, y_data, title)

    # Spline transformer
    title = "Spline regression"
    linear_regression = build_pipeline(SplineTransformer,
                                       ["feature"],
                                       degree=3,
                                       include_bias=False)
    run_model_with_pipeline(linear_regression.model, data, x_data, y_data, title)

    # Nystroem
    title = "Polynomial Nystroem regression"
    linear_regression = build_pipeline(Nystroem,
                                       ["feature"],
                                       kernel="poly",
                                       degree=3,
                                       n_components=5,
                                       random_state=0)
    run_model_with_pipeline(linear_regression.model, data, x_data, y_data, title)


##########
# KERNEL #
##########
def run_svr(data: DataSetType,
            x_data: npt.NDArray[np.float64],
            y_data: pd.Series) -> None:
    """
    Third solution to deal with non-linearity is to make a linear model more expressive by using a
    "kernel". Instead of learning only one weight per feature, a weight is assigned to each sample.
    However, not all samples are used. Some redundant data points of the training set are assigned
    a weight of 0 to avoid any influence of the model's prediction function.

    For larger dataset with n_samples >= 10000, it's often  computationally more efficient to
    perform explicit feature expansion using PolynomialFeatures or other non-linear transformers
    from sklearn such as KBinsDiscretize or SplineTransformer.
    """
    title = "Linear support vector machine"
    run_model(SVR, data, x_data, y_data, title, kernel="linear")

    title = "Polynomial support vector machine"
    run_model(SVR, data, x_data, y_data, title, kernel="poly", degree=3)


######################################
# DATA ENGINEERING ON GENERATED DATA #
######################################
def data_engineering_on_generated_data():
    """
    Try out various technique to deal with non-linear data when using a linear model. Use
    non-linear generated data. To keep it simple, we only work with one feature.
    """
    # Generate some non-linear data
    generated_data = generate_data()

    # Reshape input data
    x_data, y_data = generated_data["x_data"].values.reshape((-1, 1)), generated_data["y_data"]

    # Some visualisation
    scatterplot(generated_data)

    # Run model
    run_linear_regression(generated_data, x_data, y_data)
    run_decision_tree_regression(generated_data, x_data, y_data)
    run_linear_regression_with_data_engineering(generated_data, x_data, y_data)
    run_svr(generated_data, x_data, y_data)


################################
# DATA ENGINEERING ON PENGUINS #
################################
def get_features_generated_by_transformer(linear_regression: LinearRegressionModel,
        x_data: pd.DataFrame,
        y_data: pd.Series):
    """
    Get the features generated by the transfomer.

    With PolynomialFeatures, we have 6 intermediate features in total.
      - The 3 original features: flipper and culmen length as well as culmen depth
      - The 3 generated features that are a combination of the original features:
            . flipper length * culmen length
            . flipper length * culmen depth
            . culmen length * culmen depth

    In general, given p original features, one has p * (p - 1) / 2 interactions and the
    PolynomialFeatures generate as many features. For example,
      - with p = 3 we have 3 * (3 - 1) / 2 = 3 additional interaction features
      - with p = 100 we have 100 * (100 - 1) / 2 = 4950 additional interaction features

    With Nystroem, we have 10 intermediate features in total because the optimal value of
    n_components found and used is 10.

    """
    # Set pipeline steps tou output DataFrame
    linear_regression.set_output()

    # Fit
    linear_regression.model.fit(x_data, y_data)

    # Print output dataframe
    output = linear_regression.model[0].transform(x_data[:5])
    print(f"Features generated by the transformer: {output.columns}")

def validation_curve(linear_regression: LinearRegressionModel,
                     x_data: pd.DataFrame,
                     y_data: pd.Series):
    """
    Validation curve to determine the optimal n_components value.

    - n_components < 10 lead to a underfitting model
    - n_components = 10 is the optimal value
    - n_components >= 50 lead to an overfitting model
    """
    param_name = "columntransformer__nystroem__n_components"
    curve = linear_regression.compute_validation_curve(x_data,
                                                       y_data,
                                                       cv=10,
                                                       scoring="neg_mean_absolute_error",
                                                       score_name="Mean absolute error",
                                                       negate_score=True,
                                                       param_range=np.array([5, 10, 50, 100]),
                                                       param_name=param_name)
    show_validation_curve(curve, xlabel="Validation curve for Nystroem regression")

def data_engineering_on_penguins():
    """
    Try out various technique to deal with non-linear data when using a linear model. Use penguins
    dataset. Contrary to the generated data, we work with multiple features.

    In such multi-dimensional feature space, we can derive new features of the form x1 * x2,
    x2 * x3, etc. Products of features are usually called non-linear or multiplicative intercations
    between features.
    """
    # Load data
    penguins = load_penguins()

    # Build model without any data engineering
    print("\nLinear model without any data engineering.")
    model = LinearRegressionModel.build()
    cross_validation(model, **penguins)

    # Build model with the preprocessing step PolynomialFeatures. The MAE is lower and less spread,
    # with the enriched feeatures. In this case, the additional features are indeed predictive and
    # useful.
    print("\nLinear model with PolynomialFeatures.")
    model = build_pipeline(
        PolynomialFeatures, PENGUIN_FEATURES, degree=2, include_bias=False, interaction_only=True
    )
    get_features_generated_by_transformer(model, **penguins)
    cross_validation(model, **penguins)

    # Build model with the preprocessing step Nystroem. The optimal value for n_components already
    # tuned.
    print("\nLinear model with Nystroem.")
    model = build_pipeline(
        Nystroem, PENGUIN_FEATURES, kernel="poly", degree=2, n_components=10, random_state=0
    )
    # validation_curve(model, *penguins)
    cross_validation(model, **penguins)


############
# ANALYSIS #
############
def run_analysis():
    data_engineering_on_generated_data()
    data_engineering_on_penguins()

if __name__ == "__main__":
    run_analysis()
