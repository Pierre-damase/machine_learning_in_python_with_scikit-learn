from config import DataPath
from model import LinearRegressionModel
import data_handler as dh

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns


# Data
FEATURE = "Flipper Length (mm)"

# Target
TARGET = "Body Mass (g)"


########
# DATA #
########
def load_penguins() -> pd.DataFrame:
    """Load penguin dataset, extract features of interest and drop na."""
    data, targets = dh.load_data_from_csv(DataPath.PENGUIN.value, TARGET)
    return pd.concat([data[FEATURE], targets], axis=1).dropna()


#############################
# MANUAL LINEAR REGRESSION  #
#############################
def scatterplot(penguins: pd.DataFrame,
                x_range: npt.NDArray[np.float64],
                list_y_predicted: list[npt.NDArray[np.float64]],
                **kwargs) -> None:
    """Plot a scatterplot to visualize data."""
    # Build graphic
    ax = sns.scatterplot(penguins, x=FEATURE, y=TARGET, color="black", alpha=0.5)

    # Set up label
    label = "{0:.2f} (g/mm) * flipper length + {1:.2f} (g)"

    # Add linear prediction(s)
    for i in range(len(list_y_predicted)):
        ax.plot(x_range,
                list_y_predicted[i],
                label=label.format(kwargs["weights"][i], kwargs["intercepts"][i]))

    ax.set_title("Body mass as a function of the flipper length")
    plt.legend()
    plt.show()


def manual_linear_model_definition(flipper_length: npt.NDArray[np.float64],
                                   weight_flipper_length: int,
                                   intercept_coefficient) -> npt.NDArray[np.float64]:
    """Build a simple linear model of the form y = ax + b with

    - y the predicted body mass
    - a the flipper length
    - x the weight applied to the flipper length in order to make the inference. Basically, in this
    case the coefficient unit is g/mm which means for each additional millimeter in flipper length
    the body weight predicted increases by x (for positive x) / decreases by x (for negative x)
    - b the intercept coefficient
    """
    return flipper_length * weight_flipper_length + intercept_coefficient

def goodness_fit_measure(expected_values: npt.NDArray[np.int64],
                         predicted_values: list[npt.NDArray[np.float64]]) \
                         -> npt.NDArray[np.float64]:
    """Measure the goodness of fit of each linear model to be able to select the best one."""
    return [np.mean(np.abs(expected_values - ele)) for ele in predicted_values]


def manual_linear_regression(penguins: pd.DataFrame,
                             data: pd.DataFrame,
                             targets: pd.Series) -> None:
    """
    Perform manually a linear regression.

    There is almost a linear relationship between the body mass of the penguin and its flipper
    length. Hence, we coulc come up with a single formula, where given a flipper length we could
    compute the body mass using a linear relationship of the form y = ax + b where a and b are the
    2 parameters of the model.

    Parameter
    ---------
    penguins: the whole dataset with data and target concatenated

    data: dataset with the features

    targets: dataset witht the targets
    """
    x_range = np.linspace(data.min(), data.max(), num=300)

    # Linear relationship
    y_predicted = manual_linear_model_definition(x_range, 45, -5000)
    scatterplot(penguins, x_range, [y_predicted], weights=[45], intercepts=[-5000])

    y_predicted = manual_linear_model_definition(x_range, -45, 13000)
    scatterplot(penguins, x_range, [y_predicted], weights=[-45], intercepts=[13000])

    # Here, we use a brute-force search in order to find the best parameter. In reality, this
    # can be found by solving an equation using scikit-learn.
    weights, intercepts = [-40, 45, 90], [15000, -5000, -14000]
    list_y_predicted = []
    for i in range(len(weights)):
        list_y_predicted.append(manual_linear_model_definition(x_range, weights[i], intercepts[i]))
    scatterplot(penguins, x_range, list_y_predicted, weights=weights, intercepts=intercepts)

    # Compute scores
    scores = goodness_fit_measure(targets.to_numpy(), list_y_predicted)
    for i in range(len(weights)):
        print(f"For linear model {weights[i]:.2f} (g/mm) * flipper length + {intercepts[i]:.2f} "
              f"(g) the mean error is {scores[i]}")


###############################
# AUTOMATIC LINEAR REGRESSION #
###############################
def automatic_linear_regression(penguins: pd.DataFrame,
                                x_data: pd.DataFrame,
                                y_data: pd.Series) -> None:
    """Perform a linear regression using scikit-learn"""
    x_range = np.linspace(data.min(), data.max(), num=342)

    # Build Model
    linear_regression = LinearRegressionModel()
    linear_regression.start(x_train=x_data, y_train=y_data)

    # Check optimum parameters
    coef_a, intercept = linear_regression.model.coef_[0], linear_regression.model.intercept_
    print(f"The optimal linear equation is y = {coef_a:.3f}x + {intercept:.3f}.")
    scatterplot(penguins,
                np.linspace(x_data.min(), x_data.max(), num=342),
                [coef_a * x_range + intercept],
                weights=[coef_a],
                intercepts=[intercept])

    # Check the mean squared error of the optimal linear regression model. Moreover, this
    # kind of model monimize the mean squared error which would be higher with any other
    #set of parameters.
    linear_regression.print_mean_squared_error(x_data, y_data)

    # Compute the mean absolute error which is more intuitive
    linear_regression.print_mean_absolute_error(x_data, y_data)


if __name__ == "__main__":
    # Load data
    penguins = load_penguins()
    data, targets = penguins.drop(columns=TARGET), pd.Series(penguins[TARGET])

    # manual_linear_regression(penguins, data, targets)
    automatic_linear_regression(penguins, data, targets)
