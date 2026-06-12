import data_handler as dh
import numpy as np
import pandas as pd
from config import DataPath
from model import (DecisionTreeRegressorModel,
                   HistGradientBoostingRegressorModel,
                   RandomForestRegressorModel)
from types_config import CvParameters, DataSetType
from visualisation import show_validation_curve

PENGUIN_FEATURES = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)"]
PENGUIN_TARGET = "Body Mass (g)"

type ClassModelTypes = (DecisionTreeRegressorModel
                        | HistGradientBoostingRegressorModel
                        | RandomForestRegressorModel)

########
# DATA #
########
def load_penguins() -> DataSetType:
    """Load penguin dataset with a random shuffle of the data."""
    return dh.load_data_from_file(DataPath.PENGUIN.value,
                                  columns=PENGUIN_FEATURES,
                                  target=PENGUIN_TARGET,
                                  drop_na=True,
                                  shuffle=True)


#########
# MDOEL #
#########
def build_regression(model_class: type[ClassModelTypes],
                     x_data: pd.DataFrame,
                     y_data: pd.Series,
                     **kwargs) -> ClassModelTypes:
    # Build model
    regression = model_class.build(random_state=0, **kwargs)

    # Cross-validation
    scores = regression.make_cross_validate(x_data,
                                            y_data,
                                            cv_strategy="KFold",
                                            cv_params=CvParameters(10),
                                            return_train_score=True)
    regression.print_cross_validate(scores, verbose=True)

    return regression

def decision_tree(x_data: pd.DataFrame,
                  y_data: pd.Series) -> None:
    """Build decision tree model."""
    # Build model
    build_regression(DecisionTreeRegressorModel, x_data, y_data)

def random_forest(x_data: pd.DataFrame,
                  y_data: pd.Series,
                  n_estimators: int,
                  validation_curve: bool = False) -> None:
    """Build random forest model."""
    # Build model
    regression = build_regression(
        RandomForestRegressorModel, x_data, y_data, n_estimators=n_estimators
    )

    # Validation curve
    if validation_curve:
        param_range = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000])
        curve = regression.compute_validation_curve(x_data,
                                                    y_data,
                                                    param_name="n_estimators",
                                                    param_range=param_range,
                                                    scoring="neg_mean_absolute_error",
                                                    score_name="Mean absolute error",
                                                    negate_score=True)
        show_validation_curve(curve, xlabel="n_estimators", point_of_interests=[500])

def hist_gradient_boosting(x_data: pd.DataFrame,
                           y_data: pd.Series) -> None:
    """Build histogram gradient-boosting."""
    # Build model
    regression = build_regression(HistGradientBoostingRegressorModel, x_data, y_data)

    param_range = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000])
    curve = regression.compute_validation_curve(x_data,
                                                y_data,
                                                param_name="max_iter",
                                                param_range=param_range,
                                                scoring="neg_mean_absolute_error",
                                                score_name="Mean absolute error",
                                                negate_score=True)
    show_validation_curve(curve, xlabel="n_estimators", point_of_interests=[5, 10, 100])

############
# ANALYSIS #
############
def run_analysis() -> None:
    """Analysis of the penguin dataset."""
    # Load data
    penguins = load_penguins()

    # The random forest is substantially better thant the single decision tree regressor.
    #decision_tree(**penguins)
    #random_forest(**penguins, n_estimators=100, validation_curve=True)

    # A random forest with 5 decision trees is substantially worse than the random forest model
    # with 100 trees trees
    #random_forest(**penguins, n_estimators=5)

    # Histogram gradient-boosting
    hist_gradient_boosting(**penguins)

if __name__ == "__main__":
    run_analysis()
