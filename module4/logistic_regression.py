import data_handler as dh
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import DataPath, TargetColumn
from model import LogisticRegressionModel
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Data
FEATURES = ["Culmen Length (mm)", "Culmen Depth (mm)"]

########
# DATA #
########
def load_penguins() -> pd.DataFrame:
    """Load penguin dataset, extract features of interest and drop na."""
    # Load data
    data, targets = dh.load_data_from_file(DataPath.PENGUIN.value, TargetColumn.PENGUIN)

    #Filter data
    mask = targets.isin(["Adelie Penguin (Pygoscelis adeliae)",
                         "Chinstrap penguin (Pygoscelis antarctica)"])
    return pd.DataFrame(pd.concat([data[FEATURES], targets], axis=1)[mask].dropna())


#################
# VISUALISATION #
#################
def histogram(penguins: pd.DataFrame):
    """Plot an histogram for each species.

    - Culmen length: the probability of a penguin being a Chinstrap is higher when the culm length
      is high

    - Culmen depth: not useful to distinct between an adelie and a chinstrap
    """
    plt.figure()
    for feature in penguins[FEATURES].columns:
        penguins.groupby(TargetColumn.PENGUIN)[feature].plot.hist(alpha=0.5, legend=True)
        plt.xlabel(feature)
        plt.show()


#########
# MODEL #
#########
def build_logistic_regression(columns: list[str]) -> LogisticRegressionModel:
    return LogisticRegressionModel.build_pipeline([(StandardScaler(), columns)])


############
# ANALYSIS #
############
def run_analysis():
    # Load data
    penguins = load_penguins()

    # Data visualisation
    # histogram(penguins)

    # Split data
    x_train, x_test, y_train, y_test = dh.sklearn_train_test_split(penguins[FEATURES],
                                                                   penguins[TargetColumn.PENGUIN])
    penguins_test = pd.concat([x_test, y_test], axis=1)

    # Build model
    model: LogisticRegressionModel = build_logistic_regression(x_train.columns.to_list())

    # Run model
    model.start(x_train, x_test, y_train, y_test)

    # Decision function boundary with predict as response method
    model.decision_boundary_display(penguins_test,
                                    x_test,
                                    response_method="predict",
                                    cmap="RdBu_r",
                                    hue=TargetColumn.PENGUIN.value,
                                    alpha=0.5)
    model.print_weights(FEATURES)

    # Decision function boundary with predict_proba as response method
    model.decision_boundary_display(penguins_test,
                                    x_test,
                                    response_method="predict_proba",
                                    cmap="RdBu_r",
                                    hue=TargetColumn.PENGUIN.value,
                                    alpha=0.5)

if __name__ == "__main__":
    run_analysis()
