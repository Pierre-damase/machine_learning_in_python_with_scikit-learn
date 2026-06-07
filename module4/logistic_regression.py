import data_handler as dh
import matplotlib.pyplot as plt
import pandas as pd
from config import DataPath, TargetColumn
from model import LogisticRegressionModel
from sklearn.preprocessing import StandardScaler
from types_config import DataSetType

# Data
FEATURES = ["Culmen Length (mm)", "Culmen Depth (mm)"]

########
# DATA #
########
def load_penguins() -> DataSetType:
    """Load penguin dataset, extract features of interest and drop na."""
    # Load data
    penguin = dh.load_data_from_file(DataPath.PENGUIN.value,
                                     columns=FEATURES + [TargetColumn.PENGUIN])

    # Filter data
    mask = penguin[TargetColumn.PENGUIN].isin(["Adelie Penguin (Pygoscelis adeliae)",
                                               "Chinstrap penguin (Pygoscelis antarctica)"])

    # Dropna
    penguin = pd.DataFrame(penguin[mask].dropna())

    return {
        "x_data": dh.get_subset(penguin, columns=FEATURES),
        "y_data": pd.Series(penguin[TargetColumn.PENGUIN])
    }


#################
# VISUALISATION #
#################
def histogram(penguins: DataSetType):
    """Plot an histogram for each species.

    - Culmen length: the probability of a penguin being a Chinstrap is higher when the culm length
      is high

    - Culmen depth: not useful to distinct between an adelie and a chinstrap
    """
    plt.figure()
    data = pd.concat([penguins["x_data"], penguins["y_data"]], axis=1)
    for feature in penguins["x_data"].columns:
        data.groupby(TargetColumn.PENGUIN)[feature].plot.hist(alpha=0.5, legend=True)
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
    histogram(penguins)

    # Split data
    split_data = dh.sklearn_train_test_split(**penguins)
    penguins_test = pd.concat([split_data["x_test"], split_data["y_test"]], axis=1)

    # Build model
    model: LogisticRegressionModel = build_logistic_regression(penguins["x_data"].columns.to_list())

    # Run model
    model.start(**split_data)

    # Decision function boundary with predict as response method
    model.decision_boundary_display(penguins_test,
                                    split_data["x_test"],
                                    response_method="predict",
                                    cmap="RdBu_r",
                                    hue=TargetColumn.PENGUIN.value,
                                    alpha=0.5)
    model.print_weights(FEATURES)

    # Decision function boundary with predict_proba as response method
    model.decision_boundary_display(penguins_test,
                                    split_data["x_test"],
                                    response_method="predict_proba",
                                    cmap="RdBu_r",
                                    hue=TargetColumn.PENGUIN.value,
                                    alpha=0.5)

if __name__ == "__main__":
    run_analysis()
