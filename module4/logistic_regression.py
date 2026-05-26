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
    data, targets = dh.load_data_from_csv(DataPath.PENGUIN.value, TargetColumn.PENGUIN)

    #Filter data
    mask = targets.isin(["Adelie Penguin (Pygoscelis adeliae)",
                         "Chinstrap penguin (Pygoscelis antarctica)"])
    return pd.concat([data[FEATURES], targets], axis=1)[mask].dropna()


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

def decision_boundary_display(logistic_regression: LogisticRegressionModel,
                              penguins_test: pd.DataFrame,
                              x_test: pd.DataFrame,
                              response_method: str):
    """
    Display the decision function boundary. We expect a straight line which separted the classes of
    the target.

    [Only possible for logistic regression problem with 2 features.]

    The equation of the decision boundary is: coef0 * x0 + coef1 * x1 + b = 0 with
    - x0 the culmen length and coef0 the associated weight
    - x1 the culmen depth and coef1 the associated weight
    - b the intercept

    Parameter
    ---------
    logistic_regression: the trained model

    penguins_test: the whole test dataset, i.e. data and targets together.

    x_test: data test

    reponse_method: respond method use, either
    - predict
    - predict_proba: in order to show the confidence on individual classifications. For example,
    close to the boundary, the confidence is quite low and the probability to be either the first
    class or the second is close to 0.5, therefore this region is white.
    """
    DecisionBoundaryDisplay.from_estimator(logistic_regression.model,
                                           x_test,
                                           response_method=response_method,
                                           cmap="RdBu_r",
                                           alpha=0.5)
    sns.scatterplot(data=penguins_test,
                    x=FEATURES[0],
                    y=FEATURES[1],
                    hue=TargetColumn.PENGUIN.value,
                    palette=["tab:red", "tab:blue"])
    plt.title("Decision boundary of the trained LogisticRegression")
    plt.show()


#########
# MODEL #
#########
def build_logistic_regression(columns: list[str]) -> LogisticRegressionModel:
    return LogisticRegressionModel.build_pipeline_with_transformer(
        transformers = [
            (StandardScaler(), columns)
        ],
        model = LogisticRegression()
    )


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
    model = build_logistic_regression(x_train.columns.to_list())

    # Run model
    model.start(x_train, x_test, y_train, y_test)

    # Decision function boundary with predict as response method
    decision_boundary_display(model, penguins_test, x_test, response_method="predict")
    model.print_weights(FEATURES)

    # Decision function boundary with predict_proba as response method
    decision_boundary_display(model, penguins_test, x_test, response_method="predict_proba")

if __name__ == "__main__":
    run_analysis()
