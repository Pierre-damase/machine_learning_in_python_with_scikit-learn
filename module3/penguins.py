from pathlib import Path

import data_handler as dh
import pandas as pd
from config import DataPath, TargetColumn
from model import KNeighborsClassifierModel
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   QuantileTransformer, StandardScaler)
from types_config import DataSetType

PENGUIN_NUMERICAL_FEATURES = [
    "Body Mass (g)", "Flipper Length (mm)", "Culmen Length (mm)"
]


########
# DATA #
########
def load_penguins() -> DataSetType:
    """Load penguin dataset,extract numerical features of interest and drop na."""
    data = dh.load_data_from_csv(
        DataPath.PENGUIN.value
    )[PENGUIN_NUMERICAL_FEATURES + [TargetColumn.PENGUIN]].dropna()
    return pd.DataFrame(data.drop(TargetColumn.PENGUIN, axis=1)), \
        pd.Series(data[TargetColumn.PENGUIN])


#########
# MODEL #
#########
def build_kneighbors_classifier(scaler,
                                columns: list[str],
                                **kwargs) -> KNeighborsClassifierModel:
    """Build a KNeighbors classifier pipeline."""
    return KNeighborsClassifierModel.build_pipeline_with_transformer(
        transformers = [
            (scaler(**kwargs), columns)
        ],
        model = KNeighborsClassifier(n_neighbors=5)
    )

def build_kneighbors_classifier_without_scaler() -> KNeighborsClassifierModel:
    """Build a KNeighbors classifier model without data scaling."""
    return KNeighborsClassifierModel(n_neighbors=5)


####################
# CROSS-VALIDATION #
####################
def cross_validation(model: KNeighborsClassifierModel, x_data: pd.DataFrame, y_data: pd.Series):
    """Cross-validation to evaluate the model performance without any tuning."""
    scores = model.kfold_cross_validate(
        x_data, y_data, nb_fold=10, scoring="balanced_accuracy", return_train_score=True
    )
    model.print_cross_validate(scores)

def run_cv_without_scaler(x_data: pd.DataFrame,
                          y_data: pd.Series,
                          x_train: pd.DataFrame,
                          y_train: pd.Series):
    """Run cross-validation for model without data scaling."""
    model = build_kneighbors_classifier_without_scaler()
    model.start(x_train=x_train, y_train=y_train)
    cross_validation(model, x_data, y_data)


#################
# MANUALLY TUNE #
#################
def manually_tune_model(model: KNeighborsClassifierModel,
                        x_data: pd.DataFrame,
                        y_data: pd.Series,
                        n_neighbors: int):
    """Manually try out some hyperparameters value to tune the model."""
    # Set new parameter value
    print(f"Try out model with n_neighbors = {n_neighbors}")
    model.set_hyperparameters(kneighborsclassifier__n_neighbors=n_neighbors)

    # Perform cross-validation
    cross_validation(model, x_data, y_data)


##################
# AUTOMATED TUNE #
##################
def tune_model(model: KNeighborsClassifierModel,
               x_data: pd.DataFrame,
               y_data: pd.Series,
               x_train: pd.DataFrame,
               y_train: pd.Series):
    """Build various KNeighbors classifier then tune hyperparameter n_neighbors."""
    # 1. Set up the paramater grid used but the grid-search algorithm
    param_grid = {
        "kneighborsclassifier__n_neighbors": [5, 51, 101],
        "columntransformer__standardscaler": [
            None, StandardScaler(), MinMaxScaler(), QuantileTransformer(n_quantiles=100),
            PowerTransformer(method="box-cox")
        ]
    }

    # 2. Tune hyperparameters
    path = Path(*DataPath.HYPERPARAMETER_TUNING.value.parts + ("penguins.csv",))
    model.automated_search_cross_validation(
        GridSearchCV, param_grid, x_data, y_data, x_train, y_train, path, cv=10
    )


############
# ANALYSIS #
############
def run_analysis():
    # 1. Load data
    penguins = load_penguins()

    # 2. Split data into random train and test subsets
    x_train, _, y_train, _ = dh.sklearn_train_test_split(*penguins, test_size=0.8)

    # 3. Build the classifier model
    model = build_kneighbors_classifier(StandardScaler, penguins[0].columns.to_list())

    # 4. Evaluate the model
    # cross_validation(model, *penguins)

    # 5. Try out various value for n_neighbors hyperparameter
    # manually_tune_model(model, *penguins, n_neighbors=51)
    # manually_tune_model(model, *penguins, n_neighbors=101)
    # run_cv_without_scaler(*penguins, x_train, y_train)

    # 6. Tune KNeighbors classifier
    tune_model(model, *penguins, x_train, y_train)

if __name__ == "__main__":
    run_analysis()
