from pathlib import Path

import data_handler as dh
import numpy as np
import pandas as pd
from config import DataPath
from model import KNeighborsRegressorModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from visualisation import show_parallel_coordinates_for_hyperparameter_tuning


#########
# MODEL #
#########
def build_kneighbors_regressor(columns: list[str]) -> KNeighborsRegressorModel:
    return KNeighborsRegressorModel.build_pipeline_with_transformer(
        transformers = [
            (StandardScaler(), columns)
        ],
        model = KNeighborsRegressor()
    )


#########################
# HYPERPARAMETER TUNING #
#########################
def randomized_search_tuning(model: KNeighborsRegressorModel,
                             x_data: pd.DataFrame,
                             y_data: pd.Series,
                             x_train: pd.DataFrame,
                             y_train: pd.Series,
                             path: Path) -> None:
    """
    Hyperparameter tuning by randomized-search.

    - n_neighbors of the KNeighborsRegressor
    - with_mean fo the StandardScaler
    - with_std of the StandarSacler
    """
    # Set up the parameter distribution used by the randomized-search algorithm
    param_dist = {
        'kneighborsregressor__n_neighbors': np.logspace(0, 3, num=10).astype(np.int32),
        'columntransformer__standardscaler__with_mean': [True, False],
        'columntransformer__standardscaler__with_std': [True, False]
    }

    # Tune hyperparameter
    model.automated_search_cross_validation(RandomizedSearchCV,
                                            param_dist,
                                            x_data,
                                            y_data,
                                            x_train,
                                            y_train,
                                            path=path,
                                            cv=5,
                                            n_iter=20,
                                            scoring="neg_mean_absolute_error")


############
# ANALYSIS #
############
def run_analysis():
    # 1. Load data
    housing = dh.load_california_dataset()

    # 2. Split data into random train and test subsets
    x_train, _, y_train, _ = dh.sklearn_train_test_split(*housing, test_size=0.8)

    # 3. Build the regressor model
    model = build_kneighbors_regressor(x_train.columns.to_list())

    # 4. Tuning using randomized-search
    file_name = "randomized_search_california_housing.csv"
    path = Path(*DataPath.HYPERPARAMETER_TUNING.value.parts + (file_name,))
    # randomized_search_tuning(model, *housing, x_train, y_train, path)
    show_parallel_coordinates_for_hyperparameter_tuning(pd.read_csv(path))

if __name__ == "__main__":
    run_analysis()
