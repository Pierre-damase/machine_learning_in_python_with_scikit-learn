import data_handler as dh
import numpy as np
import pandas as pd
from config import DataPath, TargetColumn
from model import HistGradientBoostingRegressorModel, RidgeRegressionModelCV
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from types_config import CvParameters, DataSetType, Tpreprocessor

type ModelClassTypes = (HistGradientBoostingRegressorModel | RidgeRegressionModelCV)

########
# DATA #
########
def load_bike_rides_dataset(only_load_numerical: bool = False) -> DataSetType:
    """
    Load bike rides dataset.
    """

    # Load data
    rides = dh.load_data_from_file(DataPath.BIKE_RIDES.value, drop_na=True)

    # Remove negative acceleration, which correspond to some power created by the braking (not
    # modeled here)
    rides = rides[rides["acceleration"] >= 0]

    # Set up data (the model is simplified):
    # - wind: the power that a cyclist is required to produce to fight wind
    # - rolling_resistance: the power that a cyclist is required to produce to fight the rolling
    # resistance created by the tires on the floor
    # - slope: the power that a cyclist is required to produce to go up a hill if the slope is
    # positive. If the slope is negative the cyclist does not need to produce any power to go
    # forward.
    data = {
        "wind_power": np.power(rides["speed"], 3),
        "rolling_resistance_power": rides["speed"],
        "slope_power": rides["speed"] * np.sin(np.arctan(rides["slope"]))
    }

    return {
        "x_data": pd.concat([pd.DataFrame(data), rides[["acceleration", "speed"]]], axis=1) \
        if only_load_numerical else pd.concat([pd.DataFrame(data), rides], axis=1),
        "y_data": pd.Series(rides[TargetColumn.BIKE_RIDES])
    }

def print_unique_day_from_date(x_data: pd.DataFrame):
    """Print unique day from timestamp colone."""
    time_stamp = pd.to_datetime(x_data["timestamp"])
    print(time_stamp.map(lambda t: t.date()).unique())

#########
# MODEL #
#########
def build_model(model_class: type[ModelClassTypes],
                transformers: list[Tpreprocessor],
                x_data: pd.DataFrame,
                y_data: pd.Series,
                get_weights: bool = False,
                **kwargs) -> None:
    """
    Build model.

    Parameters
    ----------
    model_class: which model to build

    transformers: transformers to add to the pipeline

    x_data: input data

    y_data: targets

    **kwargs: additional model parameters
    """
    # Build model
    model = model_class.build_pipeline(transformers=transformers, **kwargs)

    # Cross-validation
    scores = model.make_cross_validate(x_data,
                                       y_data,
                                       cv_strategy="ShuffleSplit",
                                       cv_params=CvParameters(4, test_size=0.2),
                                       scoring="neg_mean_absolute_error",
                                       return_estimator=True,
                                       return_train_score=True)
    model.print_cross_validate(scores, verbose=True)

    # Get coefficients
    if get_weights:
        model.start(x_train=x_data, y_train=y_data)
        model.print_weights(x_data.columns.to_list())

############
# ANALYSIS #
############
def run_analysis():
    # RidgeCv
    rides = load_bike_rides_dataset(only_load_numerical=True)
    build_model(RidgeRegressionModelCV,
                [StandardScaler()],
                rides["x_data"],
                rides["y_data"],
                get_weights=True)

    # Histogram gradient boosting
    rides = load_bike_rides_dataset()
    print_unique_day_from_date(rides["x_data"])

    rides = load_bike_rides_dataset(only_load_numerical=True)

    transformers = [
        ("passthrough", selector(dtype_exclude="str")),
    ]
    build_model(HistGradientBoostingRegressorModel,
                transformers,
                rides["x_data"],
                rides["y_data"],
                max_iter=1000,
                early_stopping=True)

if __name__ == "__main__":
    run_analysis()
