from .ames_housing import run_analysis as run_ames_housing
from .fitting_a_sklearn_model_on_numerical_data import \
    run_analysis as run_fitting_a_sklearn_model_on_numerical_data
from .handling_categorical_data import \
    run_analysis as run_handling_categorical_data

__all__ = [
    'run_ames_housing',
    'run_fitting_a_sklearn_model_on_numerical_data',
    'run_handling_categorical_data'
]
