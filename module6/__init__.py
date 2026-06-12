from .boosting import run_analysis as run_boosting
from .bootstrapping import run_analysis as run_bootstrapping
from .ensemble_penguins import run_analysis as run_ensemble_penguins
from .ensemble_tuning import run_analysis as run_ensemble_tuning

__all__ = [
    'run_boosting',
    'run_bootstrapping',
    'run_ensemble_penguins',
    'run_ensemble_tuning'
]
