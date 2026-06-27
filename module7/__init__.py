from .baseline import run_analysis as run_baseline_comparison
from .bike_rides import run_analysis as run_bike_rides
from .cross_validation_strategies import \
    run_analysis as run_cv_strategies_comparison
from .model_metrics import run_analysis as run_metrics_comparison

__all__ = [
    'run_baseline_comparison',
    'run_bike_rides',
    'run_cv_strategies_comparison',
    'run_metrics_comparison'
]
