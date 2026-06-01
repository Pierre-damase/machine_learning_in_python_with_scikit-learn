from .automated_tuning import run_analysis as run_automated_tuning
from .california_housing import run_analysis as run_california_housing
from .manual_tuning import run_analysis as run_manual_tuning
from .penguins import run_analysis as run_penguins

__all__ = [
    'run_automated_tuning',
    'run_california_housing',
    'run_manual_tuning',
    'run_penguins'
]
