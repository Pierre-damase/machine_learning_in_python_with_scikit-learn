from .ames_housing import run_analysis as run_ames_housing_with_decision_tree
from .classification_tree import run_analysis as run_classification_tree
from .regression_tree import run_analysis as run_regression_tree

__all__ = [
    'run_ames_housing_with_decision_tree',
    'run_classification_tree',
    'run_regression_tree'
]
