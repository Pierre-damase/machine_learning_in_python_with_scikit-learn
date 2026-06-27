from module1 import (run_ames_housing,
                     run_fitting_a_sklearn_model_on_numerical_data,
                     run_handling_categorical_data)
from module2 import (run_blood_transfusion, run_cross_validation,
                     run_validation_and_learning_curves)
from module3 import (run_automated_tuning, run_california_housing,
                     run_manual_tuning, run_penguins)
from module4 import (run_ames_housing_linear_model,
                     run_feature_engineering_for_linear_regression,
                     run_feature_engineering_for_logistic_regression,
                     run_linear_regression, run_logistic_regression,
                     run_regularized_regression)
from module5 import (run_ames_housing_with_decision_tree,
                     run_classification_tree, run_regression_tree)
from module6 import (run_boosting, run_bootstrapping, run_ensemble_penguins,
                     run_ensemble_tuning)
from module7 import (run_baseline_comparison, run_bike_rides,
                     run_cv_strategies_comparison, run_metrics_comparison)


def run_module_1():
    """Run methods of the 1st module named The predictive modeling pipeline."""
    run_fitting_a_sklearn_model_on_numerical_data()
    run_handling_categorical_data()
    run_ames_housing()

def run_module_2():
    """Run methods of the 2nd module named Selecting the best model."""
    run_cross_validation()
    run_validation_and_learning_curves()
    run_blood_transfusion()

def run_module_3():
    """Run methods of the 3rd module named Hyperparameter tuning."""
    run_manual_tuning()
    run_automated_tuning()
    run_california_housing()
    run_penguins()

def run_module_4():
    """Run methods of the 4th module named Linear Models."""
    run_linear_regression()
    run_feature_engineering_for_linear_regression()
    run_logistic_regression()
    run_feature_engineering_for_logistic_regression()
    run_regularized_regression()
    run_ames_housing_linear_model()

def run_module_5():
    """Run methods of the 5th module named Decision tree models."""
    run_classification_tree()
    run_regression_tree()
    run_ames_housing_with_decision_tree()

def run_module_6():
    """Run methods of the 6th module named Ensemble of models."""
    run_bootstrapping()
    run_boosting()
    run_ensemble_tuning()
    run_ensemble_penguins()

def run_module_7():
    """Run methods of the 7th module named Evaluating model performance."""
    run_baseline_comparison()
    run_cv_strategies_comparison()
    run_metrics_comparison()
    run_bike_rides()

if __name__ == "__main__":
    run_module_1()
    run_module_2()
    run_module_3()
    run_module_4()
    run_module_5()
    run_module_6()
    run_module_7()
