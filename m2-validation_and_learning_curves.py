import numpy as np
import pandas as pd

import data_handler as dh
from model import DecisionTreeRegressorModel
from visualisation import (error_distribution, show_learning_curve,
                           show_validation_curve)


#####################
# MODEL PERFORMANCE #
#####################
def overfitting_vs_underfitting(data: pd.DataFrame, targets: pd.Series):
    """
    Compare training and testing error.

    One way to detect overfitting is to perform a cross-validation and check
    the training error. i.e. the capacity of the model to predict the data
    use for the training.

    Same protocol for underfitting with the testing error, i.e. the capacity
    of the model to predict unseen data (in this use case the testing test)
    """
    # 1. Build a decesion tree regressor model
    regressor = DecisionTreeRegressorModel()

    # 2. Cross-validation to compare testing and training error
    scoring:str = "neg_mean_absolute_error"
    scores = regressor.shuffle_split_cross_validate(
        data, targets, 30, scoring, test_size = 0.2, return_train_score=True
    )
    regressor.print_shuffle_split_cross_validation_accuracy(scores)
    error_distribution(scores)


#########################
# HYPERPARAMETER TUNING #
#########################
def validation_curve(data: pd.DataFrame, targets: pd.Series):
    """
    Validation curve help to detect the impact of hyperparameters on the training and testing.

    Result explanation:
    - max_depth < 10: the decision under-fits. The training error and therefore the testing
                    error are both high. The model is too constrained and cannot capture
                    much of the variability.
    - max_depth = 10: the decision tree seems to be optimum with this value of max_depth.
                    The model is flexible enough to capture a fraction of the variability
                    of the target, while not memorizing all the noise.
    - max_depth > 10: the decision tree over-fits. The training error becomes very small,
                    even marginal while the testing error start to increase again. Hence,
                    the model create decisions specifically for noisy samples harming its
                    ability to generalize to testing data.

    Note that for max_depth = 10, the model over-fits a bit as there is a gap between the
    training and testing error. On the other hand, the training error is still far from 0,
    meaning that the model still be too constrained to take into account interesting parts
    of the data. However, increasing the max_depth does not further decrease the testing
    error. Therefore, the testing error is minimal and this is the best compromise.
    """
    # 1. Build a decesion tree regressor model
    regressor = DecisionTreeRegressorModel()

    # 2. Compute validation curve
    curve = regressor.compute_validation_curve(data,
                                               targets,
                                               scoring="neg_mean_absolute_error",
                                               score_name="Mean absolute error",
                                               negate_score=True,
                                               cv=regressor.shuffle_split_cv_generator(30),
                                               param_range=[1, 5, 10, 15, 20, 25])
    show_validation_curve(curve, xlabel="Maximum depth of decision tree")


############################
# TRAINING SET SIZE TUNING #
############################
def learning_curve(data: pd.DataFrame, targets: pd.Series):
    """
    Leaning curve to display impact of various learning set sizes on the
    training set.
    """
    # 1. Build a decesion tree regressor model
    regressor = DecisionTreeRegressorModel()

    # 2. Compute validation curve
    curve = regressor.compute_learning_curve(data,
                                             targets,
                                             cv=regressor.shuffle_split_cv_generator(30),
                                             scoring="neg_mean_absolute_error",
                                             score_name="Mean absolute error",
                                             negate_score=True,
                                             train_sizes=np.linspace(0.1, 1.0, num=5, endpoint=True))
    show_learning_curve(curve)


if __name__ == "__main__":
    # Load data
    housing = dh.load_california_dataset()

    overfitting_vs_underfitting(*housing)

    # Hyperparameter tuning
    validation_curve(*housing)

    # Training set size tuning
    learning_curve(*housing)
