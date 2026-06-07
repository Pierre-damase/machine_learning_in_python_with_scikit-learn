import data_handler as dh
import numpy as np
import numpy.typing as npt
import pandas as pd
from config import DataPath, TargetColumn
from model import (DummyClassifierModel, KNeighborsClassifierModel,
                   SupportVectorClassificationModel)
from sklearn.preprocessing import StandardScaler
from types_config import CvResults, DataSetType
from visualisation.visualisation import (
    show_errorbars_for_hyperparameter_tuning, show_learning_curve,
    show_validation_curve)


###################
# DATA MANAGEMENT #
###################
def load_blood_transfusion_dataset() -> DataSetType:
    """
    Load blood transfusion dataset.

    Returns
    -------
    data: blood donation information including
        - R: recency - months since last donation
        - F: frequency - total number of donation
        - M: Monetary - total blood donated in c.c.
        - T: Time - months since first donation

    target: whether the individual donated blood in March 2007
    """
    return dh.load_data_from_file(DataPath.BLOOD_TRANSFUSION.value, TargetColumn.BLOOD_TRANSFUSION)


###########
# SECTION #
###########
def cross_validation(svm: SupportVectorClassificationModel,
                     x_data: pd.DataFrame,
                     y_data: pd.Series) -> None:
    """Evaluate the generalization performance of the model."""
    scores = svm.shuffle_split_cross_validate(x_data, y_data, 10, return_train_score=True)
    svm.print_shuffle_split_cross_validation_accuracy(scores)


#########################
# HYPERPARAMETER TUNING #
#########################
def validation_curve(svm: SupportVectorClassificationModel,
                     x_data: pd.DataFrame,
                     y_data: pd.Series) -> None:
    """
    Use validation curve to tune hyperparameter gamma.

    - gamma > 1: over-fitting
    - gamma ~= 1: best parameter for this model
    - gamma < 1: it's not very clear if the classifier is under-fitting
               but the testing error is worse than for gamma ~= 1
    """
    curve = svm.compute_validation_curve(x_data,
                                         y_data,
                                         cv=svm.shuffle_split_cv_generator(10),
                                         scoring="accuracy",
                                         score_name="Accuracy",
                                         param_range=np.logspace(-3, 2, num=30))
    show_validation_curve(curve, xlabel="Hyperparameter gamma")


############################
# TRAINING SET SIZE TUNING #
############################
def learning_curve(svm: SupportVectorClassificationModel,
                   x_data: pd.DataFrame,
                   y_data: pd.Series) -> None:
    """
    Use learning curve to tune the training set size.

    Adding new samples to the training set does not significantly improve the training and testing
    score. Moreover, it's interesting to notice that the testing score is close to 76% knowing that
    ~ 76% of the sample belong to "not donated". This may mean that our small pipeline is not
    really able to use the input features to improve upon that simplistic baseline and increasing
    the training set size does not help either.

      - This observation may be explained by the input features which are not very informative and
        the classification problem is therefore impossible to solve to a high accuracy
      - On the other hand, using the default hyperparameter value of the SVC class to try out the
        training set size is sub-optimal
      - An, it's also possible that SVC is sub-optimal to solve this problem
    """
    curve = svm.compute_learning_curve(x_data,
                                       y_data,
                                       cv=svm.shuffle_split_cv_generator(10),
                                       scoring="accuracy",
                                       score_name="Accuracy",
                                       train_sizes=np.linspace(0.1,
                                                               1.0,
                                                               num=10,
                                                               endpoint=True))
    show_learning_curve(curve)


#######
# SVM #
#######
def blood_transfusion_svm(x_data: pd.DataFrame, y_data: pd.Series) -> None:
    """Set up SVM classifier for blood transfusion dataset."""
    # 1. Set up a support vector machine classifier (SVM)
    svm = SupportVectorClassificationModel.build_pipeline([StandardScaler()], kernel="rbf")

    # 2. Cross-validation
    cross_validation(svm, x_data, y_data)

    # 3. Hyperparameter tuning
    validation_curve(svm, x_data, y_data)

    # 4. Training set size tuning
    learning_curve(svm, x_data, y_data)

    # 5. Training set size tuning with gamma = 1, the optimal value of the hyperparameter
    #    This hyperparameter + a larger dataset help to improve the testing accuracy (up to 80%)
    #    It's remain quite low.
    svm = SupportVectorClassificationModel.build_pipeline([StandardScaler()],
                                                          kernel="rbf",
                                                          gamma=1.0)
    learning_curve(svm, x_data, y_data)


####################
# DUMMY CLASSIFIER #
####################
def blood_transfusion_dummy_classifier(
        x_data: pd.DataFrame,
        y_data: pd.Series) -> None:
    """Set up dummy classifier for blood transfusion dataset."""
    # 1. Set up a dummy classifier model
    model = DummyClassifierModel.build(strategy="most_frequent")

    # 2. KFold cross-validation
    scores = model.kfold_cross_validate(x_data, y_data, 10)
    model.print_kfold_cross_validation_accuracy(scores)

    scores = model.kfold_cross_validate(x_data, y_data, 10, scoring="balanced_accuracy")
    model.print_kfold_cross_validation_accuracy(scores)


#########################
# KNEIGHBORS CLASSIFIER #
#########################
def blood_transfusion_knearest_classifier(
        x_data: pd.DataFrame,
        y_data: pd.Series,
        nb_fold: int = 10,
        n_neighbors: int = 1,
        print_score: bool = False) -> CvResults:
    """Set up k-nearest classifier for blood transfusion dataset."""
    # 1. Set up a k-nearest classifier
    model = KNeighborsClassifierModel.build_pipeline([StandardScaler()], n_neighbors=n_neighbors)

    # 2. KFold cross-validation
    scores = model.kfold_cross_validate(
        x_data, y_data, nb_fold, scoring="balanced_accuracy", return_train_score=True
    )
    if print_score:
        model.print_kfold_cross_validation_accuracy(scores)

    return scores

def manually_tune_knearest_classifier(x_data: pd.DataFrame, y_data: pd.Series) -> None:
    """
    Manually tune neighbors number for k-nearest classifier.

    Result identical than ValidationCurveDisplay.from_estimator
    """
    train_scores: dict[str, list[float]] = {"mean": [], "std": []}
    test_scores: dict[str, list[float]] = {"mean": [], "std": []}
    x: list[float] = []

    # Perform k-nearest classifier with various k-fold value
    for i in np.array([1, 2, 5, 10, 20, 50, 100, 200, 250]):
        scores = blood_transfusion_knearest_classifier(x_data, y_data, n_neighbors=i)
        # Train
        if "train_score" in scores.keys():
            train_scores["mean"].append(scores["train_score"].mean())
            train_scores["std"].append(scores["train_score"].std())

        # Test
        test_scores["mean"].append(scores["test_score"].mean())
        test_scores["std"].append(scores["test_score"].std())

        # x
        x.append(i)

    # Show scores
    show_errorbars_for_hyperparameter_tuning(test_scores,
                                             x,
                                             xlabel="N neighbors",
                                             train_scores=train_scores)

def tune_knearest_classifier(x_data: pd.DataFrame, y_data: pd.Series):
    """Tune neighbors number for k-nearest classifier."""
    # 1. Build a k-nearest classifier
    model = KNeighborsClassifierModel.build_pipeline([StandardScaler()])

    # 2.
    param_range = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
    curve = model.compute_validation_curve(x_data,
                                           y_data,
                                           scoring="balanced_accuracy",
                                           score_name="Balanced accuracy",
                                           param_name="kneighborsclassifier__n_neighbors",
                                           param_range=param_range)
    show_validation_curve(curve, xlabel="Hyperparameter gamma")


############
# ANALYSIS #
############
def run_analysis():
    # Load data and targets
    blood_transfusion = load_blood_transfusion_dataset()

    # Try out SVM classifier with blood transfusion dataset
    blood_transfusion_svm(**blood_transfusion)

    # Try out dummy classifier with blood transfusion dataset
    blood_transfusion_dummy_classifier(**blood_transfusion)
    blood_transfusion_knearest_classifier(**blood_transfusion, print_score=True)

    # Tune neighbors number for k-nearest classifier
    manually_tune_knearest_classifier(**blood_transfusion)
    tune_knearest_classifier(**blood_transfusion)

if __name__ == "__main__":
    run_analysis()
