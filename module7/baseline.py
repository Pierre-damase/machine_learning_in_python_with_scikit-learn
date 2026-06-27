import data_handler as dh
import pandas as pd
from config import DataPath, TargetColumn
from model import (DecisionTreeRegressorModel, DummyClassifierModel,
                   DummyRegressorModel, LogisticRegressionModel)
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from types_config import (CvParameters, CvResults, DataSetType,
                          ScoringFunctionType, Tpreprocessor)
from visualisation.visualisation import plot_cross_validation_scores

type ModelClassTypes = (DecisionTreeRegressorModel
                        | DummyClassifierModel
                        | DummyRegressorModel
                        | LogisticRegressionModel)


#########
# MODEL #
#########
def build_model(model_class: type[ModelClassTypes],
                x_data: pd.DataFrame,
                y_data: pd.Series,
                scoring: ScoringFunctionType | None = None,
                n_splits: int = 30,
                test_size: float = 0.2,
                transformers: list[Tpreprocessor] | None = None,
                **kwargs) -> CvResults:
    """
    Build a model and perform a shuffle split cross-validation.

    Parameters
    ----------
    model_class: which model to build

    x_data: input data

    y_data: targets

    scoring: scoring funcvtion to use
    """
    # Build model
    if transformers is not None:
        model = model_class.build_pipeline(transformers=transformers, **kwargs)
    else:
        model = model_class.build(**kwargs)

    # Cross-validation
    scores = model.make_cross_validate(x_data,
                                       y_data,
                                       cv_strategy="ShuffleSplit",
                                       cv_params=CvParameters(n_splits,
                                                              test_size=test_size,
                                                              random_state=0),
                                       scoring=scoring)
    model.print_cross_validate(scores)

    return scores


############
# ANALYSIS #
############
def run_regressor_comparisons(housing: DataSetType, scoring: ScoringFunctionType):
    """
    Run model comparisons between a dummy regressor with a mean or median strategy and a decision
    tree.
    """
    scores: list[CvResults] = []
    scores.append(build_model(DecisionTreeRegressorModel, **housing, scoring=scoring))
    scores.append(build_model(DummyRegressorModel, **housing, scoring=scoring))
    scores.append(build_model(DummyRegressorModel, **housing, scoring=scoring, strategy="median"))

    # Plot cv testing scores
    plot_cross_validation_scores(scores,
                                 names=["Decision tree", "Dummy (mean)", "Dummy (median)"],
                                 score_name=f"{scoring} (k$)",
                                 is_regression=True)

def run_regressor():
    """Regresssion task."""
    # Load data
    housing = dh.load_california_dataset()

    # Decision tree testing error (~46k$) is far from perfect but still much better than a dummy
    # regressor with mean (~91k$) or median (~88k$) strategy.
    run_regressor_comparisons(housing, scoring="neg_mean_absolute_error")

    # The R² score for training is always 0, which can be explained by the mathematical definition
    # of this score. However, it's useful to put our model (here the decision tree) in perspective.
    run_regressor_comparisons(housing, scoring="r2")

def run_classifier_comparisons(adult_census: DataSetType):
    """
    Run model comparisons between a dummy classifier with a constant strategy and a logistic
    regression classifier.
    """
    scores: list[CvResults] = []
    transformers = [
        (StandardScaler(), selector(dtype_exclude="str")),
        (OneHotEncoder(handle_unknown="ignore"), selector(dtype_include="str"))
    ]
    scores.append(build_model(LogisticRegressionModel,
                              adult_census["x_data"],
                              adult_census["y_data"],
                              n_splits=10,
                              test_size=0.5,
                              transformers=transformers))

    for cst in ["<=50K", ">50K"]:
        scores.append(build_model(DummyClassifierModel,
                                  adult_census["x_data"],
                                  adult_census["y_data"],
                                  n_splits=10,
                                  test_size=0.5,
                                  constant=cst))

    for strategy in ["stratified", "uniform"]:
        scores.append(build_model(DummyClassifierModel,
                                  adult_census["x_data"],
                                  adult_census["y_data"],
                                  n_splits=10,
                                  test_size=0.5,
                                  strategy=strategy))

    # Plot cv testing scores
    names = ["Logistic regression", "Dummy (<=50K)", "Dummy (>50K)", "Dummy (stratified)",
             "Dummy (uniform)"]
    plot_cross_validation_scores(scores, names=names)

def run_classifier():
    """
    Classification task.

    Logistic regression error (~0.85) is far from perfect but still much better than a dummy
    classifier with:

    - A constant strategy to predict <=50K (~0.76), which is equivalent to most_frequent strategy

    The the following strategies are worse because there is a huge class imbalanced. 75% of the
    training samples are low-income individuals and 25% are high-income individuals.

    - A constant strategy to predict >50K (~0.24)
    - A stratified strategy (~0.6), which will randomly generate predictions by respecting the
    class of the training set distribution.
    - A uniform strategy (~0.5), which will assigns class labels uniformly at random. Therefore,
    on a binary classification problem, the cv accuracy is 50% on average.
    """
    # Load data
    adult_census = dh.load_data_from_file(DataPath.ADULT_CENSUS.value, TargetColumn.ADULT_CENSUS)

    # Build model
    run_classifier_comparisons(adult_census)

def run_analysis():
    """Compare the generalization performance of a model to a minimal baseline model."""
    # Regression task
    #run_regressor()

    # Classification task
    run_classifier()

if __name__ == "__main__":
    pass
