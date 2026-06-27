import data_handler as dh
import numpy as np
import numpy.typing as npt
import pandas as pd
from model import (DecisionTreeRegressorModel, LogisticRegressionModel,
                   SupportVectorClassificationModel)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from types_config import (AcceptPreprocessorType, CvParameters, CvStrategy,
                          SearchCvParameters, SearchOuterCv, Tpreprocessor)

type ModelClassTypes = (DecisionTreeRegressorModel | LogisticRegressionModel)


########
# DATA #
########
def sample_grouping(y_data: pd.Series) -> npt.NDArray[np.int64]:
    """
    Some patterns in the data. Tehrefore set groups in order to group data during the
    cross-validation.
    """
    # Set boundaries from pattern  in the target and information about the dataframe
    boundaries = [0, 130, 256, 386, 516, 646, 776, 915, 1029, 1157, 1287, 1415, 1545, 1667, 1797]

    # Set groups
    groups = np.zeros_like(y_data)
    for i in range(len(boundaries[:-1])):
        groups[boundaries[i]:boundaries[i+1]] = i
    return groups


#########
# MODEL #
#########
def build_model(model_class: type[ModelClassTypes],
                x_data: pd.DataFrame,
                y_data: pd.Series,
                cv_params: CvParameters = CvParameters(n_splits=5),
                cv_strategy: CvStrategy = "KFold",
                transformers: list[Tpreprocessor] | None = None,
                max_iter: int = 100,
                verbose: bool = False,
                **kwargs) -> None:
    """
    Build a model.

    Parameters
    ----------
    **kwargs: additional parameters for the cross-validation.
    """
    # Build model
    if transformers is not None:
        model = model_class.build_pipeline(transformers, max_iter=max_iter)
    else:
        model = model_class.build()

    # Cross-validation
    scores = model.make_cross_validate(
        x_data, y_data, cv_params=cv_params, cv_strategy=cv_strategy, **kwargs
    )
    model.print_cross_validate(scores, verbose=verbose)


############
# ANALYSIS #
############
def run_stratified_kfold_cv():
    """Perform k-fold cross-validation and stratified k-fold cross-validation."""
    # Load data
    iris = dh.load_iris_dataset()

    transformers: list[AcceptPreprocessorType] = [StandardScaler()]
    cv_params: CvParameters = CvParameters(3)

    # 1\ k-fold cross-validation without shuffling. Iris dataset is ordered, thus the score is 0.
    build_model(LogisticRegressionModel, transformers=transformers, cv_params=cv_params, **iris)

    # 2\ k-fold cross-validation with shuffling. The score is much higher and closer to what we
    # would expect. However, with shuffling neither the training and testing sets will have the
    # same class frequencies as the original dataset accross each fold.
    cv_params = cv_params._replace(shuffle=True)
    build_model(LogisticRegressionModel, transformers=transformers, cv_params=cv_params, **iris)

    # 3\ Stratified k-fold cross-validation. In this case, the training and testing class
    # frequencies is really close to the original dataset accross each fold. However, there is a
    # small difference probably due to the small number of samples in the iris dataset.
    cv_params = cv_params._replace(shuffle=False)
    build_model(LogisticRegressionModel,
                transformers=transformers,
                cv_params=cv_params,
                cv_strategy="StratifiedKFold",
                **iris)

def run_sample_grouping_cv():
    """Perform sample grouping cross-validation."""
    # Load data
    digit = dh.load_digit_dataset()

    transformers: list[AcceptPreprocessorType] = [MinMaxScaler()]
    cv_params: CvParameters = CvParameters(3)

    # k-fold cross-validation without shuffling.
    build_model(LogisticRegressionModel,
                transformers=transformers,
                cv_params=cv_params,
                **digit,
                max_iter=1000)

    # k-fold cross-validation with shuffling. The average test score is higher, which means some
    # specific fold leads to a lower score (default k-fold) and shuffling breaks the underlying
    # structure of the data.
    cv_params = cv_params._replace(shuffle=True)
    build_model(LogisticRegressionModel,
                transformers=transformers,
                cv_params=cv_params,
                **digit,
                max_iter=1000)

    # 4\ Sample grouping cross-validation.
    cv_params = cv_params._replace(shuffle=False)
    build_model(LogisticRegressionModel,
                transformers=transformers,
                cv_strategy="GroupKFold",
                cv_params=cv_params,
                **digit,
                groups=sample_grouping(digit["y_data"]))

def run_non_iid_data_cv():
    """
    Perform cross-validation for non i.i.d data. Most of the time data are i.i.d, meaning that the
    generative process does not have any memory of past samples to generate new samples. This
    assumption is usually violated in time series, where each sample can be influenced by previous
    ones in an inherently ordered sequence.
    """
    # Load data
    financial = dh.load_financial_dataset(target="CVX")

    # Shuffle split cross-validation. Surprisingly, we get outstanding generalization performance.
    build_model(DecisionTreeRegressorModel,
                cv_strategy="ShuffleSplit",
                cv_params=CvParameters(10, test_size=0.2),
                **financial)

    # Time series detection. With the shuffled data, the predictions are close to the actual
    # values. However, with unshuffled data the model does not work at all. And a clear separation
    # bewteen training and testing value is visible which is a good indicator of time serie.
    model = DecisionTreeRegressorModel.build()
    model.time_series_detection(**financial, scoring="r2")

    # Leave one group out and time split cross-validation. For both case one split is really bad.
    # Overall the prediction is not good at all.
    for scoring in ["LeaveOneGroupOut", "TimeSeriesSplit"]:
        groups = financial["x_data"].columns.to_list() + [financial["y_data"].name] \
            if scoring == "TimeSeriesSplit" else None
        build_model(DecisionTreeRegressorModel,
                    cv_strategy="TimeSeriesSplit",
                    verbose=True,
                    **financial,
                    scoring="r2",
                    groups=groups)

def run_nested_cross_validation():
    """
    Cross-validation can be used both for hyperparameter tuning and for estimating the
    generalization performance of a model. However, it's required to perform two cv:
    - One inner cross-validation to tune the hyperparameters.
    - One outer cross-validation to assess the generalization performance of the tuned model, i.e.
    the model with the best hyperparameters.
    """
    # Load data
    breast_cancer = dh.load_breast_cancer_dataset()
    split_data = dh.sklearn_train_test_split(**breast_cancer)

    # Build model
    model = SupportVectorClassificationModel.build()

    # Simple parameter tuning with a grid-search cross-validation
    model.automated_search_cv(search_class=GridSearchCV,
                              search_params=SearchCvParameters(5),
                              parameters={"C": [0.1, 1, 10], "gamma": [0.01, 0.1]},
                              x_train=split_data["x_train"],
                              y_train=split_data["y_train"],
                              search_outer_cv=SearchOuterCv(**breast_cancer))

def run_analysis():
    """Try various cross-validation strategies."""
    run_stratified_kfold_cv()
    run_sample_grouping_cv()
    run_non_iid_data_cv()
    run_nested_cross_validation()

if __name__ == "__main__":
    pass
