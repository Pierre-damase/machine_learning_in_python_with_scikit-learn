import data_handler as dh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import GENERATED_DATASET_FEATURES as FEATURES
from config import DataPath, TargetColumn
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from model import LogisticRegressionModel
from sklearn.compose import make_column_selector as selector
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, OneHotEncoder,
                                   PolynomialFeatures, SplineTransformer,
                                   StandardScaler)
from types_config import CvResults, DataSetType
from visualisation import (plot_coefficients_of_linear_model,
                           plot_cross_validation_scores)

TITLES = ["Moons dataset", "Gaussian quantiles dataset", "XOR dataset"]

type PreProcessorType = (KBinsDiscretizer
                     | Nystroem
                     | PolynomialFeatures
                     | SplineTransformer
                     | StandardScaler)

type TransformerType = dict[str, PreProcessorType | bool | int | str]


########
# DATA #
########
def make_non_linear_data() -> list[DataSetType]:
    """Generate non-linear data."""
    return [dh.make_gaussian_quantiles_dataset(), dh.make_moons_dataset(), dh.make_xor_dataset()]

def load_adult_census() -> DataSetType:
    """Load adult census dataset."""
    return dh.load_data_from_file(DataPath.ADULT_CENSUS.value, TargetColumn.ADULT_CENSUS)

#################
# VISUALISATION #
#################
def scatterplot(data: list[DataSetType],
                titles: list[str],
                regressions: list[LogisticRegressionModel] | None = None,
                classifier_title: str | None = None) -> None:
    """
    Simultaneously plot the 3 generated dataset: moons, gauss and xor.

    Parameter
    ---------
    data: list of tuples. The 1st element is the data and the 2nd the target.
    """
    fig, axs = plt.subplots(ncols=3, figsize=(14, 4), constrained_layout=True)
    for i in range(len(data)):
        # Decision boundary
        if regressions is not None:
            decision_boundary(regressions[i].pipeline, data[i][0], axs[i])

        # Scatterplot
        axs[i].scatter(data[i][0][FEATURES[0]],
                       data[i][0][FEATURES[1]],
                       c=data[i][1],
                       cmap=ListedColormap(["tab:red", "tab:blue"]),
                       edgecolor="white",
                       linewidth=1)
        axs[i].set(title=titles[i],
                   xlabel=FEATURES[0],
                   ylabel=FEATURES[1] if i == 0 else None)

    if classifier_title is not None:
        fig.suptitle(classifier_title)
    plt.show()

def decision_boundary(model: Pipeline,
                      x_data: pd.DataFrame,
                      ax: Axes):
    DecisionBoundaryDisplay.from_estimator(model,
                                           x_data,
                                           response_method="predict_proba",
                                           plot_method="pcolormesh",
                                           cmap="RdBu",
                                           alpha=0.8,
                                           vmin=0,
                                           vmax=1,
                                           ax=ax)

    DecisionBoundaryDisplay.from_estimator(model,
                                           x_data,
                                           response_method="predict_proba",
                                           plot_method="contour",
                                           alpha=0.8,
                                           levels=[0.5],
                                           linestyles="--",
                                           linewidth=2,
                                           ax=ax)


#########
# MODEL #
#########
def set_transformer_parameter(transformer: TransformerType) -> TransformerType:
    return {k: v for k, v in transformer.items() if k not in ["columns", "type"]}

def logistic_regression(x_data: pd.DataFrame,
                        y_data: pd.Series,
                        transformers: list[TransformerType],
                        train_model: bool = True,
                        **classifier_args) -> LogisticRegressionModel:
    """
    Perform a logistic regression.

    Parameter
    ---------
    transformers: a list of dictionary.

    {
      "type": the transformer to apply,
      "columns": indicate on which column the transformer must be applied,
      "param_name": a parameter of the transformer to set with a given value,
      ...
    }

    as many parameters as need can be set as well as zero.

    classifier_args: parameter of the model, in this case a logistic regression
    """
    # Build the pipeline steps
    regression = LogisticRegressionModel.build_pipeline_with_transformer(
        transformers=[
            *[
                (
                    ele["type"](**set_transformer_parameter(ele)),
                    ele["columns"]
                )
                if "columns" in ele.keys()
                else ele["type"](**set_transformer_parameter(ele))
                for ele in transformers
            ]
        ],
        model=LogisticRegression(**classifier_args)
    )

    # Train the model
    if train_model:
        regression.start(x_train=x_data, y_train=y_data)

    return regression

def logistic_regression_old(x_data: pd.DataFrame,
                        y_data: pd.Series,
                        transformers: list[TransformerType],
                        train_model: bool = True,
                        **classifier_args) -> LogisticRegressionModel:
    """
    Perform a logistic regression.

    Parameter
    ---------
    transformers: a list of dictionary.

    {
      "type": the transformer to apply,
      "columns": indicate on which column the transformer must be applied,
      "param_name": a parameter of the transformer to set with a given value,
      ...
    }

    as many parameters as need can be set as well as zero.

    classifier_args: parameter of the model, in this case a logistic regression
    """
    # Build the pipeline steps
    regression = LogisticRegressionModel.build_pipeline_with_transformer(
        transformers=[
            *[
                (
                    ele["type"](**{k: v for k, v in ele.items() if k not in ["columns", "type"]}),
                    ele["columns"]
                ) for ele in [ele for ele in transformers if "columns" in ele.keys()]
            ]
        ],
        model=LogisticRegression(**classifier_args)
    )

    # Train the model
    if train_model:
        regression.start(x_train=x_data, y_train=y_data)

    return regression

def logistic_regressions(data: list[DataSetType],
                        transformers: list[TransformerType],
                        **classifier_args) -> list[LogisticRegressionModel]:
    """Perform a logistic regression for each data, a list of tuple of data and targets."""
    regressions = []
    for x_data, y_data in data:
        regressions.append(logistic_regression(x_data, y_data, transformers, **classifier_args))
    return regressions

def run_model_without_feature_engineering(data: list[DataSetType]) -> None:
    """
    Logistic regression without any features engineering, as expected it's not possible to separate
    the two classes with a linear model
    """
    transformers = [{"type": StandardScaler, "columns": FEATURES}]
    scatterplot(data,
                TITLES,
                regressions=logistic_regressions(data, transformers),
                classifier_title="Linear classifier without any features engineering")

def run_model_with_feature_transformation(data: list[DataSetType]) -> None:
    """Logistic regression with a binning or spline transformation."""
    # Build model with the preprocessing step KBinsDiscretizer. The resulting decision boundary is
    # constrained to follow axis-aligned segments, which is very similar to what a decision tree
    # would do. Binning works pretty well for moons and gauss datasets but does not help for the
    # XOR dataset.
    transformers = [{"type": KBinsDiscretizer, "n_bins": 5, "encode": "onehot"}]
    scatterplot(data,
                TITLES,
                regressions=logistic_regressions(data, transformers),
                classifier_title="Binning classifier")

    # Build model with the preprocessing step SplineTransformer. The resulting decision boundary is
    # smooth and while it favors axis-aligned decision rules when extrapolating in low density
    # regions, it can adopt a more curvy decision boundary in the high density regions. However, as
    # for the binning transformation, the model still fail ot separate the data for the XOR dataset
    transformers = [{"type": SplineTransformer, "degree": 3, "n_knots": 5}]
    scatterplot(data,
                TITLES,
                regressions=logistic_regressions(data, transformers),
                classifier_title="Spline classifier")

def run_model_with_polynomial_expansion(data: list[DataSetType]) -> None:
    """
    Logistic regression with polynomial expansion either from PolynomialFeatures or using a kernel
    approximation technique (Nystroem).
    """
    # Build model with the preprocessing step PolynomialFeatures. Data scalling is required in this
    # case. The resulting decision boundary is smoth and can succesfully separate the data on all
    # the three datasets (depending on degree and C hyperparameters).
    transformers = [
        {"type": StandardScaler, "columns": FEATURES},
        {"type": PolynomialFeatures, "degree": 3, "include_bias": False}
    ]
    scatterplot(data,
                TITLES,
                regressions=logistic_regressions(data, transformers, C=10),
                classifier_title="Polynomial classifier")

    # Build model with the preprocessing step Nystroem to perform a polynomial expansion
    # (kernel="poly"). Similar to PolynomialFeatures.
    transformers = [
        {"type": StandardScaler, "columns": FEATURES},
        {"type": Nystroem, "kernel": "poly", "degree": 3, "coef0": 1,
         "n_components": 100}
    ]
    scatterplot(data,
                TITLES,
                regressions=logistic_regressions(data, transformers, C=10),
                classifier_title="Polynomial Nystroem classifier")

def run_model_with_rbf(data: list[DataSetType]) -> None:
    """Logistic regression with a radial basis function (RBF)."""
    # Build model with the preprocessing step Nystroem by using a radial basis function. The
    # resulting decision boundary is smooth and works for the three datasets. In this case, the
    # model tends to be much less confident in its predictions in the low density regions of the
    # feature space
    transformers = [
        {"type": StandardScaler, "columns": FEATURES},
        {"type": Nystroem, "kernel": "rbf", "gamma": 1.0, "n_components": 100}
    ]
    scatterplot(data,
                TITLES,
                regressions=logistic_regressions(data, transformers, C=5),
                classifier_title="RBF Nystroem classifier")

def run_model_with_multi_step_feature_engineering(data: list[DataSetType]) -> CvResults:
    """
    Run the model by combining several feature engineering transformers into a single pipeline to
    blend their respective inductive biases.
    """
    # Combining binning transformation and a kernel approximation
    transformers = [
        {"type": KBinsDiscretizer, "n_bins": 5},
        {"type": Nystroem, "kernel": "rbf", "gamma": 1.0, "n_components": 100}
    ]
    scatterplot(data,
                TITLES,
                regressions=logistic_regressions(data, transformers),
                classifier_title="Binning + RBF Nystroem classifier")

    # Combining spline transformation and a kernel approximation
    transformers = [
        {"type": SplineTransformer, "n_knots": 5},
        {"type": Nystroem, "kernel": "rbf", "gamma": 1.0, "n_components": 100}
    ]
    scatterplot(data,
                TITLES,
                regressions=logistic_regressions(data, transformers),
                classifier_title="Spline + RBF Nystroem classifier")

def logistic_regression_with_cv(data: pd.DataFrame,
                                targets: pd.Series,
                                transformers: list[TransformerType],
                                return_estimator: bool = True,
                                **classifier_args):
    """Perform a logistic regression with a cross-validation step and analysis of the estimator."""
    # 1. Build the model
    regression = logistic_regression(data,
                                     targets,
                                     transformers,
                                     train_model=False,
                                     **classifier_args)

    # 2. Cross validation
    scores = regression.kfold_cross_validate(data,
                                             targets,
                                             nb_fold=10,
                                             return_estimator=return_estimator)
    regression.print_kfold_cross_validation_accuracy(scores)

    # 3. Determine which is the most important feature seen by the model.
    if return_estimator:
        plot_coefficients_of_linear_model(regression.get_coefficients(scores))

    return scores

def run_model_with_numerical_data_only(data: pd.DataFrame, targets: pd.Series) -> CvResults:
    """Run a model with numerical data only."""
    # Extract numerical data
    num_data = dh.get_subset(data, dtypes=[int, float])

    # Set up transfomers
    transformers = [{"type": StandardScaler, "columns": num_data.columns.to_list()}]

    # Run model
    return logistic_regression_with_cv(num_data, targets, transformers)

def run_model_with_all_data(data: pd.DataFrame,
                            targets: pd.Series,
                            polynomial_expansion: bool = False,
                            **classifier_args):
    """
    Run a model with all data.

    A polynomial expansion to deal with non-linear data can be used.
    """
    # Set up transformers
    transformers = [
        {"type": StandardScaler, "columns": selector(dtype_exclude=str)},
        {"type": OneHotEncoder, "columns": selector(dtype_include=str), "handle_unknown": "ignore",
         "min_frequency": 0.01},
    ]
    if polynomial_expansion:
        transformers.append({
            "type": PolynomialFeatures, "degree": 2,
            "interaction_only": True, "include_bias": False
        })

    # Run model
    return logistic_regression_with_cv(data, targets, transformers, **classifier_args)

def cross_validation_comparison(scores: list[CvResults],
                                types: list[str]) -> None:
    """Compare cross-validation results between two different model."""
    # Which model outperformed the most ?
    for i in range(len(scores)-1):
        top = sum(scores[i]['test_score'] < scores[i+1]['test_score'])
        print(f"The 1st model ({types[i]}) is outperformed by the 2nd ({types[i+1]}) "
              f"{top} out of {len(scores[i]['test_score'])}.\n")

    # Plot the test score between the 2 models for each cv fold
    plot_cross_validation_scores([score["test_score"] for score in scores], types)

############
# ANALYSIS #
############
def run_analysis_with_generated_data():
    """
    Use generated data to deal with various non-linear features and use several data engineering
    methods.
    """
    # Load data
    data = make_non_linear_data()

    run_model_without_feature_engineering(data)
    run_model_with_feature_transformation(data)
    run_model_with_polynomial_expansion(data)
    run_model_with_rbf(data)
    run_model_with_multi_step_feature_engineering(data)

def run_analysis_on_adult_census():
    """Use adult census dataset and perform some logistic regression."""
    # 1. Load data
    adult_census = load_adult_census()

    # 2. Build a logistic regression with no feature engineering using only the numerical data
    # In this cas, the most important feature for the logistic regression is the capital gain.
    scores_with_num_data = run_model_with_numerical_data_only(*adult_census)

    # 3. Build a logistic regression with no feature engineering using both non-numerical and
    # numerical data. The most important features are capital gain and a high level of education
    # (masters, prof-school, doctorate)
    scores_with_all_data = run_model_with_all_data(*adult_census, max_iter=200)

    # 4. Build a logistic regression with feature engineering. Extend the previous transformers to
    # add the feature engineering step.
    scores_with_polynomial_expansion = run_model_with_all_data(*adult_census,
                                                               polynomial_expansion=True,
                                                               C=0.01,
                                                               max_iter=200)

    # 5. Check the pertinence of using both non-numerical and numerical data + feature engineering
    # In this case, using both categorical and numerical features is really important but feature
    # engineering does not improve the model performance (the improvement is not significant).
    cross_validation_comparison(
        [scores_with_num_data, scores_with_all_data, scores_with_polynomial_expansion],
        ["Numerical features only", "All features", "Polynomial expansion"]
    )


def run_analysis():
    run_analysis_with_generated_data()
    run_analysis_on_adult_census()


if __name__ == "__main__":
    run_analysis()
