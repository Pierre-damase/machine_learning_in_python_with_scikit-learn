import data_handler as dh
import matplotlib.pyplot as plt
import pandas as pd
from config import GENERATED_DATASET_FEATURES
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from model import LogisticRegressionModel
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, PolynomialFeatures,
                                   SplineTransformer, StandardScaler)
from types_config import DataSetType

TITLES = ["Moons dataset", "Gaussian quantiles dataset", "XOR dataset"]


########
# DATA #
########
def load_data() -> tuple[DataSetType, DataSetType, DataSetType]:
    return dh.make_gaussian_quantiles_dataset(), dh.make_moons_dataset(), dh.make_xor_dataset()

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
        axs[i].scatter(data[i][0][GENERATED_DATASET_FEATURES[0]],
                       data[i][0][GENERATED_DATASET_FEATURES[1]],
                       c=data[i][1],
                       cmap=ListedColormap(["tab:red", "tab:blue"]),
                       edgecolor="white",
                       linewidth=1)
        axs[i].set(title=titles[i],
                   xlabel=GENERATED_DATASET_FEATURES[0],
                   ylabel=GENERATED_DATASET_FEATURES[1] if i == 0 else None)

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
type PreProcessor = (KBinsDiscretizer
                     | Nystroem
                     | PolynomialFeatures
                     | SplineTransformer
                     | StandardScaler)

def logistic_regression(data: list[DataSetType],
                        transformers: list[dict[str, PreProcessor | bool | int | str]],
                        **kwargs) -> list[LogisticRegressionModel]:
    """
    Perform a logistic regression.

    Parameter
    ---------
    transformers: a list of dictionary.

    {
      "type": the transformer to apply,
      "param_name": a parameter of the transformer to set with a given value,
      ...
    }

    as many parameters as need can be set as well as zero.

    kwargs: parameter of the model, in this case a logistic regression
    """
    regressions = []
    for x_data, y_data in data:
        # Build the pipeline steps
        logistic_regression = LogisticRegressionModel.build_pipeline_with_transformer(
            transformers=[
                *[
                    (
                        ele["type"](**{k: v for k, v in ele.items() if k != "type"}),
                        GENERATED_DATASET_FEATURES
                    ) for ele in transformers
                ]
            ],
            model=LogisticRegression(**kwargs)
        )

        # Train the model
        logistic_regression.start(x_train=x_data, y_train=y_data)

        regressions.append(logistic_regression)
    return regressions


############
# ANALYSIS #
############
def run_model_without_feature_engineering(data: list[DataSetType]) -> None:
    """
    Logistic regression without any features engineering, as expected it's not possible to separate
    the two classes with a linear model
    """
    scatterplot(data,
                TITLES,
                regressions=logistic_regression(data, [{"type": StandardScaler}]),
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
                regressions=logistic_regression(data, transformers),
                classifier_title="Binning classifier")

    # Build model with the preprocessing step SplineTransformer. The resulting decision boundary is
    # smooth and while it favors axis-aligned decision rules when extrapolating in low density
    # regions, it can adopt a more curvy decision boundary in the high density regions. However, as
    # for the binning transformation, the model still fail ot separate the data for the XOR dataset
    transformers = [{"type": SplineTransformer, "degree": 3, "n_knots": 5}]
    scatterplot(data,
                TITLES,
                regressions=logistic_regression(data, transformers),
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
        {"type": StandardScaler},
        {"type": PolynomialFeatures, "degree": 3, "include_bias": False}
    ]
    scatterplot(data,
                TITLES,
                regressions=logistic_regression(data, transformers, C=10),
                classifier_title="Polynomial classifier")

    # Build model with the preprocessing step Nystroem to perform a polynomial expansion
    # (kernel="poly"). Similar to PolynomialFeatures.
    transformers = [
        {"type": StandardScaler},
        {"type": Nystroem, "kernel": "poly", "degree": 3, "coef0": 1, "n_components": 100}
    ]
    scatterplot(data,
                TITLES,
                regressions=logistic_regression(data, transformers, C=10),
                classifier_title="Polynomial Nystroem classifier")

def run_model_with_rbf(data: list[DataSetType]) -> None:
    """Logistic regression with a radial basis function (RBF)."""
    # Build model with the preprocessing step Nystroem by using a radial basis function. The
    # resulting decision boundary is smooth and works for the three datasets. In this case, the
    # model tends to be much less confident in its predictions in the low density regions of the
    # feature space
    transformers = [
        {"type": StandardScaler},
        {"type": Nystroem, "kernel": "rbf", "gamma": 1.0, "n_components": 100}
    ]
    scatterplot(data,
                TITLES,
                regressions=logistic_regression(data, transformers, C=5),
                classifier_title="RBF Nystroem classifier")

def run_model_with_multi_step_feature_engineering(data: list[DataSetType]) -> None:
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
                regressions=logistic_regression(data, transformers),
                classifier_title="Binning + RBF Nystroem classifier")

    # Combining spline transformation and a kernel approximation
    transformers = [
        {"type": SplineTransformer, "n_knots": 5},
        {"type": Nystroem, "kernel": "rbf", "gamma": 1.0, "n_components": 100}
    ]
    scatterplot(data,
                TITLES,
                regressions=logistic_regression(data, transformers),
                classifier_title="Spline + RBF Nystroem classifier")

def run_analysis():
    # Load data
    data = [dh.make_moons_dataset(), dh.make_gaussian_quantiles_dataset(), dh.make_xor_dataset()]

    run_model_without_feature_engineering(data)
    run_model_with_feature_transformation(data)
    run_model_with_polynomial_expansion(data)
    run_model_with_rbf(data)
    run_model_with_multi_step_feature_engineering(data)



if __name__ == "__main__":
    run_analysis()
