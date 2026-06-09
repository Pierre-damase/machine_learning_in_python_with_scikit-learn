import data_handler as dh
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pandas as pd
from config import DataPath, TargetColumn
from model import DecisionTreeClassifierModel
from types_config import DataSetType

type ClassModelTypes = (DecisionTreeClassifierModel)

PENGUIN_FEATURES = ["Culmen Length (mm)", "Culmen Depth (mm)"]

########
# DATA #
########
def load_penguins() -> DataSetType:
    """Load penguins dataset."""
    return dh.load_data_from_file(DataPath.PENGUIN.value,
                                  target=TargetColumn.PENGUIN,
                                  columns=PENGUIN_FEATURES,
                                  drop_na=True)


#########
# MODEL #
#########
def plot_decision_boundary(model: ClassModelTypes,
                           x_data: pd.DataFrame,
                           y_data: pd.Series,
                           misclassified: pd.DataFrame | None = None,
                           previously_misclassified: pd.DataFrame | None = None):
    """

    Parameters
    ----------
    model: machine learning model

    x_data: input data (features)

    y_data: targets

    misclassified: misclassified samples
    """
    model.decision_boundary_display(pd.concat([x_data, y_data], axis=1),
                                    x_data,
                                    response_method="predict",
                                    cmap="tab10",
                                    norm=getattr(mpl, "colors").Normalize(vmin=-0.5, vmax=8.5),
                                    hue=TargetColumn.PENGUIN.value,
                                    multiclass_colors=["blue", "orange", "green"],
                                    palette=["tab:blue", "tab:orange", "tab:green"],
                                    title="Decision tree predictions",
                                    misclassified=misclassified,
                                    previously_misclassified=previously_misclassified,
                                    alpha=0.5)

def build_model(model_class: type[ClassModelTypes],
                x_data: pd.DataFrame,
                y_data: pd.Series,
                **kwargs) -> ClassModelTypes:
    """
    Build a model.

    Parameters
    ----------
    model_class: which model to initialise

    x_data: input data

    y_data: targets

    **kwargs: additional parameters of the model
    """
    # Build model
    model: ClassModelTypes = model_class.build(**kwargs)

    # Cross-validation
    scores = model.make_cross_validate(x_data, y_data, nb_fold=10)
    model.print_cross_validate(scores)

    return model

def train_model(model: ClassModelTypes,
                x_data: pd.DataFrame,
                y_data: pd.Series,
                sample_weight: npt.NDArray[np.int8] | list[float] | None = None) -> None:
    """Train a model."""
    model.train(x_data, y_data, sample_weight=sample_weight)
    model.print_testing_accuracy(x_data, y_data)


############
# ANALYSIS #
############
def run_decision_tree(penguins: DataSetType) -> None:
    """Run decision tree model."""
    # 1\ Shallow decision tree classifier. The testing error is quite good (~0.93) but some samples
    # have been misclassified by the model.
    classifier = build_model(DecisionTreeClassifierModel, **penguins, max_depth=2, random_state=0)

    # Train the model and get misclassified samples
    train_model(classifier, **penguins)
    misclassified = classifier.get_misclassified(**penguins)
    plot_decision_boundary(classifier, **penguins, misclassified=misclassified)

    # 2\ Train the model by setting the sample_weight parameter at training to give more importance
    # to previously misclassified samples. Which will drastically change the decision function.
    # And add error to previous well classified samples. The testing is worse (~ 0.7).
    train_model(classifier,
                **penguins,
                sample_weight=classifier.get_sample_weight(penguins["y_data"], misclassified))
    plot_decision_boundary(classifier, **penguins, previously_misclassified=misclassified)
    new_misclassified = classifier.get_misclassified(**penguins)

    # 3\ The previous sample weighting is not optimal and add new mistake. Moreover, the testing
    # error is slightly worse. Therefore, we should trust the 1st classifier slightly more than the
    # 2nd. This is the idea behing boosting, a reweighting of the original dataset to help each
    # model (here we just retrain the same model for the example) to focus on specific samples.
    ensemble_weight = classifier.get_ensemble_weight(penguins["y_data"],
                                                     [misclassified, new_misclassified])
    print(ensemble_weight)

def run_ada_boost(penguins: DataSetType) -> None:
    """Run AdaBoost model."""
    print(penguins)

def run_analysis() -> None:
    """Boosting with adaptative boosting (AdaBoost) and grandient-boosting decision tree."""
    # Load data
    penguins = load_penguins()

    # Decision tree
    #run_decision_tree(penguins)

    # AdaBoost
    run_ada_boost(penguins)


if __name__ == "__main__":
    run_analysis()
