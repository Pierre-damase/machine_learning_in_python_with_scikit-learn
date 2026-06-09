import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.tree import plot_tree


class TreeMixin:
    """
    Mixin class to define some methods specific to tree models such as plot the corresponding
    decision tree.
    """
    def plot_decision_tree(self, features: list[str]):
        """Plot a decision tree."""
        # Get model
        model = getattr(self, "model")

        # Plot
        _, ax = plt.subplots(figsize=(18, 16))
        plot_tree(model,
                  feature_names=features,
                  class_names=model.classes_.tolist() if "classes_" in model.__dict__.keys() \
                  else None,
                  impurity=False,
                  ax=ax)
        plt.show()

    def plot_decision_tree_predictions(self,
                                       x_train: pd.DataFrame,
                                       x_test: pd.DataFrame,
                                       y_train: pd.Series,
                                       predictions: npt.NDArray[np.float64],
                                       title: str):
        """
        Plot the predictions of each individual decision tree.

        Parameters
        ----------
        x_train: training data

        x_test: testing data

        y_train: training targets
        """
        # Get model & estimators
        model = getattr(self, "model")
        estimators = getattr(model, "estimators_")

        # Convert testing data into numpy array to avoid a warning in scikit-lean
        x_test_np = x_test.to_numpy()

        # Set up plot parameter
        nb_features = len(x_train.columns)
        nb_estimators = len(estimators)

        # Make a plot for each input feature
        _, axs = plt.subplots(ncols=nb_features, figsize=(20, 12))
        for i in range(nb_features):
            column = x_train.columns.values[i]

            # For dataset with a single feature, the subplots method return a single Axes object
            if nb_features == 1:
                axs: list[Axes] = [axs]

            # Predict testing target and plot it only for small number of decision tree
            if nb_estimators < 50:
                for j in range(nb_estimators):
                    y_predicted = estimators[j].predict(x_test_np)
                    axs[i].plot(x_test[column],
                                y_predicted,
                                linestyle="--",
                                alpha=0.5,
                                color="tab:gray" if nb_estimators > 10 else None,
                                label=f"Tree #{j}" if nb_estimators <= 10 \
                                else "Predictions of individual trees" if j == 0 \
                                else None)

            # Scatterplot
            ax = sns.scatterplot(x=x_train[column], y=y_train, color="black", alpha=0.5, ax=axs[i])
            ax.plot(x_test[column], predictions, label="Prediction of ensemble", color="tab:red")
            ax.legend()
        plt.title(title)
        plt.show()

