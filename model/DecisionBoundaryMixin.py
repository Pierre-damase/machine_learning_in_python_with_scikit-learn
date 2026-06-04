import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.axes import Axes
from sklearn.inspection import DecisionBoundaryDisplay

RESPONSE_METHODS = ["predict", "predict_proba"]

CMAP = ["Blues", "RdBu", "RdBu_r", "tab10"]


class DecisionBoundaryMixin:
    def _assert(self, x_data: pd.DataFrame, response_method: str, cmap: str, **kwargs):
        """Check some parameters value."""
        assert len(x_data.columns) == 2, \
            "To be able to plot a decision boundary, two features are expected"
        assert response_method in RESPONSE_METHODS, \
            f"Expected response method are {RESPONSE_METHODS}"
        assert cmap in CMAP, f"Expected cmap are {CMAP}"
        if cmap == "tab10":
            assert "norm" in kwargs.keys(), f"For tab10 cmap, additional param norm is required"

    """Mixin class to define decision boundary display."""
    def decision_boundary_display(self,
                                  data: pd.DataFrame,
                                  x_data: pd.DataFrame,
                                  response_method: str,
                                  hue: str,
                                  cmap: str = "RdBu",
                                  multiclass_colors: list[str] | None = None,
                                  plot_method="contourf",
                                  draw_contour_lines: bool = False,
                                  ax: Axes | None = None,
                                  title: str | None = None,
                                  palette: list[str] | None = None,
                                  **kwargs) -> None:
        """
        Display the decision function boundary. For 2-classes problem, we expect a straight line
        which separted the classes of the target.

        [Only possible for problem with 2 features.]

        The equation of the decision boundary is: coef0 * x0 + coef1 * x1 + b = 0 with
        - x0 the 1st feature and coef0 the associated weight
        - x1 the 2nd feature and coef1 the associated weight
        - b the intercept

        Parameter
        ---------
        data: the whole test dataset, i.e. data and targets together. Either the train or the test
        dataset

        x_data: the features

        reponse_method: {'predict', 'predict_proba'}, predict_proba in order to show the confidence
        on individual classifications. For example, close to the boundary, the confidence is quite
        low and the probability to be either the first class or the second is close to 0.5,
        therefore this region is white.

        cmap: {'Blues', RdBu', 'RdBu_r', 'tab10'} color map use in a simple class problems,
        i.e. a problem with only 2 classes

        plot_method: {'contourf', 'pcolormesh'}, default='contourf'

        ax: specify an ax to plot the decision boundary on it

        multiclass_colors: colors used to plot each class in a multiclass problems

        palette: colors used to plot each data point of a given class

            The two parametes multiclass_colors & palette are required in order to have a match
            bewteen the colors used for the class and the ones used for the data point.

        class_of_interest: the class to be plotted, mostly for multi-classes problem

        **kwargs: additional keyword arguments to be passed to the `plot_method` such as alpha,
        vmin, vmax, norm if cmap is tab10
        """
        # Assert
        self._assert(x_data, response_method, cmap, **kwargs)

        # Get model
        model = getattr(self, "model")

        # Plot decision boundary
        plot = DecisionBoundaryDisplay.from_estimator(model,
                                                      x_data,
                                                      response_method=response_method,
                                                      cmap=cmap,
                                                      multiclass_colors=multiclass_colors,
                                                      plot_method=plot_method,
                                                      ax=ax,
                                                      **kwargs)

        if plot_method == "pcolormesh" and draw_contour_lines:
            DecisionBoundaryDisplay.from_estimator(model,
                                                   x_data,
                                                   response_method=response_method,
                                                   plot_method="contour",
                                                   linestyles="--",
                                                   linewidths=1,
                                                   alpha=0.8,
                                                   levels=[0.5],
                                                   ax=plot.ax_)

        # For multiclass problem  with a single plot, it's important to set hue_order in order to
        # have the same order between the DecisionBoundaryDisplay and the scatterplot
        sns.scatterplot(data=data,
                        x=x_data.columns[0], # 1st feature
                        y=x_data.columns[1], # 2nd feature
                        hue=hue,
                        hue_order=model.classes_ if "classes_" in model.__dict__.keys() else None,
                        palette=palette if palette else ["tab:red", "tab:blue"],
                        edgecolor="white",
                        ax=plot.ax_)
        plt.title(title if title else "Decision boundary of the trained model")

        if not ax:
            plt.show()

    def multi_decision_boundary_display(self,
                                        x_data: pd.DataFrame,
                                        y_data: pd.Series,
                                        response_method: str,
                                        cmap: str) -> None:
        """
        For a K-class problem there is K probability outputs for each data point. Using a single
        plot may be quite difficult to interpret. Therefore, it's quite common to instead produce K
        separate plot, one for each class.

        [Only possible for problem with 2 features.]

        Parameter
        ---------
        x_data: the features

        y_data: the targets

        """
        # Assert
        self._assert(x_data, response_method, cmap)

        # Get model & class
        model = getattr(self, "model")
        classes: list[str] = getattr(self, "model").classes_

        # Plot a decision boundary for each class
        plot, axs = plt.subplots(ncols=len(classes), nrows=1, sharey=True, figsize=(24, 10))
        for i in range(len(classes)):
            axs[i].set_title(f"Class {classes[i]}")
            DecisionBoundaryDisplay.from_estimator(model,
                                                   x_data,
                                                   response_method=response_method,
                                                   class_of_interest=classes[i],
                                                   cmap=cmap,
                                                   ax=axs[i],
                                                   vmin=0,
                                                   vmax=1)

            axs[i].scatter(x=x_data[x_data.columns[0]].loc[y_data == classes[i]],
                           y=x_data[x_data.columns[1]].loc[y_data == classes[i]],
                           marker="o",
                           c="w",
                           edgecolor="k")
            axs[i].set_xlabel(x_data.columns[0])
            axs[i].set_ylabel(x_data.columns[1])
        plot.suptitle("Decision boundary of the trained model", fontsize=14)

        # Add probabilities as colorbar at the bottom
        plt.subplots_adjust(bottom=0.30)
        ax = plt.axes([0.15, 0.14, 0.7, 0.05])
        plt.colorbar(cm.ScalarMappable(cmap=cmap), cax=ax, orientation="horizontal")
        _ = ax.set_title("Predicted class membership probability")

        plt.show()
