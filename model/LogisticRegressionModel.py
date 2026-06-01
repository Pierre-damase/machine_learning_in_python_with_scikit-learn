import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from types_config import Tlinearmodel

from .LinearModel import LinearModel

RESPONSE_METHODS = ["predict", "predict_proba"]

CMAP = ["RdBu", "RdBu_r"]

class LogisticRegressionModel(LinearModel[LogisticRegression, Tlinearmodel]):
    """
    Logistic regression model.

    [To predict discrete target]

    C is the hyperparameter of this kind of model.
    """
    _estimator_class = LogisticRegression

    def decision_boundary_display(self,
                                  data: pd.DataFrame,
                                  x_data: pd.DataFrame,
                                  response_method: str,
                                  hue: str,
                                  cmap: str = "RdBu",
                                  multiclass_colors: list[str] |None = None,
                                  plot_method="contourf",
                                  draw_contour_lines: bool = False,
                                  ax: Axes | None = None,
                                  title: str | None = None,
                                  palette: list[str] |None = None,
                                  **kwargs):
        """
        Display the decision function boundary. We expect a straight line which separted the
        classes of the target.

        [Only possible for logistic regression problem with 2 features.]

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

        cmap: {'RdBu', 'RdBu_r'} color map use in a simple class problems, i.e. a problem with only
        2 classes

        multiclass_colors: colors used to plot each class in a multiclass problems

        plot_method: {'contourf', 'pcolormesh'}, default='contourf'

        ax: specify an ax to plot the decision boundary on it

        **kwargs: additional keyword arguments to be passed to the `plot_method` such as alpha,
        vmin, vmax
        """
        assert len(x_data.columns) == 2, \
            "To be able to plot a decision boundary, two features are expected"
        assert response_method in RESPONSE_METHODS, \
            f"Expected response method are {RESPONSE_METHODS}"
        assert cmap in CMAP, f"Expected cmap are {CMAP}"

        print(cmap)
        plot = DecisionBoundaryDisplay.from_estimator(self.model,
                                                      x_data,
                                                      response_method=response_method,
                                                      cmap=cmap,
                                                      multiclass_colors=multiclass_colors,
                                                      plot_method=plot_method,
                                                      ax=ax,
                                                      **kwargs)

        if plot_method == "pcolormesh" and draw_contour_lines:
            DecisionBoundaryDisplay.from_estimator(self.model,
                                                   x_data,
                                                   response_method=response_method,
                                                   plot_method="contour",
                                                   linestyles="--",
                                                   linewidths=1,
                                                   alpha=0.8,
                                                   levels=[0.5],
                                                   ax=plot.ax_)

        # For multiclass problem it's important to set hue_order in order to have the same order
        # between the DecisionBoundaryDisplay and the scatterplot
        sns.scatterplot(data=data,
                        x=x_data.columns[0], # 1st feature
                        y=x_data.columns[1], # 2nd feature
                        hue=hue,
                        hue_order=getattr(self.model, "classes_"),
                        palette=palette if palette else ["tab:red", "tab:blue"],
                        edgecolor="white",
                        ax=plot.ax_)
        plt.title(title if title else "Decision boundary of the trained LogisticRegression")

        if not ax:
            plt.show()
