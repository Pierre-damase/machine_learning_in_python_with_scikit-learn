import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, balanced_accuracy_score)
from visualisation import show_confusion_matrix, show_pr_curve, show_roc_curve

type MetricCurveType = (PrecisionRecallDisplay | RocCurveDisplay)

class ClassifierMixin:
    """Mixin class to define specific logic of classifier."""
    ###################
    # TRAIN & PREDICT #
    ###################
    def predict(self,
                x_train: pd.DataFrame,
                x_test: pd.DataFrame,
                y_train: pd.Series,
                y_test: pd.Series) -> None:
        """
        Predict the target and check model performance.

        Parameters
        ----------
        x_train: training data

        x_test: testing data

        y_train: training target

        y_test: testing target
        """
        # Get model
        model = getattr(self, "model")

        # Predict target
        y_predicted = getattr(model, "predict")(x_train)

        # Check model performance with training data
        getattr(self, "print_training_accuracy")(y_train, y_predicted)

        # Check model performance with testing data
        getattr(self, "print_testing_accuracy")(x_test, y_test)

    def get_misclassified(self,
                           x_data: pd.DataFrame,
                           y_data: pd.Series) -> pd.DataFrame:
        """
        Get misclassified samples.

        Parameters
        ----------
        x_data: input data use for the prediction

        y_data: target, i.e. the actual class for each sample and use to compare the prediction
        """
        # Get model
        model = getattr(self, "model")

        # Predict target
        y_predicted = getattr(model, "predict")(x_data)

        # Get misclassified
        return x_data.iloc[np.flatnonzero(y_data != y_predicted)]

    def get_sample_weight(self,
                          y_data: pd.Series,
                          misclassified: pd.DataFrame) -> npt.NDArray[np.int8]:
        """
        Set a higher weight to misclassified samples to give them more importance at the training.

        The strategy is quite simple, misclassified samples have a weight of 1 and any other
        samples have a weight of 0.

        Parameters
        ----------
        y_data: targets

        misclassified: misclassified samples at the prediction
        """
        # Create an array of zeros with the same shape than the targets
        sample_weight = np.zeros_like(y_data, dtype=int)

        # Set the weight of misclassified samples to 1
        sample_weight[misclassified.index] = 1

        return sample_weight

    def get_ensemble_weight(self,
                            y_data: pd.Series,
                            l_misclassified: list[pd.DataFrame]) -> list[float]:
        """
        The idea is that several classifier won't have the same accuracy depending of the weight
        given to the samples. Therefore, when predicting a class, we should trust the best
        classifier slightly more than the others.

        This methods return the weight given to each classifier.

        Parameters
        ----------
        y_data: targets

        misclassified: misclassified samples at the prediction of the first model

        newly_misclassified: the model was trained again with samples reweighting to give more
        importance to misclassified samples
        """
        y_size = len(y_data)
        return [(y_size - len(misclassified)) / y_size for misclassified in l_misclassified]


    ################
    # METRIC CURVE #
    ################
    def compute_metric_curve(self,
                             metric_curve: type[MetricCurveType],
                             x_test: pd.DataFrame,
                             y_test: pd.Series,
                             pos_label: str,
                             ax: Axes | None = None) -> MetricCurveType:
        """
        Compute metric curve either a precision recall (PR) curve or a receiver operating
        characteristic (ROC) curve.
        """
        # Use plot chance for comparison with a dummy classifier.
        curve = metric_curve.from_estimator(getattr(self, "model"),
                                            x_test,
                                            y_test,
                                            pos_label=pos_label,
                                            marker="+",
                                            plot_chance_level=True,
                                            chance_level_kw={"color": "tab:orange",
                                                             "linestyle": "--"},
                                            ax=ax)

        return curve


    #####################
    # MODEL PERFORMANCE #
    #####################
    def evaluate_generalization_performance(self,
                                            x_test: pd.DataFrame,
                                            y_test: pd.Series,
                                            **kwargs) -> None:
        """
        Use various evaluation metrics to evaluate the predictive model generalization performance.

        [This method is higly specific to classifier.]

        Parameters
        ----------
        x_test: testing data

        y_test: testing targets

        **kwargs: additional parameters such as pos_label - the class considered as the positive
        class whene precision and recall metrics are computed.
        """
        assert "pos_label" in kwargs.keys(), \
            "Additional parameter pos_label is required to evaluate a classifier"

        # Get model
        model = getattr(self, "model")

        # Predict the testing target using the testing data
        y_predicted = getattr(model, "predict")(x_test)

        # Accuracy. A good baseline to compare the predictions with the true testing data,
        # sometimes called ground-truth. However, with class imbalance accuracy should not be used.
        print("\nThe accuracy, comparaison between the predictions and the true testing data, is "
              f"{np.mean(y_test == y_predicted):.3f}")

        # For classification problem with class imbalance it's better to use a balanced accuracy.
        counts = y_test.value_counts()
        if max([count / sum(counts) for count in counts]) > 0.6:
            print("\nThe balanced accuracy, which take into account class imbalance, is "
                    f"{balanced_accuracy_score(y_test, y_predicted):.3f}")

        # Set up plots
        _, axs = plt.subplots(ncols=3, figsize=(34, 10))

        # Confusion matrix
        confusion = ConfusionMatrixDisplay.from_estimator(model,
                                                          x_test,
                                                          y_test,
                                                          ax=axs[0])
        show_confusion_matrix(confusion, ax=axs[0])

        # Precision recall (PR) curve
        pr_curve = self.compute_metric_curve(PrecisionRecallDisplay,
                                             x_test,
                                             y_test,
                                             kwargs["pos_label"],
                                             axs[1])
        show_pr_curve(pr_curve, y_test, y_predicted, kwargs["pos_label"], ax=axs[1])

        # Receiver operating characteristic (ROC) curve
        roc_curve = self.compute_metric_curve(RocCurveDisplay,
                                              x_test,
                                              y_test,
                                              kwargs["pos_label"],
                                              axs[2])
        show_roc_curve(roc_curve)
