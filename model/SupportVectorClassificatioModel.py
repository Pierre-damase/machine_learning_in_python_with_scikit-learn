from sklearn.svm import SVC
from types_config import Tmodel

from .Model import Model


class SupportVectorClassificationModel(Model[SVC, Tmodel]):
    """
    Support vector machine classifier (SVM) model.

    In it's most simple form, a SVM classifier is a linear classifier behaving
    similarity to a logistic regression. Although, the optimization to find the
    optimal weights of the linear model are different.

    SVM classifier can become more flexible/expressive by using a so-called kernel
    which can make the model non-linear.

    Parameter gamma allows to tune the flexibility of the model.
    """
    _estimator_class = SVC
