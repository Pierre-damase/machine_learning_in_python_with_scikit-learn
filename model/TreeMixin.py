import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


class TreeMixin:
    """
    Mixin class to define some methods specific to tree models such as plot the corresponding
    decision tree.
    """
    def plot_decision_tree(self, features: list[str]):
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
