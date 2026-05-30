import data_handler as dh
import pandas as pd
from model import DecisionTreeRegressorModel
from sklearn.metrics import mean_absolute_error


###########
# SECTION #
###########
def simple_test_continuous_target_prediction(data: pd.DataFrame, targets: pd.Series):
    """
    Test continuous target prediction.

    We get alsmot a perfect prediction score. It's way to optimistic and there is some
    problems with the current model.

    Indeed, we trained and predicted on the same dataset. Since our decision tree was
    fully grown, every sample in the dataset is stored in a leaf node. Therefore, our
    decision tree fully memorized the dataset given during fit and therefore made no
    error when predicting.

    => This error is called the empirical or training error.
    """
    # 1. Build a decision tree regressor model
    regressor = DecisionTreeRegressorModel.build()

    # 2. Train the model on the whole dataset
    regressor.train(data, targets)

    # 3. Try out model performance
    predicted_targets = regressor.model.predict(data)

    # 4. The mean absolute error of our regressor is 0
    score = mean_absolute_error(targets, predicted_targets)
    print(f"On average, our regressor makes an error of {score:.2f} k$")

    # 4bis. The accuracy of the model is almost 1 (0.97)
    regressor.print_training_accuracy(targets, predicted_targets)


def test_continuous_target_prediction(data: pd.DataFrame, targets: pd.Series):
    """
    Instead of the previous metod, use train-test split on the dataset.

    Then performe a cross validation.
    """
    # 1. Randomnly split data between train and test set
    train_test_split = dh.sklearn_train_test_split(data, targets)

    # 2. Build a decision tree regressor model
    regressor = DecisionTreeRegressorModel.build()

    # 3. Train the model and calculate its accuracy (training and testing)
    regressor.start(*train_test_split)

    # 4. Perform a shuffle-split cross validation.
    scores = regressor.shuffle_split_cross_validate(
        data, targets, 40, "neg_mean_absolute_error"
    )
    regressor.print_shuffle_split_cross_validation_accuracy(scores)

    scores = regressor.shuffle_split_cross_validate(
        data, targets, 40, "neg_mean_absolute_percentage_error"
    )
    regressor.print_shuffle_split_cross_validation_accuracy(scores)


############
# ANALYSIS #
############
def run_analysis():
    # Load data
    housing = dh.load_california_dataset()

    # simple_test_continuous_target_prediction(*housing)
    test_continuous_target_prediction(*housing)

if __name__ == "__main__":
    run_analysis()
