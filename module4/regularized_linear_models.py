from typing import TypeVar

import data_handler as dh
import pandas as pd
from config import DataPath, TargetColumn
from model import LinearRegressionModel, LogisticRegressionModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from types_config import CvResults, DataSetType
from visualisation import plot_coefficients_of_linear_model

FEATURES = ["LotFrontage", "LotArea", "PoolArea", "YearBuilt", "YrSold"]

type ClassModelTypes = (LinearRegressionModel | LogisticRegressionModel)
Tclassmodel = TypeVar('Tclassmodel', bound=ClassModelTypes)

########
# DATA #
########
def load_data() -> DataSetType:
    """Load ames housing dataset."""
    data, targets = dh.load_data_from_file(DataPath.AMES_HOUSING.value, TargetColumn.AMES_HOUSING)
    return dh.get_subset(data, FEATURES), targets


#########
# MODEL #
#########


####################
# CROSS-VALIDATION #
####################
def cross_validation(model: ClassModelTypes, x_data: pd.DataFrame, y_data: pd.Series) -> CvResults:
    scores = model.kfold_cross_validate(x_data,
                                        y_data,
                                        nb_fold=10,
                                        scoring="neg_mean_squared_error",
                                        return_estimator=True,
                                        return_train_score=True)
    model.print_cross_validate(scores)

    return scores



############
# ANALYSIS #
############
def run_analysis():
    # Load data
    housing = load_data()

    # Build model. By default, the training error is in average one order of magnitude lower than
    # the testing error, which indicates overfitting.
    regression = LinearRegressionModel.build_pipeline_with_transformer(
        transformers=[PolynomialFeatures(degree=2, include_bias=False)],
        model=LinearRegression()
    )
    scores = cross_validation(regression, *housing)
    plot_coefficients_of_linear_model(regression.get_coefficients(scores))

if __name__ == "__main__":
    run_analysis()
