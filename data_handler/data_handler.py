from pathlib import Path
from scipy.io import arff
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)

import pandas as pd
import numpy as np


#############
# LOAD DATA #
#############
"""Load data as a DataFrame."""
def load_data(path: Path, target: str):
    if str(path).endswith(".csv"):
        return load_data_from_csv(path, target)
    return load_data_from_arff(path, target)

"""Load .arff file as a DataFrame."""
def load_data_from_arff(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    # Load as DataFrame
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)

    # Drop education-num because educati/extratcon-num and education give the same information
    # It's possible to add it a this point because the data are known
    df = df.drop(columns="education-num")

    # Str load as bytes, a conversion is needed
    str_df = get_subset_from_dtypes(df, [object])
    str_df = str_df.stack().str.decode('utf-8').unstack()

    # Return the complete df; i.e. the concatenation between the numerical df 
    # and the df with byte converted as str
    return _extract_target(
        pd.concat([get_subset_filtered_out(df, str_df.columns), str_df], sort=False, axis=1),
        target
        )

"""Load .csv file as DataFrame."""
def load_data_from_csv(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    return _extract_target(pd.read_csv(path), target)

"""
Load california housing dataset.

Returns
-------
data: general real-estate and geographical information

target: median value of houses in California. Transform the prices from
        the 100 range to the thousand dollars range for visualization
"""
def load_california_dataset() -> tuple[pd.DataFrame, pd.Series]:
    housing = fetch_california_housing(as_frame=True)
    return housing.data, housing.target * 100


########################
# MANIPULATE DATAFRAME #
########################
"""Extract target from DataFrame"""
def _extract_target(data: pd.DataFrame, targets: str) -> tuple[pd.DataFrame, pd.Series]:
    return data.drop(columns=[targets]), data[targets]

"""Get a subset from column names."""
def get_subset(data: pd.DataFrame, target_subset: list[str]) -> pd.DataFrame:
    return data[target_subset]

"""Get a subset with the given types."""
def get_subset_from_dtypes(data: pd.DataFrame, types: list[type]):
    return data.select_dtypes(types)

"""Get all rows in a DataFrame that do not have a match with a peculiar subset."""
def get_subset_filtered_out(data: pd.DataFrame, to_filter_out: pd.Index) -> pd.DataFrame:
    return data.loc[:, ~data.columns.isin(to_filter_out)]

"""
Randomnly split data to build train and test set.

Returns
-------
List containing the train-test split of the data.
    x_train: training data

    x_test: testing data

    y_train: training target

    y_test: testing target
"""
def manual_train_test_split(data: pd.DataFrame, targets: pd.Series, test_size: int = 0.25) \
    -> list[pd.DataFrame|pd.Series]:
    # Build training set by using DataFrame.sample() select randomly some rows
    training_data = data.sample(frac=1-test_size)
    training_targets = targets.loc[targets.index.isin(training_data.index)]

    if training_data.shape[0] != training_targets.shape[0]:
        raise Exception(f"Error while building train data and targets.")

    # Test set
    testing_data = data.loc[~data.index.isin(training_data.index)]
    testing_targets = targets.loc[~targets.index.isin(training_data.index)]

    if testing_data.shape[0] != testing_targets.shape[0]:
        raise Exception(f"Error while building test data and targets.")

    return [training_data, testing_data, training_targets, testing_targets]

"""
Split data to build train and test set using scikit learn.

In scikit-learn setting the randome-state allows
to get deterministic results when we use a random number generator.
The randomness comes from shuffling the data,
which decides how the dataset is split into a train and test set.

The data scaling is applied to each feature individually, i.e. each column of the matrix.
For each feature, we substract it's mean and divide by its standard deviation.

Returns
-------
List containing the train-test split of the data.
    x_train: training data

    x_test: testing data

    y_train: training target

    y_test: testing target
"""
def sklearn_train_test_split(data: pd.DataFrame,
                             targets: pd.Series,
                             test_size: float = 0.25) -> list[pd.DataFrame|pd.Series]:
    return train_test_split(data, targets, random_state=42, test_size=test_size, shuffle=True)

"""Get all categories available for each categorical feature."""
def get_all_categories(data: pd.DataFrame) -> list[str]:
    return [sorted(data[col].unique().tolist()) for col in data.columns]

"""
Get categorical features with high cardinality.

Threshold is really low in order to try target encoder with current data.

Return
------
The tuple(high, low) with high the categorical features with high cardinality
                          low the categorical features with low cardinality
"""
def get_cardinality_features(data: pd.DataFrame, threshold: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    cardinality = {col : data[col].nunique() for col in data.columns}
    return data[[k for k,v in cardinality.items() if v > threshold]], \
           data[[k for k,v in cardinality.items() if v <= threshold]]

######################
# DATA NORMALIZATION #
######################
"""
Apply normalization to data using the standard scaler of scikit-learn.

This transformer shifts and scales each feature individually in order to
have a 0-mean and a unit standard deviation for each feature.
"""
def standard_scaler(x_train: pd.DataFrame) -> np.ndarray[any]:
    # 1. Use standard scaler
    scaler = StandardScaler().set_output(transform="pandas")
    scaler.fit(x_train)

    # scikit-learn convention:
    # the name of any attribute learned from the data ends with and underscrore
    print(f"The computed means {scaler.mean_} and standard deviations {scaler.scale_}")

    # 2. Data transformation
    print("Perform data transformation.")
    return scaler.transform(x_train)


#####################
# ENCODE CATEGORIES #
#####################
"""Encode each category with a different number using OrdinalEncoder."""
def ordinal_encoder(data: pd.DataFrame):
    encoder = OrdinalEncoder().set_output(transform="pandas")
    return encoder.fit_transform(data)

"""
Create a new column for each category with either 0 or 1 as value for the row

Parameter
---------
data: adult census DataFrame

sparse_output: set to true to use sparse matrix, a effecient data structure 
               when most of the elements are zero which store only nonzero
               items 
"""
def one_hot_encoder(data: pd.DataFrame, sparse_output: bool=True):
    transform = "default" if sparse_output else "pandas"
    encoder = OneHotEncoder(sparse_output=sparse_output).set_output(transform=transform)
    return encoder.fit_transform(data)
