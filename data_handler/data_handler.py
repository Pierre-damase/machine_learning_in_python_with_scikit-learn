from pathlib import Path
from scipy.io import arff
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)
from typing import overload

import pandas as pd
import numpy as np


#############
# LOAD DATA #
#############
def load_data(path: Path, target: str):
    """Load data as a DataFrame."""
    if str(path).endswith(".csv"):
        return load_data_from_csv(path, target)
    return load_data_from_arff(path, target)

def load_data_from_arff(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load .arff file as a DataFrame."""
    # Load as DataFrame
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)

    # Drop education-num because educati/extratcon-num and education give the same information
    # It's possible to add it a this point because the data are known
    df = df.drop(columns="education-num")

    # Str load as bytes, a conversion is needed
    str_df = get_subset(df, dtypes=[object])
    str_df = str_df.stack().str.decode('utf-8').unstack()

    # Return the complete df; i.e. the concatenation between the numerical df
    # and the df with byte converted as str
    return _extract_target(
        pd.concat([get_subset(df, to_filter_out=str_df.columns), str_df], sort=False, axis=1),
        target
        )

@overload
def load_data_from_csv(path: Path, target: None = None) -> pd.DataFrame:
    """Load .csv file as DataFrame without extracting a target."""
    return load_data_from_csv(path)

@overload
def load_data_from_csv(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load .csv file as DataFrame with extracting a target."""
    return load_data_from_csv(path, target)

def load_data_from_csv(path: Path,
                       target: str | None = None) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Load .csv file as DataFrame."""
    data = pd.read_csv(path)
    if target is not None:
        return _extract_target(data, target)
    return data

def load_california_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load california housing dataset.

    Returns
    -------
    data: general real-estate and geographical information

    target: median value of houses in California. Transform the prices from
        the 100 range to the thousand dollars range for visualization
    """
    housing = fetch_california_housing(as_frame=True)
    return housing.data, housing.target * 100


########################
# MANIPULATE DATAFRAME #
########################
def _extract_target(data: pd.DataFrame, targets: str) -> tuple[pd.DataFrame, pd.Series]:
    """Extract target from DataFrame"""
    return data.drop(columns=[targets]), pd.Series(data[targets])

@overload
def get_subset(data: pd.DataFrame,
               columns: list[str],
               dtypes: list[type] | None = None,
               to_filter_out: pd.Index | None = None) -> pd.DataFrame:
    """Get a subset from column names."""
    get_subset(data, columns=columns)

@overload
def get_subset(data: pd.DataFrame,
               columns: list[str] | None = None,
               dtypes: list[type] = [],
               to_filter_out: pd.Index | None = None) -> pd.DataFrame:
    """Get a subset from dtypes."""
    if not dtypes:
        raise Exception("To get a subset from dtypes, the list of dtypes must be filled in.")
    get_subset(data, dtypes=dtypes)

@overload
def get_subset(data: pd.DataFrame,
               columns: list[str] | None = None,
               dtypes: list[type] | None = None,
               to_filter_out: pd.Index = pd.Index([])) -> pd.DataFrame:
    """Get a subset from filtering out a peculiar subset."""
    if to_filter_out.empty:
        raise Exception("A peculiar subset is required to filter it out from the DataFrame.")
    get_subset(data, to_filter_out=to_filter_out)

def get_subset(data: pd.DataFrame,
               columns: list[str] | None = None,
               dtypes: list[type] | None = None,
               to_filter_out: pd.Index | None = None) -> pd.DataFrame:
    """
    Get a subset either from column names or dtypes and as well from rows that do not have a match
    with a peculiar subset.
    """
    if columns is not None:
        return pd.DataFrame(data[columns])
    elif dtypes is not None:
        return data.select_dtypes(dtypes)
    elif to_filter_out is not None:
        return data.loc[:, ~data.columns.isin(to_filter_out)]
    return data

def manual_train_test_split(data: pd.DataFrame,
                            targets: pd.Series,
                            test_size: float = 0.25) -> list[pd.DataFrame|pd.Series]:
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

def sklearn_train_test_split(data: pd.DataFrame,
                             targets: pd.Series,
                             test_size: float = 0.25) -> list[pd.DataFrame|pd.Series]:
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
    return train_test_split(data, targets, random_state=42, test_size=test_size, shuffle=True)

def get_all_categories(data: pd.DataFrame) -> list[str]:
    """Get all categories available for each categorical feature."""
    return [sorted(data[col].unique().tolist()) for col in data.columns]

def get_cardinality_features(data: pd.DataFrame,
                             threshold: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get categorical features with high cardinality.

    Threshold is really low in order to try target encoder with current data.

    Return
    ------
    The tuple(high, low) with high the categorical features with high cardinality
                          low the categorical features with low cardinality
    """
    cardinality = {col : data[col].nunique() for col in data.columns}
    return data[[k for k,v in cardinality.items() if v > threshold]], \
           data[[k for k,v in cardinality.items() if v <= threshold]]


######################
# DATA NORMALIZATION #
######################
def standard_scaler(x_train: pd.DataFrame) -> np.ndarray[any]:
    """
    Apply normalization to data using the standard scaler of scikit-learn.

    This transformer shifts and scales each feature individually in order to
    have a 0-mean and a unit standard deviation for each feature.
    """
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
def ordinal_encoder(data: pd.DataFrame):
    """Encode each category with a different number using OrdinalEncoder."""
    encoder = OrdinalEncoder().set_output(transform="pandas")
    return encoder.fit_transform(data)

def one_hot_encoder(data: pd.DataFrame, sparse_output: bool=True):
    """
    Create a new column for each category with either 0 or 1 as value for the row

    Parameter
    ---------
    data: adult census DataFrame

    sparse_output: set to true to use sparse matrix, a effecient data structure when most of the
    elements are zero which store only nonzero items
    """
    transform = "default" if sparse_output else "pandas"
    encoder = OneHotEncoder(sparse_output=sparse_output).set_output(transform=transform)
    return encoder.fit_transform(data)
