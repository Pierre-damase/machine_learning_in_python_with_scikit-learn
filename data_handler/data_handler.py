from ctypes import ArgumentError
from pathlib import Path
from typing import overload

import numpy as np
import pandas as pd
from config import (GENERATED_DATASET_FEATURES, SYNTHETIC_DATASET_FEATURE,
                    SYNTHETIC_DATASET_TARGET, TargetColumn)
from scipy.io import arff
from sklearn.datasets import (fetch_california_housing, make_blobs,
                              make_gaussian_quantiles, make_moons)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from types_config import DataSetType, SplitSetType


#############
# LOAD DATA #
#############
@overload
def load_data_from_file(path: Path,
                        target: None = None,
                        columns: list[str] | None = None,
                        drop_na: bool | None = None) -> pd.DataFrame:
    """
    Load data from a file without extracting the target.

    Parameter
    ---------
    path: path of the file, either a .csv or .arff file

    target: is None to not extract the target from the dataset

    columns: to filter out some columns

    drop_na: remove missing values
    """
    return load_data_from_file(path, target=target, columns=columns, drop_na=drop_na)

@overload
def load_data_from_file(path: Path,
                        target: str,
                        columns: list[str] | None = None,
                        drop_na: bool | None = None) -> DataSetType:
    """
    Load data from a file with extracting a target.

    Parameter
    ---------
    path: path of the file, either a .csv or .arff file

    target: extract the specified target from the dataset

    columns: to filter out some columns

    drop_na: remove missing values
    """
    return load_data_from_file(path, target=target, columns=columns, drop_na=drop_na)

def load_data_from_file(path: Path,
                        target: str | None = None,
                        columns: list[str] | None = None,
                        drop_na: bool | None = None) -> pd.DataFrame | DataSetType:
    """
    Load data as a DataFrame.

    Parameter
    ---------
    path: path of the file, either a .csv or .arff file

    target: if specify, extract the target from the dataset

    columns: to filter out some columns

    drop_na: remove missing values
    """
    # .csv file
    if str(path).endswith(".csv"):
        return _load_data_from_csv(path, target=target, columns=columns, drop_na=drop_na)

    # .arff file
    if target is None:
        raise ArgumentError("For .arff file, a target is required.")
    return _load_data_from_arff(path, target=target)

def _load_data_from_arff(path: Path, target: str) -> DataSetType:
    """Load .arff file as a DataFrame."""
    # Load as DataFrame
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)

    # Drop education-num because education-num and education give the same information
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
def _load_data_from_csv(path: Path,
                        target: None = None,
                        columns: list[str] | None = None,
                        drop_na: bool | None = None) -> pd.DataFrame:
    """Load .csv file as DataFrame without extracting a target."""
    return _load_data_from_csv(path, columns=columns, drop_na=drop_na)

@overload
def _load_data_from_csv(path: Path,
                        target: str,
                        columns: list[str] | None = None,
                        drop_na: bool | None = None) -> DataSetType:
    """Load .csv file as DataFrame with extracting a target."""
    return _load_data_from_csv(path, target=target, columns=columns, drop_na=drop_na)

def _load_data_from_csv(path: Path,
                        target: str | None = None,
                        columns: list[str] | None = None,
                        drop_na: bool | None = None) -> pd.DataFrame | DataSetType:
    """Load .csv file as DataFrame."""
    data = pd.read_csv(path)

    # Filter the data
    if columns is not None:
        # If a target is given, it must be part of the subset in order to extract it after
        tmp = columns + [target] if target is not None and target not in columns else columns
        data = get_subset(data, columns=tmp)

    # Drop missing valuese
    if drop_na:
        data.dropna(inplace=True)

    # Extrat the target
    if target is not None:
        return _extract_target(data, target)

    return data

def load_california_dataset() -> DataSetType:
    """
    Load california housing dataset.

    Returns
    -------
    data: general real-estate and geographical information

    target: median value of houses in California. Transform the prices from
        the 100 range to the thousand dollars range for visualization
    """
    housing = fetch_california_housing(as_frame=True)
    return make_dataset(x_data=getattr(housing, "data"), y_data=getattr(housing, "target") * 100)


#############
# MAKE DATA #
#############
def make_blobs_dataset(n_samples: int, centers: list[list[int]]) -> DataSetType:
    """Make isotropic gaussian blobs for clustering."""
    x_data, y_data = make_blobs(n_samples=n_samples, centers=centers, random_state=0)
    return make_dataset(x_data=pd.DataFrame(x_data, columns=GENERATED_DATASET_FEATURES),
                        y_data=pd.Series(y_data, name=TargetColumn.GENERATED_DATASET.value))

def make_gaussian_quantiles_dataset() -> DataSetType:
    """Make gaussian quantiles dataset. a non-linear dataset."""
    x_data, y_data = make_gaussian_quantiles(
        n_samples=100, n_features=2, n_classes=2, random_state=42
    )
    return make_dataset(x_data=pd.DataFrame(x_data, columns=GENERATED_DATASET_FEATURES),
                        y_data=pd.Series(y_data, name=TargetColumn.GENERATED_DATASET.value))

def make_moons_dataset() -> DataSetType:
    """
    Make moons dataset, a simple toy dataset to visualize clustering and classification
    algorithm. A non-linear dataset.
    """
    x_data, y_data = make_moons(n_samples=100, noise=0.13, random_state=42)
    return make_dataset(x_data=pd.DataFrame(x_data, columns=GENERATED_DATASET_FEATURES),
                        y_data=pd.Series(y_data, name=TargetColumn.GENERATED_DATASET.value))

def make_xor_dataset() -> DataSetType:
    """
    Dataset where the data points are sampled from a uniform distribution in a 2D sapce and the
    class is defined by the Exclusive OR (XOR) operation on the two features. The target class is 1
    if only one of the two features is greater than 0. The target class is 0 otherwise.
    A non-linear dataset.
    """
    x_data = pd.DataFrame(np.random.RandomState(0).uniform(low=-1, high=1, size=(200, 2)),
                          columns=GENERATED_DATASET_FEATURES)
    y_data = pd.Series(np.logical_xor(
        x_data[GENERATED_DATASET_FEATURES[0]] > 0,
        x_data[GENERATED_DATASET_FEATURES[1]] > 0
    ), name=TargetColumn.GENERATED_DATASET.value)

    return make_dataset(x_data=x_data, y_data=y_data)

def make_bootstrap_samples(x_train: pd.DataFrame,
                           y_train: pd.Series,
                           n_bootstraps: int) -> list[DataSetType]:
    """
    Bootstrapping involves uniformly resampling n data points from a dataset of n points, with
    replacement, ensuring each sample has an equal chance of selection.

    As a result, the output is another dataset with n data points, likely containing duplicates.
    Consequently, some data points from the original dataset may not be selected for a bootstrap
    sample. These unselected data points are often reffered to as the out-of-bag sample.

    Moreover, in practice hundreds of variations of the original training set are made.

    Parameters
    ----------
    x_train: training features

    y_train: training targets

    n_boostraps: how many variations of the original dataset to produce

    seed: a seed to initialize the BitGenerator in order to randomly select data points from the
    original dataset

    Return
    ------
    A list of (x_data, y_data) of size n_bootstaps. x_data contains the randomly selected data
    points and y_data the associated targets
    """
    x_data: list[DataSetType] = []
    for i in range(n_bootstraps):
        rng = np.random.default_rng(i)

        # np.arange create an evenly spaced values from 0 to n-1 with n the size of y_train, i.e
        # the number of features/targets.
        # Then randomly select n data points index that will be keep for the bootstrap sample.
        # Replace set to true means that the same index can be selected multiple times
        boostrap_index = rng.choice(np.arange(y_train.shape[0]),
                                    size=y_train.shape[0],
                                    replace=True)

        # Only keep data points with the selected index
        x_data.append(
            make_dataset(x_data=x_train.iloc[boostrap_index], y_data=y_train.iloc[boostrap_index])
        )

    return x_data

def analyse_bootstrap_samples(bootstrap_samples: list[DataSetType]) -> None:
    """
    To determine how many unique data from the original dataset are present in the bootstrap
    samples.
    """
    ratio: list[float] = []
    for sample in bootstrap_samples:
        ratio.append((np.unique(sample["x_data"]).size / sample["x_data"].size) * 100)
    print(f"On average, {np.mean(ratio):.2f}% of the original data present in the "
          "bootstrap samples.")

def make_synthetic_dataset(x_min: float,
                           x_max: float,
                           seed: int,
                           noise_shift: float,
                           y_train_shift: int = 0,
                           n_samples:int = 50) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Generate simple synthetic dataset to better understand bootstraping / grandient-boosting.

    Parameters
    ----------
    x_min: lower boundary of the features

    x_max: higher boundary of the features

    seed: a seed to initialize the generator

    noise_shift: add some shift to the noise

    y_train_shift: add some shift to the target

    n_samples: how many samples to generate

    Returns
    -------
    x_train: training data

    x_test: testing data, i.e. generated data not use to train the model

    y_train: targets
    """
    # Create a random number generator
    rng = np.random.default_rng(seed)

    # Features
    x_train = rng.uniform(x_min, x_max, size=n_samples)
    x_test = np.linspace(x_max, x_min, num=300)

    # Target
    noise = noise_shift * rng.normal(size=(n_samples,))
    y_train = x_train**3 - 0.5 * (x_train + y_train_shift) ** 2 + noise
    y_train /= y_train.std()

    return pd.DataFrame(x_train, columns=[SYNTHETIC_DATASET_FEATURE]), \
        pd.DataFrame(x_test, columns=[SYNTHETIC_DATASET_FEATURE]), \
        pd.Series(y_train, name=SYNTHETIC_DATASET_TARGET)


########################
# MANIPULATE DATAFRAME #
########################
def make_dataset(x_data: pd.DataFrame, y_data: pd.Series) -> DataSetType:
    """Return a dataset that contains both the input data and the target."""
    return {"x_data": x_data, "y_data": y_data}

def _extract_target(x_data: pd.DataFrame, y_data: str) -> DataSetType:
    """Extract target from DataFrame"""
    return make_dataset(x_data=x_data.drop(columns=[y_data]), y_data=pd.Series(x_data[y_data]))

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
    return pd.DataFrame(data[[k for k,v in cardinality.items() if v > threshold]]), \
           pd.DataFrame(data[[k for k,v in cardinality.items() if v <= threshold]])

def get_sample(data: DataSetType, size: int) -> DataSetType:
    """
    Get a random sample of items from the original dataset.

    Parameter
    ---------
    data: input data and target

    size: represents the absolute number of samples to keep in the subset

    stratify: to split data in a stratified fashion, in order to keep the same percentage of
    samples of each target class as the complete set.
    """
    assert size < len(data["x_data"]), "Cannot take a larger sample than population"

    # Build the dataset from the input data and targets
    df = pd.concat([data["x_data"], data["y_data"]], axis=1)

    # In order to split the data in a stratified fashion, discretize continuous variable into
    # equal-sized buckets. For example, 1000 values for 10 quantiles would produce a categorical
    # targets with 10 classes. For discret variable, no need to discretize beforehand.
    if data["y_data"].dtype in [float, int]:
        df["target_bins"] = pd.qcut(df[data["y_data"].name], q=10, labels=False, duplicates="drop")
    stratify = df["target_bins"] if "target_bins" in df.columns else df[data["y_data"].name]

    # Determine the proportion of data to keep.
    proportion = size / len(df)

    # Get the random sample of items using the train_test_split methods of sklearn
    sample, _ = train_test_split(df, train_size=proportion, stratify=stratify, random_state=42)

    return pd.DataFrame(sample[x_data.columns]), pd.Series(sample[y_data.name])

def get_between(data: DataSetType, feature: str, left: int, right: int) -> DataSetType:
    """
    Return a new dataset where feature values lie between lef t and right.

    Parameter
    ---------
    data: input data and target

    feature: which feature value that must lie between left and right

    left: left boundary

    right: right boundary
    """
    mask = data["x_data"][feature].between(left, right)
    return make_dataset(x_data=pd.DataFrame(data["x_data"][mask]),
                        y_data=pd.Series(data["y_data"][mask]))

####################
# TRAIN/TEST SPLIT #
####################
def manual_train_test_split(x_data: pd.DataFrame,
                            y_data: pd.Series,
                            test_size: float = 0.25) -> SplitSetType:
    """
    Randomnly split data to build train and test set.

    Returns
    -------
    List containing the train-test split of the data.
    x_train: training data

    x_test: testing data

    y_train: training targets

    y_test: testing targets
    """
    # Build training set by using DataFrame.sample() select randomly some rows
    x_train = x_data.sample(frac=1-test_size)
    y_train = y_data.loc[y_data.index.isin(x_train.index)]

    if x_train.shape[0] != y_train.shape[0]:
        raise Exception(f"Error while building train data and targets.")

    # Test set
    x_test = x_data.loc[~x_data.index.isin(x_train.index)]
    y_test = y_data.loc[~y_data.index.isin(x_train.index)]

    if x_test.shape[0] != y_test.shape[0]:
        raise Exception(f"Error while building test data and targets.")

    return {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}

def sklearn_train_test_split(x_data: pd.DataFrame,
                             y_data: pd.Series,
                             test_size: float = 0.25) -> SplitSetType:
    """
    Split data to build train and test set using scikit learn.

    In scikit-learn setting the randome-state allows
    to get deterministic results when we use a random number generator.
    The randomness comes from shuffling the data,
    which decides how the dataset is split into a train and test set.

    The data scaling is applied to each feature individually, i.e. each column of the matrix.
    For each feature, we substract it's mean and divide by its standard deviation.

    Parameter
    ---------
    x_data: the whole input data (features)

    y_data: the whole targets

    test_size: represents the proportion of the dataset to include in the test split

    stratify: to split data in a stratified fashion, in order to keep the same percentage of
    samples of each target class as the complete set.

    Returns
    -------
    List containing the train-test split of the data.
    x_train: training data

    x_test: testing data

    y_train: training target

    y_test: testing target
    """
    assert test_size > 0 and test_size < 1, f"Invalid test_siz value {test_size}. Must be in ]0;1["

    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        random_state=42,
                                                        test_size=test_size,
                                                        shuffle=True)
    return {
        "x_train": pd.DataFrame(x_train), "x_test": pd.DataFrame(x_test),
        "y_train": pd.Series(y_train), "y_test": pd.Series(y_test)
    }

def get_train_split(split_data: SplitSetType) -> tuple[pd.DataFrame, pd.Series]:
    """From a train/test split dataset get only the training data/targets."""
    return split_data["x_train"], split_data["y_train"]

######################
# DATA NORMALIZATION #
######################
def standard_scaler(x_train: pd.DataFrame) -> pd.DataFrame:
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
