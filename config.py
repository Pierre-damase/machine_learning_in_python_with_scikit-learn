from enum import Enum
from pathlib import Path

# Numerical features of ames housing dataset
AMES_HOUSING_NUMERICAL_FEATURES = [
  "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
  "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
  "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
  "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
  "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]

# Features for generated dataset
GENERATED_DATASET_FEATURES = ["One", "Two"]

# Features and target for synthetic dataset
SYNTHETIC_DATASET_FEATURE = "Feature"
SYNTHETIC_DATASET_TARGET = "Target"

class DataPath(Enum):
    """Path to dataset or tuned hyperparameter saved as csv."""
    ADULT_CENSUS = Path("./data/data_set/adult.arff")
    AMES_HOUSING = Path("./data/data_set/ames_housing_no_missing.csv")
    BLOOD_TRANSFUSION = Path("./data/data_set/blood_transfusion.csv")
    PENGUIN = Path("./data/data_set/penguins.csv")
    HYPERPARAMETER_TUNING = Path("./data/hyperparameter_tuning/")


class TargetColumn(str, Enum):
    """Enum to specify the target column of a given dataset."""
    ADULT_CENSUS = "class"
    AMES_HOUSING = "SalePrice"
    BLOOD_TRANSFUSION = "Class"
    PENGUIN = "Species"
    GENERATED_DATASET = "Class" # target column for generated dataset
