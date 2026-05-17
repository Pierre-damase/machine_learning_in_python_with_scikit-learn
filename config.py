from enum import Enum
from pathlib import Path

AMES_HOUSING_NUMERICAL_FEATURES = [
  "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
  "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
  "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
  "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
  "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]


class DataPath(Enum):
    ADULT_CENSUS = Path("./data/adult.arff")
    AMES_HOUSING = Path("./data/ames_housing_no_missing.csv")
    BLOOD_TRANSFUSION = Path("./data/blood_transfusion.csv")


class TargetColumn(str, Enum):
    """Enum to specify the target column of a given dataset."""
    ADULT_CENSUS = "class"
    AMES_HOUSING = "SalePrice"
    BLOOD_TRANSFUSION = "Class"
