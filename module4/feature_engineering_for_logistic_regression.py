import data_handler as dh
import pandas as pd
from types_config import DataSetType


########
# DATA #
########
def load_data() -> tuple[DataSetType, DataSetType, DataSetType]:
    return dh.make_gaussian_quantiles_dataset(), dh.make_moons_dataset(), dh.make_xor_dataset()

#################
# VISUALISATION #
#################


############
# ANALYSIS #
############
def run_analysis():
    moons = dh.make_moons_dataset()
    gauss = dh.make_gaussian_quantiles_dataset()
    xor = dh.make_xor_dataset()
    print(xor)

if __name__ == "__main__":
    run_analysis()
