import data_handler as dh

import pandas as pd


########
# DATA #
########
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return dh.make_gaussian_quantiles_dataset(), dh.make_moons_dataset(), dh.make_xor_dataset()

#################
# VISUALISATION #
#################

if __name__ == "__main__":
    moons = dh.make_moons_dataset()
    gauss = dh.make_gaussian_quantiles_dataset()
    xor = dh.make_xor_dataset()
    print(xor)
