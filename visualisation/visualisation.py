import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
Use jointplot to display histograms of the distribution and a
scatterplot of any pair of numerical features. In order to observe
than scaler does not change the structure of the data. 
Only the axes get shifted and scaled.
"""
def scaler_jointplot(x_train: pd.DataFrame, x_train_scaled: pd.DataFrame, x_axis: str, y_axis: str) -> None:
    # Limit points to plot for clearer result
    limit = 300

    _jointplot(x_train[:limit], x_axis, y_axis, f"Jointplot of {x_axis} and {y_axis} before scalerer")
    _jointplot(x_train_scaled[:limit], x_axis, y_axis, f"Jointplot of {x_axis} and {y_axis} after scalerer")
    plt.show()

"""Jointplot."""
def _jointplot(data: pd.DataFrame, x_axis: str, y_axis: str, title: str) -> None:
    sns.jointplot(
        data=data,
        x=x_axis,
        y=y_axis,
        marginal_kws=dict(bins=15)
    )
    plt.suptitle(title)


"""Simple workflow to look at the data."""
def check_data(data: pd.DataFrame, targets: pd.Series) -> None:
    display_shape, display_hist, display_count, display_relationship, display_pairplot = \
        True, True, True, True, True
    str_columns = data.select_dtypes([str]).columns

    # Display data shape
    if display_shape:
        print(f"The dataset contains {data.shape[0]} samples and {data.shape[1]} features with\n"
              f"- numerical features: {', '.join([col for col in data.columns if col not in str_columns])}\n"
              f"- categorical features: {', '.join(str_columns)}")
    
    # Histogram visualisation of numerical values
    if display_hist:
        _histogram(data)

    # Count categorical values - could be useful to detect class imbalance
    if display_count:
        for col in str_columns:
            print(data[col].value_counts())

    # Look at the relationship between two variables
    if display_relationship:
        _crosstab(data, "education", "education-num")

    if display_pairplot:
        _pairplot(data, targets, ["age", "education-num", "hours-per-week"])

"""Simple visualisation of numerical values as histrogram."""
def _histogram(data: pd.DataFrame) -> None:
    _ = data.hist(figsize=(20, 14))
    plt.title('Histogram for numerical values')
    plt.show()

"""
Crosstab virusalisation to look at the relationship between two variables.

To detect redundant (or highly correlated) variables.
"""
def _crosstab(data: pd.DataFrame, x: str, y: str) -> None:
    tab = pd.crosstab(index=data[x], columns=data[y])
    print(tab)

"""
Pairplot virusalisation to show how each variable differs according to the target.

Can reveal interaction between variables.
"""
def _pairplot(data: pd.DataFrame, target: pd.Series, columns: list[str]) -> None:
    # Only plot a subset of the data for performance issue and keep the plot readable
    n_samples = 5000

    _ = sns.pairplot(
        data=pd.concat([data[:n_samples], target[:n_samples]], sort=False, axis=1),
        vars=columns,
        hue=target.name, # target is a pandas series and Series.name return his name
        plot_kws={"alpha": 0.2},
        height=3,
        diag_kind="hist",
        diag_kws={"bins": 30},
    )
    plt.show()