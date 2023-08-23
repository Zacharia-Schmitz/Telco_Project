import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import prepare as p
import explore as e

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)


def explore_categorical(train, target, cat_var):
    """
    Explore the relationship between a binary target variable and a categorical variable.

    Parameters:
    train (pandas.DataFrame): The training data.
    target (str): The name of the binary target variable.
    cat_var (str): The name of the categorical variable to explore.

    Returns:
    None
    """
    # Print the name of the categorical variable
    print(cat_var, "&", target)
    print("")

    # Calculate the chi-squared test statistic, p-value, degrees of freedom, and expected values
    ct = pd.crosstab(train[cat_var], train[target], margins=True)
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    print(f"Chi2: {chi2}")
    print(f"P-value: {p}")
    print(f"Degrees of Freedom: {dof}")

    # Create a count plot of the categorical variable split by the target variable
    p = sns.catplot(x=cat_var, hue=target, data=train, kind="count")
    p.fig.suptitle(f"{cat_var} by {target}")

    # Show the plot
    plt.show()


def explore_bivariate_quant(train, target, quant_var):
    """
    Explore the relationship between a quantitative variable and a binary target variable.

    Parameters:
    train (pandas.DataFrame): The training data.
    target (str): The name of the binary target variable.
    quant_var (str): The name of the quantitative variable to explore.

    Returns:
    None
    """

    # Compare the means of the quantitative variable between the two groups defined by the target variable
    e.compare_means(train, target, quant_var)

    # Create a boxen plot of the quantitative variable split by the target variable
    # plt.figure(figsize=(4, 4))
    e.plot_boxen(train, target, quant_var)

    # Show the plot
    plt.show()


def plot_boxen(train, target, quant_var):
    """
    Plot a boxen plot of a quantitative variable split by a binary target variable.

    Parameters:
    train (pandas.DataFrame): The training data.
    target (str): The name of the binary target variable.
    quant_var (str): The name of the quantitative variable to plot.

    Returns:
    None
    """
    # Calculate the average value of the quantitative variable
    average = train[quant_var].mean()

    # Create a boxen plot with the target variable on the x-axis and the quantitative variable on the y-axis
    # The plot is colored by the target variable and uses the "red" color palette
    sns.boxenplot(data=train, x=target, y=quant_var, hue=target)

    # Add a title to the plot with the name of the quantitative variable
    plt.title(quant_var)

    # Add a horizontal line to the plot at the average value of the quantitative variable
    plt.axhline(average, ls="--")


def compare_means(train, target, quant_var, alt_hyp="two-sided"):
    """
    Compare the means of a quantitative variable between two groups defined by a binary target variable using the Mann-Whitney U test.

    Parameters:
    train (pandas.DataFrame): The training data.
    target (str): The name of the binary target variable.
    quant_var (str): The name of the quantitative variable to compare.
    alt_hyp (str, optional): The alternative hypothesis for the test. Defaults to 'two-sided'.

    Returns:
    None
    """
    # Select the values of the quantitative variable for each group
    x = train[train[target] == 1][quant_var]
    y = train[train[target] == 0][quant_var]

    # Calculate the Mann-Whitney U test statistic and p-value
    stat, p = stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)

    # Print the results of the test
    print("Mann-Whitney Test:")
    print(f"Stat = {stat}")
    print(f"P-Value = {p}")


def compute_categorical_feature_importance(df, target_variable):
    """
    Computes the mutual information score between each categorical variable and the target variable.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    target_variable (str): The name of the target variable.

    Returns:
    pandas.Series: A Pandas Series containing the mutual information score for each categorical variable.
    """
    # select categorical variables excluding the response variable
    categorical_variables = df.select_dtypes(include=object).drop(
        target_variable, axis=1
    )

    # compute the mutual information score between each categorical variable and the target
    feature_importance = categorical_variables.apply(
        lambda x: mutual_info_score(x, df[target_variable])
    ).sort_values(ascending=False)

    # visualize feature importance
    ax = feature_importance.plot(kind="barh", figsize=(10, 8), zorder=2, width=0.8)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance (SK Learn's Mutual Info Score)")
    plt.show()

    return feature_importance
