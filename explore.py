import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def explore_cat(train, target, cat_var):
    """
    Explore the relationship between a binary target variable and a categorical variable.

    Parameters:
    train (pandas.DataFrame): The training data.
    target (str): The name of the binary target variable.
    cat_var (str): The name of the categorical variable to explore.

    Returns:
    None
    """
    # Calculate the observed contingency table and perform a chi-squared test
    observed = pd.crosstab(train[cat_var], train[target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'Chi2: {chi2}, p-value: {p}')
    
    # Create a vertical bar plot of the categorical variable split by the target variable
    plt.figure(figsize=(4,4))
    sns.barplot(x=target, y=cat_var, data=train, alpha=.8)
    
    # Add a horizontal line to the plot at the overall rate of the target variable
    overall_rate = train[target].mean()
    plt.axvline(overall_rate, ls='--')
    
    # Show the plot
    plt.show()

def explore_int(train, target, cat_var):
    """
    Explore the relationship between a binary target variable and an integer variable.

    Parameters:
    train (pandas.DataFrame): The training data.
    target (str): The name of the binary target variable.
    cat_var (str): The name of the integer variable to explore.

    Returns:
    None
    """
    # Calculate the observed contingency table and perform a chi-squared test
    observed = pd.crosstab(train[target], train[cat_var])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'Chi2: {chi2}, p-value: {p}')
    
    # Create a horizontal bar plot of the integer variable split by the target variable
    plt.figure(figsize=(4,4))
    sns.barplot(x=cat_var, y=target, data=train, alpha=.8, orient='h')
    
    # Add a vertical line to the plot at the overall rate of the target variable
    overall_rate = train[target].mean()
    plt.axvline(overall_rate, ls='--')
    
    # Show the plot
    plt.show()

def explore_bivariate_categorical(train, target, cat_var):
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
    print(cat_var, "\n_____________________\n")
    
    # Calculate the chi-squared test statistic, p-value, degrees of freedom, and expected values
    ct = pd.crosstab(train[cat_var], train[target], margins=True)
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    print(f'Chi2: {chi2}, p-value: {p}, dof: {dof}')
    
    # Create a count plot of the categorical variable split by the target variable
    p = sns.catplot(x=cat_var, hue=target, data=train, kind='count')
    p.fig.suptitle(f'{cat_var} by {target}')
    
    # Print the observed and expected contingency tables
    print("\nobserved:\n", ct)
    print("\nexpected:\n", expected)
    
    # Show the plot
    plt.show()
    print("\n_____________________\n")
    
def explore_bivariate(train, target, cat_vars, quant_vars):
    """
    Explore the relationship between a binary target variable and multiple categorical and quantitative variables.

    Parameters:
    train (pandas.DataFrame): The training data.
    target (str): The name of the binary target variable.
    cat_vars (list of str): The names of the categorical variables to explore.
    quant_vars (list of str): The names of the quantitative variables to explore.

    Returns:
    None
    """
    # Explore each categorical variable
    for cat in cat_vars:
        explore_bivariate_categorical(train, target, cat)
    
    # Explore each quantitative variable
    for quant in quant_vars:
        explore_bivariate_quant(train, target, quant)


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
    # Print the name of the quantitative variable
    print(quant_var, "\n____________________")
    
    # Compare the means of the quantitative variable between the two groups defined by the target variable
    compare_means(train, target, quant_var)
    
    # Print a separator line
    print("____________________")
    
    # Create a boxen plot of the quantitative variable split by the target variable
    plt.figure(figsize=(4,4))
    plot_boxen(train, target, quant_var)
    
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
    plt.axhline(average, ls='--')

def compare_means(train, target, quant_var, alt_hyp='two-sided'):
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
    x = train[train[target]==1][quant_var]
    y = train[train[target]==0][quant_var]
    
    # Calculate the Mann-Whitney U test statistic and p-value
    stat, p = stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)
    
    # Print the results of the test
    print("Mann-Whitney Test:", f'stat = {stat}, p = {p}')


def plot_cat_by_target(train: pd.DataFrame, target: str, cat_var: str) -> plt.Axes:
    """
    Plot a bar chart of a categorical variable against a target variable, with a horizontal line indicating the overall rate of the target variable.

    Parameters:
    train (pd.DataFrame): The training dataset.
    target (str): The name of the target variable.
    cat_var (str): The name of the categorical variable.

    Returns:
    ax (plt.Axes): The plot object.
    """
    # Create a new figure with a size of 2x2 inches
    fig = plt.figure(figsize=(2,2))
    
    # Create a bar plot with the categorical variable on the x-axis and the target variable on the y-axis
    ax = sns.barplot(x=cat_var, y=target, data=train, alpha=.8)
    
    # Calculate the overall rate of the target variable and add a horizontal line to the plot at that level
    overall_rate = train[target].mean()
    ax.axhline(overall_rate, ls='--')
    
    # Return the plot object
    return ax

def baseline(target):
    """
    Calculates the baseline accuracy of a classification model that always predicts the most common value of the target variable.

    Parameters:
    target (pandas.Series): The target variable to be predicted.

    Returns:
    None
    """
    # Calculate the most common value of the target variable
    most_common = target.value_counts().idxmax()
    
    # Calculate the accuracy of a model that always predicts the most common value
    accuracy = (target == most_common).mean() * 100
    
    # Print the baseline accuracy as a percentage
    print(f'Baseline: {round(accuracy, 2)}% Accuracy')