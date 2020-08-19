---
title: "Exploratory Data Analysis Functions"
date: 2020-08-19
tags: [exploratory data analysis, data science, eda]
header:
excerpt: "EDA, Exploratory Data Analysis, Data Science"
mathjax: "true"
---

# Exploratory Data Analysis

Although often a tedious part of the machine learning process, exploratory data analysis (EDA) is one of the most important initial steps in obtaining a useful model or extracting valuable insights. Simply, EDA can be thought of as attempting to learn more about your raw data. We typically start by considering each variable independently and look to determine, among other things, the type of the variable (numeric or categorical), the distribution of NA values, etc. I usually start with spliting variables up by type, and then analysing them further with these two nifty functions I composed below. Let me know if there any improvements I can make!

Let's begin by loading the relevant modules that my functons make use of. I have also made some cosmetic adjustments to how plots will be produced by my functions.


```python
import pandas as pd	
import seaborn as sns	
import matplotlib.pyplot as plt	

sns.set(style="ticks", color_codes=True)	
plt.style.use('ggplot')	
```

## Numeric variables
The function given below is used for analysisng numeric variables in an attempt to understand some of the fundamental characteristics these variables possess. This will help us with data cleaning, and variable selection in the model building stage.


```python
def EDANV(var, df):	
    """	
    A function to analyze numeric variables.	
    Parameters	
    ----------	
    var : Numeric variable name as a string.	
    df : Dataframe the categorical variable belongs to.	
    Returns	
    -------	
    Histogram of var with a rugplot.	
    Boxplot of var.	
    type : The variable type.	
    description : A pandas Series including the mean, std, 5 number summary, and 	
                  the number of NA values of the numeric variable.	
    """	
    # Obtain the numeric variable	
    x = df[var]	

    ty = x.dtype.name	

    # Stop if the input variable is of the incorrect type	
    if(ty != "float64"):	
        err = "Variable must be numeric"	
        return(err)	

    # Plot a histogram of var	
    plt.figure()	
    sns.distplot(x, kde=False, rug=True)	

    # Plot a boxplot of var	
    plt.figure()	
    sns.boxplot(x=var, data=df, palette="pastel")	

    # Obtain some summary statstics of var	
    desc = x.describe()	
    desc = desc.append(pd.Series({"sum_na":sum(x.isnull())}))	

    # Output	
    out = {"type": ty , "description":desc}	

    return(out)
```

## Categorical variables
Next, we have a function that can be used to explore categorical variables.


```python
def EDACV(var, df):	
    """	
    A function to analyze categorical variables.	
    Parameters	
    ----------	
    var : Categorical variable name as a string.	
    df : Dataframe the categorical variable belongs to.	
    Returns	
    -------	
    Horizonal barplot of var.	
    type : The variable type.	
    freq_dist : The frequency distribution of the categorical variable	
    description : A pandas Series including the number of categories as well	
                  as the number of NA values of the categorical variable.	
    """	

    # Obtain the categorical variable	
    x = df[var]	

    ty = x.dtype.name	

    # Stop if the input variable is of the incorrect type	
    if(ty != "object"):	
        err = "Variable must be categorical"	
        return(err)	

    # Plot a horizontal barplot of var	
    sns.catplot(y=var, kind="count",	
            palette="pastel", edgecolor=".6",	
            data=df);	

    # Obtain a description of var	
    desc = x.describe()	
    desc = desc.append(pd.Series({"sum_na":sum(x.isnull())}))	

    # Obtain the frequency distributon of var	
    freq = x.value_counts()	

    # Output	
    out = {"type": ty, "freq_dist" : freq, "description":desc}	

    return(out)	
```

I hope you can make use of these functions in assisting with one of the most important steps in the data science workflow! 

All the best,

Sam
