---
title: "Data Science Nigeria 2019 Challenge 1: Insurance Prediction by Zindi.africa Attempt"
date: 2020-11-30
tags: [mnist, convolutional neural network, tensorflow, keras, digit recognizer, python]
header:
excerpt: "Digit Recognizer"
mathjax: "true"
---

In this article, I attempt the Data Science Nigeria 2019 Challenge 1: Insurance Prediction competition on Zindi.africa. In this competition, participants are required to predict whether a given building will be damaged or not in a given time period. Olusola Insurance Company offers insurance against building damage for reasons such fire, vandalism, flood, and storm. Olusola Insurance Company would like to know which buildings are likely to be damaged in a given time window, and the competition entails buidling a model to predict the probability of having at least one claim over the insured period for a building.

A description of the variables in the training and test sets can be found on https://zindi.africa/.

We begin by importing the necessary libraries and data.


```python
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics 

import matplotlib as plt

import statistics

# Import the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```

A brief examination of that data shows that some missing values are encoded as "  .". We replace these.


```python
# Replace all occurences of '   .' with None
train = train.replace({'   .': None})
```

Next, we move on to determining the frequency of NA values per variable in the training set.


```python
# Determine the frequency of missing values per variable in the training set
train.isnull().sum()/train.shape[0]
```




    Customer Id           0.000000
    YearOfObservation     0.000000
    Insured_Period        0.000000
    Residential           0.000000
    Building_Painted      0.000000
    Building_Fenced       0.000000
    Garden                0.000978
    Settlement            0.000000
    Building Dimension    0.014804
    Building_Type         0.000000
    Date_of_Occupancy     0.070950
    NumberOfWindows       0.495950
    Geo_Code              0.014246
    Claim                 0.000000
    dtype: float64



We see that NumberOfWindows has approximatley 50% NA values. We drop this variable.


```python
# Drop NumberOfWindows (50% NA)
train.drop("NumberOfWindows", axis=1, inplace=True)

train.isnull().values.ravel().sum()/train.shape[0]
```




    0.10097765363128491



10% of the observations in the training set have NA values for at least one variable after the removal of NumberOfWindows. We continue with a complete case analysis and note that there is opportunity to potentially improve on the test AUC by using an appropriate imputation model.

Next, drop all NA values and ensure all categorical variables are correctly encoded.


```python
# Drop the observations with NA entries
train.dropna(inplace=True)

# Ensure categorical variables are encoded as categorical
train.Residential = train.Residential.astype('category')
train.Building_Type = train.Building_Type.astype('category')
train.Claim = train.Claim.astype('category')
```

Quick testing shows that Geo_Code will have over 1000 categories. This is inappropriate for the model building stage. Therefore, we drop Geo_Code.


```python
# Drop Geo_Code as the categorical variable will have too many categories to be useable
train.drop("Geo_Code", axis=1, inplace=True)
```

Next, we create a feature that measures the number of years between the date of occupancy and the year of observation of each building (i.e the age of each building). We then drop YearOfObservation and Date_of_Occupancy from the training set.


```python
# Create a variabe that measures the age of each building (from first occupancy to year of observation of the policy)
train["BuildingAge"] = train.YearOfObservation.subtract(train.Date_of_Occupancy)

# Drop Year of YearOfObservation, Customer Id and Date_of_Occupancy
train.drop(["YearOfObservation", "Customer Id", "Date_of_Occupancy"], axis=1, inplace=True)
```

Next, we examine the case-control ratio.


```python
# Determine the case to control ratio
(train.Claim == 1).sum()/train.shape[0]
```




    0.23093382240562432



We see that there are 23% cases in the training set. This is optimal.

Next we create some exploratory data analysis plots for the remaining predictors in the training set.


```python
# Create boxplots for the numeric predictors
num_preds = train.select_dtypes(include=['int64','float64']).columns
for i in range(0, len(num_preds)):
    sns.catplot(x="Claim", y=num_preds[i], kind="box",
            data=train)
    
# Create frequency plots for the categorical predictors
cat_preds = train.select_dtypes(include=['category']).columns[train.select_dtypes(include=['category']).columns != "Claim"]
for i in range(0, len(cat_preds)):
    sns.catplot(y="Claim", hue=cat_preds[i], kind="count",
            palette="pastel", edgecolor=".6",
            data=train)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/insurance/box1.png" alt="box1">



<img src="{{ site.url }}{{ site.baseurl }}/images/insurance/box2.png" alt="box2">



<img src="{{ site.url }}{{ site.baseurl }}/images/insurance/box3.png" alt="box3">



<img src="{{ site.url }}{{ site.baseurl }}/images/insurance/freq1.png" alt="freq1">



<img src="{{ site.url }}{{ site.baseurl }}/images/insurance/freq2.png" alt="freq2">


On the whole, meaningful patterns are challenging to distiguish. However, the boxplots indicate the presence of outliers. Identifying and removing these outliers may lead to an increase in test AUC. We do not do this here but note it as an opportunity.

Next, we prepare the training data for model building.


```python
# Split the training data into a training and validation set
features = train.columns[train.columns != "Claim"]
response = "Claim"

# Prepare the data
X = pd.get_dummies(train[features])
y = train[response]

# Split the training set up into traning and validate (75-25)
X_train,X_validate,y_train,y_validate=train_test_split(X,y,test_size=0.25,random_state=0)
```

The baseline model is chosen to be the linear logistic regression model. We fit this model now without regularisation.


```python
# Fit a linear logistic regression model with no regularisation
linlogreg = LogisticRegression(penalty="none", solver="sag").fit(X_train, y_train)

# Coefficient matrix
estimates = np.append(linlogreg.intercept_, linlogreg.coef_[0])
coefs = pd.DataFrame({"Predictors":["Intercept"] + X_train.columns.tolist(), "Coefficients": estimates})
coefs
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predictors</th>
      <th>Coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Intercept</td>
      <td>-0.000584</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Insured_Period</td>
      <td>-0.000500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Building Dimension</td>
      <td>0.000048</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BuildingAge</td>
      <td>-0.018554</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Residential_0</td>
      <td>-0.000483</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Residential_1</td>
      <td>-0.000101</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Building_Painted_N</td>
      <td>-0.000183</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Building_Painted_V</td>
      <td>-0.000401</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Building_Fenced_N</td>
      <td>-0.000250</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Building_Fenced_V</td>
      <td>-0.000333</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Garden_O</td>
      <td>-0.000250</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Garden_V</td>
      <td>-0.000333</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Settlement_R</td>
      <td>-0.000250</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Settlement_U</td>
      <td>-0.000333</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Building_Type_1</td>
      <td>-0.000221</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Building_Type_2</td>
      <td>-0.000316</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Building_Type_3</td>
      <td>-0.000053</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Building_Type_4</td>
      <td>0.000006</td>
    </tr>
  </tbody>
</table>
</div>



We see above that BuildingAge is the variable with the largest (in absolute value) coefficient estimate. The negative sign implies that the younger the building, the higher the chance that the building will claim in the given time interval.

Next, we obtain some validation set metrics.


```python
# Predict
linlogreg_preds = linlogreg.predict(X_validate)

# Validation set accuracy rate
linlogreg_accuracy_validate = linlogreg.score(X_validate, y_validate)
linlogreg_accuracy_validate
```




    0.7518337408312958




```python
# Obtain a confusion matrix
cm = confusion_matrix(y_validate, linlogreg_preds)
cm
```




    array([[1200,   62],
           [ 344,   30]], dtype=int64)




```python
# Calculate validation AUC
linlogreg_predprob = linlogreg.predict_proba(X_validate)[:,1]
metrics.roc_auc_score(y_validate, linlogreg_predprob)
```




    0.5580830444841818



We see that the linear logistic regression model has a validation set accuracy of 75.06% and a validation set AUC of 0.56.

Next, we fit an L1-regularised logistic regression model.


```python
# Fit a linear legistic regression model wil L1 regularisation
# Select C using 5-fold cross-validation
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
linlogl1 = GridSearchCV(LogisticRegression(penalty='l1', solver='saga'), param_grid).fit(X_train, y_train)
linlogl1.best_params_
``` 
    {'C': 0.001}



We see that the optimal value of the cost parameter, C, is 0.001. We now fit the L1-regularised logistic regression model with this value of C, examine the coefficient estimates, and obtain the validation set performance metrics.


```python
linlogl1 = LogisticRegression(penalty="l1", C=0.001, solver='saga').fit(X_train, y_train)

estimates = np.append(linlogl1.intercept_, linlogl1.coef_[0])

# Coefficient matrix
coefs = pd.DataFrame({"Predictors":["Intercept"] + X_train.columns.tolist(), "Coefficients": estimates})
coefs
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predictors</th>
      <th>Coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Intercept</td>
      <td>-0.000356</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Insured_Period</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Building Dimension</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BuildingAge</td>
      <td>-0.013568</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Residential_0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Residential_1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Building_Painted_N</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Building_Painted_V</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Building_Fenced_N</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Building_Fenced_V</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Garden_O</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Garden_V</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Settlement_R</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Settlement_U</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Building_Type_1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Building_Type_2</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Building_Type_3</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Building_Type_4</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



We see that only BuildingAge has a non-zero parameter estimate.


```python
# Obtain a confusion matrix
cm = confusion_matrix(y_validate, linlogl1.predict(X_validate))
cm
```




    array([[1241,   21],
           [ 368,    6]], dtype=int64)




```python
# Validation set accuracy rate
linlogl1_accuracy_validate = linlogl1.score(X_validate, y_validate)
linlogl1_accuracy_validate
```




    0.7622249388753056




```python
# Validation AUC
linlogl1_predprob = linlogl1.predict_proba(X_validate)[:,1]
metrics.roc_auc_score(y_validate, linlogl1_predprob)
```




    0.5311988864123665



We see that the L1-regularised logistic regression model has a lower validation AUC than the ordinary linear logistic regression model.

Next, we fit and tune a boosted classification tree model.


```python
# Fit a boosted regression tree model
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalidate = xgb.DMatrix(X_validate, label=y_validate)


# Tune max_depth and min_child_weight
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc', cv=5)

gsearch1.fit(X_train, y_train, eval_metric='auc')
gsearch1.best_params_
```




    {'max_depth': 3, 'min_child_weight': 1}




```python
# Tune gamma
param_test2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc', cv=5)
gsearch2.fit(X_train, y_train, eval_metric='auc')
gsearch2.best_params_
```




    {'gamma': 0.4}




```python
# Tune subsample and colsample_bytree
param_test3 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc', cv=5)
gsearch3.fit(X_train, y_train, eval_metric='auc')
gsearch3.best_params_
```




    {'colsample_bytree': 0.6, 'subsample': 0.9}



Now, we use the optimal values of max_depth, min_child_weight, gamma, subsample, and colsample_tree to fit the final model.


```python
# Fit the optimal model and examine the features importance plot
xgb_mod = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=3,
 min_child_weight=1,
 gamma=0.4,
 subsample=0.6,
 colsample_bytree=0.9,
 objective= 'binary:logistic',
 seed=27)

# Fit the algorithm on the data
xgb_mod.fit(X=X_train, y=y_train, eval_metric='auc')

# Predict validation set: 
dtrain_predictions = xgb_mod.predict(X_validate)
dtrain_predprob = xgb_mod.predict_proba(X_validate)[:,1]
```


```python
# Print model report:
print("Accuracy : %.4g" % metrics.accuracy_score(y_validate, dtrain_predictions))
print("AUC Score (Validation): %f" % metrics.roc_auc_score(y_validate, dtrain_predprob))
```

    Accuracy : 0.7812
    AUC Score (Validation): 0.684713
    

We see that the optimal boosted classification tree model has a validation accuracy of 78.12% and a validation AUC of 0.6847. 

Next, we examine a feature importance plot of the optimal boosted classification tree model.


```python
feat_imp = pd.Series(xgb_mod.get_booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e3164aea88>




<img src="{{ site.url }}{{ site.baseurl }}/images/insurance/relfreq.png" alt="relfreq">


We see that Building Dimension, BuildingAge, and Insurance_Period are the three most important features in predicting Claim by a substantial margin.

Next, we train the optimal model on the full training set.


```python
# Train the model on the orignial training set
xgb_mod.fit(X=X, y=y, eval_metric='auc')
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=0.9, gamma=0.4,
                  learning_rate=0.01, max_delta_step=0, max_depth=3,
                  min_child_weight=1, missing=None, n_estimators=5000, n_jobs=1,
                  nthread=None, objective='binary:logistic', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=27,
                  silent=None, subsample=0.6, verbosity=1)



Now, we apply the same treatment to the test set as we did to the training set.


```python
# Apply the same treatment to the test set as the training set
# Replace all occurences of '   .' with None
test = test.replace({'   .': None})

# Drop NumberOfWindows (50% training NA)
test.drop("NumberOfWindows", axis=1, inplace=True)

# Determine the frequency of missing values per variable in the training set
test.isnull().sum()/test.shape[0]
```




    Customer Id           0.000000
    YearOfObservation     0.000000
    Insured_Period        0.000000
    Residential           0.000000
    Building_Painted      0.000000
    Building_Fenced       0.000000
    Garden                0.001303
    Settlement            0.000000
    Building Dimension    0.004236
    Building_Type         0.000000
    Date_of_Occupancy     0.237211
    Geo_Code              0.004236
    dtype: float64



Since we require predictions on the full test set, we simply use training mean and mode imputation to fill the missing values.


```python
# Use mean and modal imputation where appropriate. Use the training means and modes.
mode_garden = statistics.mode(train.Garden)
mean_buildingdim = statistics.mean(train["Building Dimension"])
mean_age = statistics.mean(train["BuildingAge"])

# Create test.BuildingAge
test["BuildingAge"] = test.YearOfObservation.subtract(test.Date_of_Occupancy)

test.Garden.fillna(mode_garden, inplace=True)
test["Building Dimension"].fillna(mean_buildingdim, inplace=True)
test.BuildingAge.fillna(mean_age, inplace=True)

# Ensure categorical variables are encoded as categorical
test.Residential = test.Residential.astype('category')
test.Building_Type = test.Building_Type.astype('category')

# Drop Geo_Code as the categorical variable will have too many categories to be useable
test.drop("Geo_Code", axis=1, inplace=True)

# Drop Year of YearOfObservation, Date_of_Occupancy
test.drop(["YearOfObservation", "Date_of_Occupancy"], axis=1, inplace=True)

# Prepare the data
X = pd.get_dummies(test[features])
```

Now, we obtain the prediction on the test set.


```python
# Predict test set: 
dtest_pred = xgb_mod.predict(X)
```

Zindi.africa indicated that our optimal boosted tree model had a test AUC of 0.5735.
