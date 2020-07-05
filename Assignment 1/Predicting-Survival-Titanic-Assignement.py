#!/usr/bin/env python
# coding: utf-8

# ## Predicting Survival on the Titanic
# 
# ### History
# Perhaps one of the most infamous shipwrecks in history, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 people on board. Interestingly, by analysing the probability of survival based on few attributes like gender, age, and social status, we can make very accurate predictions on which passengers would survive. Some groups of people were more likely to survive than others, such as women, children, and the upper-class. Therefore, we can learn about the society priorities and privileges at the time.
# 
# ### Assignment:
# 
# Build a Machine Learning Pipeline, to engineer the features in the data set and predict who is more likely to Survive the catastrophe.
# 
# Follow the Jupyter notebook below, and complete the missing bits of code, to achieve each one of the pipeline steps.

# In[1]:


import re

# to handle datasets
import pandas as pd
import numpy as np

# for visualization
import matplotlib.pyplot as plt

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

# to persist the model and the scaler
import joblib

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# ## Prepare the data set

# In[206]:


# load the data - it is available open source and online

data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')

# display data
data.head()


# In[207]:


# replace interrogation marks by NaN values

data = data.replace('?', np.nan)


# In[208]:


# retain only the first cabin if more than
# 1 are available per passenger

def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan
    
data['cabin'] = data['cabin'].apply(get_first_cabin)


# In[209]:


# extracts the title (Mr, Ms, etc) from the name variable

def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
    
data['title'] = data['name'].apply(get_title)


# In[210]:


# cast numerical variables as floats

data['fare'] = data['fare'].astype('float')
data['age'] = data['age'].astype('float')


# In[211]:


# drop unnecessary variables

data.drop(labels=['name','ticket', 'boat', 'body','home.dest'], axis=1, inplace=True)

# display data
data.head()


# In[212]:


# save the data set

data.to_csv('titanic.csv', index=False)


# ## Data Exploration
# 
# ### Find numerical and categorical variables

# In[213]:


target = 'survived'


# In[214]:


data.columns


# In[215]:


data.title.unique()


# In[216]:


data.dtypes


# In[217]:


data['sex'].dtypes


# In[224]:


vars_num = [var for var in data.columns if data[var].dtypes != 'O'  and var not in [target]]

vars_cat = [var for var in data.columns if data[var].dtypes == 'O' and var not in vars_num + [target]]

print('Number of numerical variables: {}'.format(len(vars_num)))
print('Number of categorical variables: {}'.format(len(vars_cat)))


# In[225]:


vars_cat


# In[226]:


data.title.value_counts(dropna=False)


# ### Find missing values in variables

# In[230]:


# first in numerical variables
vars_num_with_na = [
    var for var in vars_num
    if data[var].isnull().sum() > 0
]
vars_num_with_na
# print percentage of missing values per variable
data[vars_num].isnull().mean()


# In[231]:


# now in categorical variables
vars_cat_with_na = [
    var for var in vars_cat
    if data[var].isnull().sum() > 0
]
vars_cat_with_na

# print percentage of missing values per variable
data[vars_cat_with_na].isnull().mean()


# ### Determine cardinality of categorical variables

# In[232]:


data[vars_cat].nunique()


# ### Determine the distribution of numerical variables

# In[241]:


def analyse_continuous(df, var):
    df = df.copy()
    df[var].hist(bins=30)
    plt.ylabel('Number of passengers')
    plt.xlabel(var)
    plt.title(var)
    plt.show()


for var in vars_num:
    analyse_continuous(data, var)


# ## Separate data into train and test
# 
# Use the code below for reproducibility. Don't change it.

# In[243]:


X_train, X_test, y_train, y_test = train_test_split(
    data.drop('survived', axis=1),  # predictors
    data['survived'],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape


# ## Feature Engineering
# 
# ### Extract only the letter (and drop the number) from the variable Cabin

# In[244]:


data['cabin'].unique()

def extract_letter(cabin_str):
    if isinstance(cabin_str, str):
        return cabin_str[0]
    else:
        return np.nan

X_train['cabin'] = X_train['cabin'].apply(extract_letter)
X_test['cabin'] = X_test['cabin'].apply(extract_letter)


# In[245]:


X_train.cabin.unique()


# ### Fill in Missing data in numerical variables:
# 
# - Add a binary missing indicator
# - Fill NA in original variable with the median

# In[246]:


for var in vars_num_with_na:

    # calculate the mode using the train set
    mode_val = X_train[var].median()

    # add binary missing indicator (in train and test)
    X_train[var+'_NA'] = np.where(X_train[var].isnull(), 1, 0)
    X_test[var+'_NA'] = np.where(X_test[var].isnull(), 1, 0)

    # replace missing values by the mode
    # (in train and test)
    X_train[var] = X_train[var].fillna(mode_val)
    X_test[var] = X_test[var].fillna(mode_val)

# check that we have no more missing values in the engineered variables
X_train[vars_num_with_na].isnull().sum()


# ### Replace Missing data in categorical variables with the string **Missing**

# In[247]:


X_train[vars_cat_with_na] = X_train[vars_cat_with_na].fillna('Missing')
X_test[vars_cat_with_na] = X_test[vars_cat_with_na].fillna('Missing')


# In[249]:


X_train.isnull().sum()


# In[250]:


X_test.isnull().sum()


# ### Remove rare labels in categorical variables
# 
# - remove labels present in less than 5 % of the passengers

# In[251]:


def find_frequent_labels(df, var, target_col, rare_perc):
    
    # function finds the labels that are shared by more than
    # a certain % of the houses in the dataset

    df = df.copy()
    tmp = df.groupby(var)[var].count() / len(df)

    return tmp[tmp > rare_perc].index


for var in vars_cat:
    
    # find the frequent categories
    frequent_ls = find_frequent_labels(X_train, var, y_train, 0.05)
    
    
    # replace rare categories by the string "Rare"
    X_train[var] = np.where(X_train[var].isin(
        frequent_ls), X_train[var], 'Rare')
    
    X_test[var] = np.where(X_test[var].isin(
        frequent_ls), X_test[var], 'Rare')


# In[252]:


X_train[vars_cat].nunique()


# In[253]:


X_test[vars_cat].nunique()


# ### Perform one hot encoding of categorical variables into k-1 binary variables
# 
# - k-1, means that if the variable contains 9 different categories, we create 8 different binary variables
# - Remember to drop the original categorical variable (the one with the strings) after the encoding

# In[254]:


ohe_cat_train = pd.get_dummies(X_train[vars_cat], drop_first=True)
ohe_cat_test = pd.get_dummies(X_test[vars_cat], drop_first=True)
X_train = pd.concat([X_train, ohe_cat_train], axis=1)
X_test = pd.concat([X_test, ohe_cat_test], axis=1)


# In[255]:


X_train.drop(columns=vars_cat, inplace=True)
X_test.drop(columns=vars_cat, inplace=True)


# In[257]:


len(X_train.columns)


# In[258]:


set(X_train.columns) - set(X_test.columns)


# In[259]:


X_test['embarked_Rare'] = np.zeros((len(X_test)))


# In[260]:


len(X_test.columns)


# ### Scale the variables
# 
# - Use the standard scaler from Scikit-learn

# In[261]:


# create scaler
scaler = StandardScaler()

#  fit  the scaler to the train set
scaler.fit(X_train) 

# transform the train and test set
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


# ## Train the Logistic Regression model
# 
# - Set the regularization parameter to 0.0005
# - Set the seed to 0

# In[263]:


lr = LogisticRegression(random_state=0, C=0.0005)


# In[264]:


lr.fit(X_train, y_train)


# ## Make predictions and evaluate model performance
# 
# Determine:
# - roc-auc
# - accuracy
# 
# **Important, remember that to determine the accuracy, you need the outcome 0, 1, referring to survived or not. But to determine the roc-auc you need the probability of survival.**

# In[266]:


# make predictions for test set
class_ = lr.predict(X_train)
pred = lr.predict_proba(X_train)[:,1]

# determine mse and rmse
print('train roc-auc: {}'.format(roc_auc_score(y_train, pred)))
print('train accuracy: {}'.format(accuracy_score(y_train, class_)))
print()

# make predictions for test set
class_ = lr.predict(X_test)
pred = lr.predict_proba(X_test)[:,1]

# determine mse and rmse
print('test roc-auc: {}'.format(roc_auc_score(y_test, pred)))
print('test accuracy: {}'.format(accuracy_score(y_test, class_)))
print()


# That's it! Well done
# 
# **Keep this code safe, as we will use this notebook later on, to build production code, in our next assignement!!**

# In[ ]:




