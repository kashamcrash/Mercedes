#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Create an ML algorithm that can accurately predict the time a car will spend on the test bench 
# based on the vehicle configuration
# Credentials - kasham1991@gmail.com / karan sharma


# Agenda
# 1. If for any column(s), the variance is equal to zero, then you need to remove those variable(s)
# 2. Check for null and unique values for test and train sets
# 3. Apply label encoder for categorical variables
# 4. Perform dimensaionlity reduction with PCA
# 5. Predict the test_df values using xgboost


# In[2]:


# Importing the required libraries
# Loading the train/test data
# The lowercase alphabets are categorical variables
import numpy as np
import pandas as pd

train = pd.read_csv('C://Datasets//MERCtrain.csv')
train.head()
# train.info()
# print('Size of training set')
# train.shape


# In[3]:


# Separating y column as this is for pediction output
y_train = train['y'].values
y_train


# In[4]:


# A lot of columns that have an X 
# Let's check for the same 
# 376 features with X
colums_x = [c for c in train.columns if 'X' in c]
# colums_x
print(len(colums_x))
print(train[colums_x].dtypes.value_counts())


# In[5]:


# Looking at the test datset for simiilar features
test = pd.read_csv('C://Datasets//MERCtest.csv')
test.head()
# train.info()
# print('Size of training set')
# train.shape


# In[6]:


# Creating the final dataset
# Removing unwanted columns (ID); y has been removed earlier
final_column = list(set(train.columns) - set(['ID', 'y']))

x_train = train[final_column]
# x_train
x_test = test[final_column]
# x_test


# In[7]:


# Searching for null values
# Creating a function for the same
# There are no missin values
def detect(df):
    if df.isnull().any().any():
        print("Yes")
    else:
        print("No")

detect(x_train)
detect(x_test)


# In[8]:


# Removal of columns with a variance of 0
# Column with a variance of 1 is irrelevant so we drop it

for column in final_column:
    check = len(np.unique(x_train[column]))
    if check == 1:
        x_train.drop(column, axis = 1) 
        x_test.drop(column, axis = 1)
    if check > 2: # Column is categorical; hence mapping to ordinal measure of value
        mapit = lambda x: sum([ord(digit) for digit in x])
        x_train[column] = x_train[column].apply(mapit)
        x_test[column] = x_test[column].apply(mapit)

x_train.head()


# In[9]:


# Performing dimensionality reduction with principal components analysis
from sklearn.decomposition import PCA
n_comp = 12
pca = PCA(n_components = n_comp, random_state = 42)
pca_result_train = pca.fit_transform(x_train)
pca_result_test = pca.transform(x_test)
# print(pca_result_train)
# print(pca_result_test)


# In[10]:


# ML Modeling with XGboost
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Splitting the data by 80/20
x_train, x_valid, y_train, y_valid = train_test_split(pca_result_train, y_train, test_size = 0.2, random_state = 42)


# In[11]:


# Building the final feature set
f_train = xgb.DMatrix(x_train, label = y_train)
f_valid = xgb.DMatrix(x_valid, label = y_valid)
f_test = xgb.DMatrix(x_test)
f_test = xgb.DMatrix(pca_result_test)


# In[12]:


# Setting the parameters for XGB
params = {}
params['objective'] = 'reg:linear'
params['eta'] = 0.02
params['max_depth'] = 4


# In[13]:


# Predicting the score
# Creating a function for the same

def scorer(m, w):
    labels = w.get_label()
    return 'r2', r2_score(labels, m)

final_set = [(f_train, 'train'), (f_valid, 'valid')]

P = xgb.train(params, f_train, 1000, final_set, early_stopping_rounds=50, feval=scorer, maximize=True, verbose_eval=10)


# In[14]:


# Predicting on test set
p_test = P.predict(f_test)
p_test


# In[15]:


Predicted_Data = pd.DataFrame()
Predicted_Data['y'] = p_test
Predicted_Data.head()


# In[16]:


# Thank You :) 

