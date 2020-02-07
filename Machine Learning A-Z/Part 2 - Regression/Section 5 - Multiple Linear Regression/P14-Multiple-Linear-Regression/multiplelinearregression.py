# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:35:40 2019

@author: Asus
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Imoprting the Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


#Avoiding the Dummy Variable Trap(Will be automatically taken care of)
X = X[:, 1:]

 #Splitting the Dataset into test and train
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression();
regressor.fit(X_train,y_train)

#Predicting the test results
y_pred = regressor.predict(X_test)

#Backward Elimination preparation
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

#Now creating the optimal matrix for Backward Elimination
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()














