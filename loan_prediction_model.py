#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:24:28 2019

@author: omkar
"""
import pandas as pd
import loan_prediction_cleaning
from loan_prediction_cleaning import cleaning_loan
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier


file_name = 'loan_prediction_train.csv'
df, X, y = cleaning_loan(file_name)

predictors = X[['LoanAmount','TotalIncome_log','Credit_History']]
#predictors = X[:]

X_train , X_test, y_train, y_test = train_test_split(predictors, y, test_size = 0.2, stratify = y)


'''
Model
'''

classifier = xgb.XGBClassifier()
classifier.fit(X_train, y_train)

train_predictions = classifier.predict(X_train)
test_predictions = classifier.predict(X_test)

cm = confusion_matrix(y_test, test_predictions)
cv_results = cross_val_score(classifier, predictors, y, cv=5)
print((cm[0][1]+cm[1][1]) / sum(sum(cm)))
print(np.mean(cv_results))


#'''
#Saving results
#'''
#
#test_file = 'loan_prediction_test.txt'
#test_df, test_X, Loan_ID = cleaning_loan(test_file, train=False)
#
#final_prediction = classifier.predict(test_X[['LoanAmount','TotalIncome_log','Credit_History']])
#final_prediction = pd.DataFrame(final_prediction,columns = ['Loan_Status'],index=None)
#result = pd.concat([Loan_ID, final_prediction],axis = 1)
#result.to_csv('result1.csv', index = False)
