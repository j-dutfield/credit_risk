#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:40:20 2023

@author: j-dutfield

Custom Functions for use in credit_risk_project.ipynb
"""

#imports
import pandas as pd
import xgboost as xgb
from config import gender_encodings, payment_status_encodings
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

# preproc
from sklearn.base import BaseEstimator, TransformerMixin

class DataframeFunctionTransformer():
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self
        
def EncodeDates(df, col='application_date'): # extract day of week, day of month, month and year
    dates = pd.to_datetime(df[col], dayfirst=True)
    df['dayofweek'] = dates.dt.dayofweek
    df['dayofmonth'] = dates.dt.day
    df['month'] = dates.dt.month
    df['year'] = dates.dt.year
    df=df.drop(columns=[col])
    return df     

class EncodePaymentStatusAndImputeMean(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.replace(payment_status_encodings)
        X = X.fillna(X.mean())
        return X
    
class EncodeGenderAndImputeMean(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.replace(gender_encodings)
        X = X.fillna(X.mean())
        return X

class ImputeZeros(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.fillna(0)
        return X

class ImputeMean(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.fillna(X.mean())
        return X

class ImputeHundred(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.fillna(100.0)
        return X
    
class columnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        return X.drop(self.columns,axis=1)

    def fit(self, X, y=None):
        return self 

class returnFloat(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.astype(float)
        return X
    
class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        print(X.shape)
        print(X)
        return X

    def fit(self, X, y=None, **fit_params):
        return self
    
def custom_model_eval(estimator, X, y):
    pred = estimator.predict(X)
    print(classification_report(y, pred))
    RocCurveDisplay.from_estimator(estimator, X, y)
    
def custom_learning_curve(results):
    # plot learning curves
    plt.plot(results['validation_1']['auc'], label='training')
    plt.plot(results['validation_0']['auc'], label='validation')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
def feature_importance(model, feature_order):
    for i in range(len(feature_order)):
        print('f'+str(i)+': '+feature_order[i])
    xgb.plot_importance(model[2])
    plt.figure(figsize = (16, 12))
    plt.show()