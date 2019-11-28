# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:03:43 2019

@author: orteg
"""

## Import Third Party 
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import eif
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

import os



os.path.dirname(os.path.realpath(__file__))
## Parameters
seed = 123

df = pd.read_csv('insurance_fraud_train.csv', index_col = 0)
df = df.drop(['PolicyNumber','RepairerDetailID'], axis=1)
cols_cat = [
    'IncidentMonth','IncidentWeekOfMonth', 'IncidentDayOfWeek', 'Make',
    'IncidentAddressType', 'DeclarationDayOfWeek','DeclarationWeekOfMonth', 'DeclarationMonth','DriverGender',
    'DriverMaritalStatus', 'Fault', 'PolicyType', 'VehicleCategory', 'VehiclePrice',
    'DriverRating', 'DaysPolicyAccident', 'DaysPolicyClaim','PastNumberOfClaims', 
    'AgeOfVehicle', 'AgeOfPolicyHolder','PoliceReportFiled', 'WitnessPresent', 
    'AgentType','NumberOfSuppliments', 'AddressChangeClaim', 'NumberOfCars',
    'IncidentYear', 'BasePolicy'
    ]
df[cols_cat] = df[cols_cat].astype('category')


## Divide X train into cat and continuous
X_train = df.drop('FraudFound', axis=1)
Xcat_train = X_train.select_dtypes('category')
Xcon_train = X_train.select_dtypes(exclude = 'category')

y_train = df['FraudFound']

## One HotEncoder
enc = OneHotEncoder(handle_unknown='ignore', sparse = False)
enc.fit_transform(Xcat_train)
enc.get_feature_names()
Xohe_train = pd.DataFrame(enc.fit_transform(Xcat_train), columns = enc.get_feature_names())

## Concatenate X_train from Categorical and Cont Variables
Xcon_train.reset_index(drop=True, inplace=True)
Xohe_train.reset_index(drop=True, inplace=True)
X_train = pd.concat([Xcon_train, Xohe_train], axis = 1)


### EXPORT ####




