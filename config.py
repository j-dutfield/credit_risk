#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:44:36 2023

@author: jamesdutfield

Config file contribution to util.py and credit_risk_project.ipynb
"""

import numpy as np

### custom encodings and column mapping for preproc transformations
payment_status_encodings = {"0":0.0,
                            'U':np.nan, # to be imputed to mean later
                            "1":1.0,
                            "2":2.0,
                            "3":3.0,
                            "4":4.0,
                            'D':5.0
                           }

gender_encodings = {"M":1.0,
                    "F":0.0} 

encode_date_cols = 'application_date'
encode_payment_status_cols = ['worst_paystatus_24m']
encode_ordinal_cols = ['gender']
impute_average_cols=['unsatisfied_ccjs', 'defaults', 'deceased', 'search_payday']
impute_zero_cols=['cc_value_3m', 'age_oldest_active_account']
impute_U_cols=['worst_paystatus_24m']
impute_100_cols=['months_since_default']
date_cols = ['dayofweek', 'dayofmonth', 'month', 'year']
feature_order = ['worst_paystatus_24m', 'gender', 'cc_value_3m', 'dayofmonth', 'unsatisfied_ccjs', 'defaults', 'search_payday', 'deceased', 'months_since_default', 'dayofweek', 'dayofmonth', 'month', 'year']
