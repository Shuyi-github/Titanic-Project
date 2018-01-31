#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:14:31 2018

@author: TANG
"""
import re as re
from sklearn.ensemble import RandomForestRegressor

def set_missing_age(df,test):
    #乘客分成已知年龄和未知年龄两部分
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    age_test=test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    unknown_testage=age_test[age_test.Age.isnull()].as_matrix()
    # y即目标年龄
    y = known_age[:, 0]
    # X即特征属性值
    X = known_age[:, 1:]
    regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=50)
    regr.fit(X, y)
    predictedAges = regr.predict(unknown_age[:, 1::])
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    #print(regr.feature_importances_)
    test_age=regr.predict(unknown_testage[:,1::])
    test.loc[ (test.Age.isnull()), 'Age' ] = test_age
    return df,test
    
def set_missing_embarkinfo(df):
    df['Embarked'] = df['Embarked'].fillna('S')
    return df

def set_missing_fare(df):
    Mean=df['Fare'].mean()
    df['Fare'] = df['Fare'].fillna(Mean)
    return df

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def get_cabin(cabin):
    if isinstance (cabin,str):
        cabin_num=re.search('(\w)\d+',cabin)
        if cabin_num:
            return cabin_num.group(1)
    return 0
    
    