#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:14:31 2018

@author: Tom
"""

from sklearn.ensemble import RandomForestRegressor

def set_missing_age(df):
    #乘客分成已知年龄和未知年龄两部分
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]
    regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=50)
    regr.fit(X, y)
    predictedAges = regr.predict(unknown_age[:, 1::])
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    #print(regr.feature_importances_)
    return df
    
def set_missing_embarkinfo(df):
    df['Embarked'] = df['Embarked'].fillna('S')
    return df