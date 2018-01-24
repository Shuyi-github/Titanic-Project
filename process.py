#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 22:35:13 2018

@author: TANG
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from setmissingval import set_missing_age, set_missing_embarkinfo

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
sub=pd.read_csv('gender_submission.csv')
full_data=[train,test]
train.info()                     # information of train set 

[train,test]=set_missing_age(train,test)  #补全age缺失值。用randomforest做回归拟合
train=set_missing_embarkinfo(train)  #补全embarked。只缺了两个，补了占比做多的登陆信息
test=set_missing_embarkinfo(test)
'''
 # Mapping Age
train.loc[ train['Age'] <= 16, 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age'] = 4
'''

for dataset in full_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

dummies_Embarked = pd.get_dummies(train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(train['Pclass'], prefix= 'pclass')
df_train=pd.concat([dummies_Embarked,dummies_Sex,dummies_Pclass,train['Age']],axis=1)
label=train['Survived']

dummies_Embarked = pd.get_dummies(test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(test['Pclass'], prefix= 'pclass')
df_test=pd.concat([dummies_Embarked,dummies_Sex,dummies_Pclass,test['Age']],axis=1)

lrg = LogisticRegression()
lrg.fit(df_train,label)
predictions=lrg.predict(df_test)
print(lrg.score(df_test,sub.loc[:,'Survived']))

result = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(), 'Survived':predictions})
result.to_csv('results.csv', header=True,index=False)