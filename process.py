#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 22:35:13 2018

@author: TANG
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import sklearn.preprocessing as preprocessing
from setmissingval import set_missing_age, set_missing_embarkinfo, set_missing_fare,get_title,get_cabin

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
full_data=[train,test]
train.info()                     # information of train set 

[train,test]=set_missing_age(train,test)  #补全age缺失值。用randomforest做回归拟合
train=set_missing_embarkinfo(train)  #补全embarked。只缺了两个，补了占比做多的登陆信息
test=set_missing_embarkinfo(test)
test=set_missing_fare(test)

for dataset in full_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']= 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    dataset['Cabin_num']=dataset['Cabin']
    dataset.loc[ dataset['Cabin_num'].isnull(), 'Cabin_num'] = 0
    dataset['Cabin_num'] = dataset['Cabin'].apply(get_cabin)
    
    dataset.loc[dataset['Cabin_num']=='A','Cabin_num']= 1
    dataset.loc[dataset['Cabin_num']=='B','Cabin_num']= 2
    dataset.loc[dataset['Cabin_num']=='C','Cabin_num']= 3
    dataset.loc[dataset['Cabin_num']=='D','Cabin_num']= 4
    dataset.loc[dataset['Cabin_num']=='E','Cabin_num']= 5
    dataset.loc[dataset['Cabin_num']=='F','Cabin_num']= 6
    dataset.loc[dataset['Cabin_num']=='G','Cabin_num']= 7
    dataset['class_cabin']=dataset['Cabin_num']+dataset['Pclass']*10
    dataset.loc[dataset['Cabin_num']== 0,'class_cabin']= 0
    dataset['class_cabin']=dataset['class_cabin']/dataset['class_cabin'].max()
    
    dataset.loc[ dataset['Cabin'].notnull(), 'Cabin'] = 1
    dataset.loc[ dataset['Cabin'].isnull(), 'Cabin'] = 0
    
    dataset['Title'] = dataset['Name'].apply(get_title)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)    
    dataset['Title'] = dataset['Title'].fillna(0)
    
print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


fsize_train=pd.DataFrame({'Fsize':train.SibSp+train.Parch}) 
dummies_Embarked = pd.get_dummies(train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(train['Pclass'], prefix= 'pclass')
df_train=pd.concat([dummies_Embarked,dummies_Sex,dummies_Pclass,train['Age'],train['Fare'],fsize_train,train['Title'],train['class_cabin']],axis=1)
label=train['Survived']


fsize_test=pd.DataFrame({'Fsize':test.SibSp+test.Parch})
dummies_Embarked = pd.get_dummies(test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(test['Pclass'], prefix= 'pclass')
df_test=pd.concat([dummies_Embarked,dummies_Sex,dummies_Pclass,test['Age'],test['Fare'],fsize_test,test['Title'],test['class_cabin']],axis=1)

lrg = LogisticRegression()
lrg.fit(df_train,label)
predictions=lrg.predict(df_test)

X = df_train
y = train['Survived']
print(cross_val_score(lrg, X, y, cv=5).mean())

result = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(), 'Survived':predictions})
result.to_csv('results.csv', header=True,index=False)