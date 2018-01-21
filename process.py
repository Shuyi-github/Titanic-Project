#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 22:35:13 2018

@author: TANG
"""

import pandas as pd
import matplotlib.pyplot as plt
from setmissingval import set_missing_age, set_missing_embarkinfo

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.info()                     # information of train set 

train=set_missing_age(train)  #补全age缺失值。用randomforest做回归拟合
train=set_missing_embarkinfo(train)  #补全embarked。只缺了两个，补了占比做多的登陆信息

dummies_Embarked = pd.get_dummies(train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(train['Pclass'], prefix= 'pclass')
