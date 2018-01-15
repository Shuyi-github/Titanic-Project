#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 20:20:10 2018

@author: Tom
"""

import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt




train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.info()                     # information of train set 

print(train.Survived.value_counts())

fig = plt.figure()
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

fig.set(alpha=0.2)  # 设定图表颜色alpha参数
Survived_0 = train.Pclass[train.Survived == 0].value_counts()
Survived_1 = train.Pclass[train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级") 
plt.ylabel(u"人数") 
plt.show()

#print(train.Age.value_counts())
age=train.Age.value_counts()
print(age.sort_index())
sur=train.Age[train.Survived == 1].value_counts()
mature=train.Age[(train.Age <= 30) & (train.Age >=18)]

print(sur.sort_index())
print(mature.sort_index())
print(train.Age)

df1=pd.DataFrame(train.Age)
df2=df1.dropna(axis=0,how='any') # 0: 对行进行操作; 1: 对列进行操作'any': 只要存在 NaN 就 drop 掉; 'all': 必须全部是 NaN 才 drop 
