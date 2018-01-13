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
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = train.Pclass[train.Survived == 0].value_counts()
Survived_1 = train.Pclass[train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级") 
plt.ylabel(u"人数") 
plt.show()



'''
labels=np.loadtxt('digits4000_digits_labels.txt')
testset=np.loadtxt('digits4000_testset.txt')

challengelabel=np.loadtxt('cdigits_digits_labels.txt')
challengedigits_vec=np.loadtxt('cdigits_digits_vec.txt')

print("hello,python")
digits=np.asmatrix(digits_vec)
'''