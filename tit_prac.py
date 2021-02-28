#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 19:14:53 2021

@author: Sora
"""

import pandas as pd
import numpy as np
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

train = pd.read_csv("/Users/macbook/data/train.csv", engine='python', encoding='CP949', index_col='PassengerId')
test = pd.read_csv("/Users/macbook/data/test.csv", engine='python', encoding='CP949', index_col='PassengerId')
submission = pd.read_csv("/Users/macbook/data/gender_submission.csv", engine='python', encoding='CP949', index_col='PassengerId')


#print(train.shape, test.shape, submission.shape) #(891, 11) (418, 10) (418, 1)

#sns.countplot(train['Survived'])
#train['Survived'].value_counts()

print(train.isnull().sum()) #결측치 - Age 177, Cabin   687, Embarked    2
print(test.isnull().sum()) #결측치 - Age   86, Fare  1, Cabin   327


#%%
#Cabin 지우기
train=train.drop(columns='Cabin')
test=test.drop(columns='Cabin')

print(train.isnull().sum()) #결측치 - Age 177,  Embarked    2
print(test.isnull().sum()) #결측치 - Age   86, Fare  1
#%%
#Age 지우기
train=train.drop(columns='Age')
test=test.drop(columns='Age')

print(train.isnull().sum()) #결측치 -  Embarked    2
print(test.isnull().sum()) #결측치 - Fare  1

#%%
#sns.countplot(data=train, x='Sex', hue='Survived')

#인코딩 진행 (문자x, 0과 1 숫자로)
train.loc[train['Sex']=='male', 'Sex']=0
train.loc[train['Sex']=='female','Sex']=1
test.loc[test['Sex']=='male','Sex']=0
test.loc[test['Sex']=='female','Sex']=1


#%%
#sns.countplot(data=train, x='Pclass', hue='Survived') #객실등급에 따른 사망자 수
train['Pclass_3']=(train['Pclass']==3)
train['Pclass_2']=(train['Pclass']==2)
train['Pclass_1']=(train['Pclass']==1)

test['Pclass_3']=(test['Pclass']==3)
test['Pclass_2']=(test['Pclass']==2)
test['Pclass_1']=(test['Pclass']==1)


train=train.drop(columns='Pclass')
test=test.drop(columns='Pclass')

#%%
#아까 남은 결측치 또 지우기
train['Embarked'].fillna('S', inplace = True)
train.loc[train['Embarked'].isnull(), 'Embarked']=0
test.loc[test['Fare'].isnull(),'Fare']=0


print(train.isnull().sum()) #결측치 -  0
print(test.isnull().sum()) #결측치 - 0

#%%


#%%
train['FamilySize']=train['SibSp']+train['Parch']+1
test['FamilySize']=test['SibSp']+test['Parch']+1

figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(18,6)
sns.countplot(data=train, x='SibSp', hue='Survived', ax=ax1)
sns.countplot(data=train, x='Parch', hue='Survived', ax=ax2)
sns.countplot(data=train, x='FamilySize',hue='Survived', ax=ax3)
#%%
train['Single']=train['FamilySize']==1
train['Nuclear']=(2<=train['FamilySize']) & (train['FamilySize']<=4)
train['Big']=train['FamilySize']>=5

test['Single']=test['FamilySize']==1
test['Nuclear']=(2<=test['FamilySize']) & (test['FamilySize']<=4)
test['Big']=test['FamilySize']>=5

figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(18,6)
sns.countplot(data=train, x='Single', hue='Survived', ax=ax1)
sns.countplot(data=train, x='Nuclear', hue='Survived', ax=ax2)
sns.countplot(data=train, x='Big',hue='Survived', ax=ax3) 
#%%
train=train.drop(columns=['Single','Big','SibSp','Parch','FamilySize'])
test=test.drop(columns=['Single','Big','SibSp','Parch','FamilySize'])
#%%
train['EmbarkedC']=train['Embarked']=='C'
train['EmbarkedS']=train['Embarked']=='S'
train['EmbarkedQ']=train['Embarked']=='Q'
test['EmbarkedC']=test['Embarked']=='C'
test['EmbarkedS']=test['Embarked']=='S'
test['EmbarkedQ']=test['Embarked']=='Q'

#%%
#이름 부분 만져주기
train['Name']=train['Name'].str.split(', ').str[1].str.split('. ').str[0]
test['Name']=test['Name'].str.split(', ').str[1].str.split('. ').str[0]

sns.countplot(data=train, x='Name', hue='Survived')

#%%
train['Master']=(train['Name']=='Master')
test['Master']=(test['Name']=='Master')

train=train.drop(columns='Name')
test=test.drop(columns='Name')

train=train.drop(columns='Ticket')
test=test.drop(columns='Ticket')

#%%
## 필요한 머신러닝 패키지들을 불러옵니다
from sklearn.linear_model import LogisticRegression # logistic regression
from sklearn import svm # support vector machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.tree import DecisionTreeClassifier # Decision Tree

from sklearn.model_selection import train_test_split # training and testing data split 
from sklearn import metrics # accuracy measure
from sklearn.metrics import confusion_matrix # confusion matrix

Ytrain=train['Survived']
feature_names=list(test)
Xtrain=train[feature_names]
Xtest=test[feature_names]

print(Xtrain.shape, Ytrain.shape, Xtest.shape) #(891, 11) (891,) (418, 11)
print(Xtrain.head())
#%%

model=DecisionTreeClassifier(max_depth=8, random_state=18)
# random_state is an arbitrary number.
print(model.fit(Xtrain, Ytrain))
#%%
predictions=model.predict(Xtest)
submission['Survived']=predictions
submission.to_csv('Result.csv')
submission.head()
#%%
model = RandomForestClassifier(n_estimators=100)
model.fit(Xtrain, Ytrain)
prediction7 = model.predict(Xtest)
print('The accuracy of the Random Forests is ', metrics.accuracy_score(prediction7, Ytrain))



