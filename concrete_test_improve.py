#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:28:33 2021

@author: Sora
"""

# -*- coding: utf-8 -*-

### 기본 라이브러리 불러오기
import pandas as pd

'''
[Step 1] 데이터 준비/ 기본 설정
'''
# load_dataset 함수를 사용하여 데이터프레임으로 변환
df = pd.read_csv("/Users/macbook/data/concrete.csv")

#  IPython 디스플레이 설정 - 출력할 열의 개수 한도 늘리기
pd.set_option('display.max_columns', 15)
print(df.head())
#%%
#결측치확인
print(df.isnull().sum())



#%%
'''
[Step 4] 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''
# 속성(변수) 선택
X=df[['cement', 'slag','ash','water','superplastic','coarseagg','fineagg','age']]  #독립 변수 X
y=df['strength']

print(X.head())

#%%


# 설명 변수 데이터를 표준화
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X)

#%%
# train data 와 test data로 구분(8:2 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10) 

print('train data 개수: ', X_train.shape)
print('test data 개수: ', X_test.shape)
print('\n')

#%%
'''
[Step 5] 신경망 모형 - sklearn 사용
'''
from sklearn.neural_network import MLPRegressor

# 3.neural network 모델 적합
clf = MLPRegressor(random_state=0, max_iter=500)
#%%
'''
성능개선 추가
'''
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV #grid search 기능을 사용할 수 있음

param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
             (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
             ]
        }
       ]
 

clf = GridSearchCV(MLPRegressor(), param_grid, cv=3, n_jobs = -1, verbose = 2 )
#%%

# train data를 가지고 모형 학습
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(clf.best_params_) #최고의 하이퍼-파리미터의 조합을 출력해준다.

'''
[Step 7] test data를 가지고 y_hat을 예측
'''
# test data를 가지고 y_hat을 예측 (분류) 
y_hat = clf.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])
print('\n')
#%%
'''
[Step 8] 모형 성능 평가 - 상관관계 계산
'''
import scipy.stats as stats
svm_report =  stats.pearsonr(y_test, y_hat)       #상관관계 확인하는 코드     
print(svm_report)

#설명: (상관계수, p-value 값: 0.05 이하면 유의미한 값)
#(0.9042961287774728, 2.385285961523753e-77)


