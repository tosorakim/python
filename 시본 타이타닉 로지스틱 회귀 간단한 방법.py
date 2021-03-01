#!/usr/bin/env python
# coding: utf-8

# ### [Step 1] 데이터 준비/ 기본 설정

# In[65]:


# -*- coding: utf-8 -*-

### 기본 라이브러리 불러오기
import pandas as pd
import seaborn as sns

'''
[Step 1] 데이터 준비/ 기본 설정
'''

# load_dataset 함수를 사용하여 데이터프레임으로 변환
df = sns.load_dataset('titanic')

#  IPython 디스플레이 설정 - 출력할 열의 개수 한도 늘리기
pd.set_option('display.max_columns', 15)
df.head()


# ### 1. 데이터셋 설명
# 
# - PassengerId : 승객 번호
# - Survived : 생존여부(1: 생존, 0 : 사망)
# - Pclass : 승선권 클래스(1 : 1st, 2 : 2nd ,3 : 3rd)
# - Name : 승객 이름
# - Sex : 승객 성별
# - Age : 승객 나이 
# - SibSp : 동반한 형제자매, 배우자 수
# - Patch : 동반한 부모, 자식 수
# - Ticket : 티켓의 고유 넘버
# - Fare 티켓의 요금
# - Cabin : 객실 번호
# - Embarked : 승선한 항구명(C : Cherbourg, Q : Queenstown, S : Southampton)

# ### 결측치 확인

# In[66]:


df.isnull().sum()


# ### [Step 2] 데이터 탐색/ 전처리

# In[67]:


'''
[Step 2] 데이터 탐색/ 전처리
'''

# NaN값이 많은 deck 열을 삭제, embarked와 내용이 겹치는 embark_town 열을 삭제
rdf = df.drop(['deck', 'embark_town'], axis=1)  
rdf.head()


# In[68]:


# age 열에 나이 데이터가 없는 모든 행을 삭제 - age 열(891개 중 177개의 NaN 값)
rdf = rdf.dropna(subset=['age'], how='any', axis=0)  
rdf.head()


# In[69]:


# embarked 열의 NaN값을 승선도시 중에서 가장 많이 출현한 값으로 치환하기
most_freq = rdf['embarked'].value_counts(dropna=True).idxmax()   
rdf['embarked'].fillna(most_freq, inplace=True)
rdf.head()


# ### [Step 3] 분석에 사용할 속성을 선택 및 원핫 인코딩

# In[70]:


'''
[Step 3] 분석에 사용할 속성을 선택
'''

# 분석에 활용할 열(속성)을 선택 
ndf = rdf[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]
ndf.head()


# In[71]:


# 원핫인코딩 - 범주형 데이터를 모형이 인식할 수 있도록 숫자형으로 변환
onehot_sex = pd.get_dummies(ndf['sex'])
ndf = pd.concat([ndf, onehot_sex], axis=1)

onehot_embarked = pd.get_dummies(ndf['embarked'], prefix='town')
ndf = pd.concat([ndf, onehot_embarked], axis=1)

ndf.drop(['sex', 'embarked'], axis=1, inplace=True)
ndf.head()


# ### [Step 4] 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)

# In[72]:


'''
[Step 4] 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''

# 속성(변수) 선택
X=ndf[['pclass', 'age', 'sibsp', 'parch', 'female', 'male', 
       'town_C', 'town_Q', 'town_S']]  #독립 변수 X
y=ndf['survived']                      #종속 변수 Y


# In[73]:


X.head()


# In[74]:


# 설명 변수 데이터를 정규화(normalization)
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X)


# In[75]:


# train data 와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10) 

print('train data 개수: ', X_train.shape)
print('test data 개수: ', X_test.shape)
print('\n')


# ### [Step 5] 로지스틱 분류 모형 - sklearn 사용

# In[76]:


'''
[Step 5] 로지스틱 분류 모형 - sklearn 사용
'''

# sklearn 라이브러리에서 로지스틱 분류 모형 가져오기
from sklearn.linear_model import LogisticRegression

# 모형 객체 생성 (kernel='rbf' 적용)
logi_model = LogisticRegression()
logi_model
# 설명: 적절한 매개변수 C값과 gamma 값을 찾는게 중요한다 
# C 가 너무 크면 훈련 데이터는 잘 분류하지만 오버피팅이 발생하게 된다.
# gamma 매개변수는 결정경계의 곡률을 조정하는 매개변수인데
# gamma 가 너무 크면 훈련 데이터는 잘 분류하지만 오버피팅이 발생 할수
# 있습니다.


# ### [Step 6] train data를 가지고 모형 학습

# In[77]:


# train data를 가지고 모형 학습
logi_model.fit(X_train, y_train)   


# ###  [Step 7] test data를 가지고 y_hat을 예측 (분류) 

# In[78]:


# test data를 가지고 y_hat을 예측 (분류) 
y_hat = logi_model.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])
print('\n')


# ###  [Step 8] 모형 성능 평가 - Confusion Matrix 계산

# In[79]:


# 모형 성능 평가 - Confusion Matrix 계산
from sklearn import metrics 
svm_matrix = metrics.confusion_matrix(y_test, y_hat)  
print(svm_matrix)
print('\n')


# In[80]:


# 모형 성능 평가 - 평가지표 계산
svm_report = metrics.classification_report(y_test, y_hat)            
print(svm_report)


# ### [step9] 모델 성능 개선

# In[52]:


df.head()


# In[82]:


mask = ( df.age < 10 ) | ( df.sex=='female')
mask.astype(int) # True 를 1 로 변경하고 False 를 0 으로 변경
df['women_child'] = mask.astype(int)
df

