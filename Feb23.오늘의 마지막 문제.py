#Feb23.오늘의 마지막 문제
#%%
# -*- coding: utf-8 -*-
### 기본 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
[Step 1] 데이터 준비
'''

weight=[ 72, 72, 70, 43, 48, 54, 51, 52, 73, 45, 60, 62, 64, 47, 51, 74, 88,64, 56, 56  ] #독립변수 X
tall = [ 176, 172, 182, 160, 163, 165, 168, 163, 182, 148, 170, 166, 172, 169, 163, 170, 182, 174, 164, 160 ] #종속변수 y

dict_data = { 'weight' : [ 72, 72, 70, 43, 48, 54, 51, 52, 73, 45, 60, 62, 64, 47, 51, 74, 88, 64, 56, 56  ],
                  'tall' : [ 176, 172, 182, 160, 163, 165, 168, 163, 182, 148, 170, 166, 172, 169, 163, 170, 182, 174, 164, 160 ]   }

df = pd.DataFrame(dict_data) #DataFrame 으로 변환
print(df)

#%%
'''
[Step 2] 데이터 탐색
'''
# 데이터 자료형 확인
print(df.info())
print('\n')

#%%
# 데이터 통계 요약정보 확인
print(df.describe())
print('\n')
#%%
'''
[Step 3] 속성(feature 또는 variable) 선택
'''
# seaborn으로 산점도 그리기
fig = plt.figure(figsize=(10, 5)) #전체 그림판 가로 10, 세로 5로 잡아주고,
ax1 = fig.add_subplot(1, 2, 1) #첫번째 그림판 영역
ax2 = fig.add_subplot(1, 2, 2) #두번째 그림판 영역
sns.regplot(x='weight', y='tall', data=df, ax=ax1) # 회귀선 표시
sns.regplot(x='weight', y='tall', data=df, ax=ax2, fit_reg=False) #회귀선 미표시
#설명: seaborn의 regplot이 단순 회귀를 그려준다.
plt.show()
plt.close()
#%%
# seaborn 조인트 그래프 - 산점도, 히스토그램
sns.jointplot(x='weight', y='tall', data=df) # 회귀선 없음
sns.jointplot(x='weight', y='tall', kind='reg', data=df) # 회귀선 표시
plt.show()
plt.close()
#%%
# seaborn pariplot으로 두 변수 간의 모든 경우의 수 그리기
sns.pairplot(df)
plt.show()
plt.close()
#%%
'''
Step 4: 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''
# 속성(변수) 선택
X=df[['weight']] #독립 변수 X
y=df['tall'] #종속 변수 Y

# train data 와 test data로 구분(9:1 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, #독립 변수
 y, #종속 변수
 test_size=0.1, #검증 10%
 random_state=10) #랜덤 추출 값

print('train data 개수: ', len(X_train)) #18
print('test data 개수: ', len(X_test)) #2

#%%
'''
Step 5: 단순회귀분석 모형 - sklearn 사용
'''
# sklearn 라이브러리에서 선형회귀분석 모듈 가져오기
from sklearn.linear_model import LinearRegression

# 단순회귀분석 모형 객체 생성
lr = LinearRegression()

# train data를 가지고 모형 학습
lr.fit(X_train, y_train)

# 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱) 계산
r_square = lr.score(X_test, y_test)
print(r_square) #0.8657609246754262
print('\n')
#%%
# 회귀식의 기울기
print('기울기 a: ', lr.coef_)
print('\n')

# 회귀식의 y절편
print('y절편 b:', lr.intercept_)
print('\n')
#%%
# 모형에 전체 X 데이터를 입력하여 예측한 값 y_hat을 실제 값 y와 비교
y_hat = lr.predict(X)
plt.figure(figsize=(10, 5))
ax1 = sns.distplot(y, hist=False, label="y")
ax2 = sns.distplot(y_hat, hist=False, label="y_hat", ax=ax1)
plt.show()
plt.close()

