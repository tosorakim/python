### 기본 라이브러리 불러오기
import pandas as pd
import numpy as np
import statsmodels.api as sm #회귀분석을 위해 필요
import statsmodels.formula.api as smf #회귀분석을 위해 필요
from sklearn.preprocessing import StandardScaler #표준화를 위해 필요

'''
[Step 1] 데이터 준비
'''

df = pd.read_csv("/Users/macbook/data/crab.csv", engine='python', encoding='CP949')

#Apply 함수로 파생변수 추가하기
df['bmi30'] = df['bmi'].apply(func_1)
print(df)
#%%
'''
[Step 2] 모델링
'''
model = smf.ols(formula = 'expenses ~ age + sex + bmi + children + smoker + region + bmi30 + bmi30 * smoker', data = df)
result = model.fit()  # 모델 훈련
print( result.summary() ) #R-squared: 0.864