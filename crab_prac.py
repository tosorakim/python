#1. data 불러오기
import pandas as pd

df = pd.read_csv("/Users/macbook/data/crab.csv", engine='python', encoding='CP949')
#print(df)
#print(df.head())
print(df.y.unique()) #[1 0]

#%%
#2. 다중회귀분석을 하고, 종속변수에 영향을 주는 독립변수들이 무엇인지 확인하기
from statsmodels.formula.api import ols

model = ols('y ~ sat + weight + width', data=df)
result = model.fit()
print(result.summary())



#%%

#3. 팽창계수를 확인하기
from statsmodels.stats.outliers_influence import variance_inflation_factor

print(model.exog_names)#모델에서 분석한 독립변수들이 출력됩니다.

#설명: 위에 출력된 독립변수들 중 1(하나)번째 컬럼의 팽창계수 확인
print(variance_inflation_factor(model.exog, 1)) #1.15883687808578
print(variance_inflation_factor(model.exog, 2)) #4.8016794240392375
print(variance_inflation_factor(model.exog, 3)) #4.688660343641888


#%%
#4. 위의 팽창계수가 높은 두개의 독립변수를 각각의 모델로 만듭니다.

model1 = ols('y ~ sat + width', data = df)
print(model1.fit().summary())


model2 = ols('y ~ sat + weight', data = df)
print(model2.fit().summary())
