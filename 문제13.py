#NB - Mushrooms.csv
import pandas as pd
import seaborn as sns

df = pd.read_csv("/Users/macbook/data/mushrooms.csv")
df = pd.get_dummies(df)
print(df)

# DataFrame 확인
print(df.shape)  # (8124, 119)
print(df.info())
print(df.describe()) #[8 rows x 119 columns]

# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:, 2:].to_numpy() 
y = df.iloc[:,1].to_numpy()   
print(y)

# 훈련 데이터 : 테스트 데이터 (75:25)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state = 10)
print(X_train.shape) #훈련데이터 (6093, 117)
print(y_train.shape) #(6093,)

from sklearn.naive_bayes import GaussianNB #가우시안
classifier = GaussianNB(var_smoothing=0.004) 
classifier.fit(X_train, y_train)
y_pred= classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
conf_matrix= confusion_matrix(y_test, y_pred)
print(conf_matrix) # 작은 이원교차표   
#[[1055   17]
# [   0  959]]

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report) # 정밀도 , 재현율, F1 Score 확인 

from sklearn.metrics import accuracy_score
accuracy = accuracy_score( y_test, y_pred)
print(accuracy)  #0.9916297390448056

import  numpy  as np

errors = []

for i in np.arange(0.001,0.01,0.001):   # 0.001, 0.002 , ... , 0.010
    nb = GaussianNB(var_smoothing=i)
    nb.fit(X_train, y_train)
    pred_i = nb.predict(X_test) #예측값
    errors.append(np.mean(pred_i != y_test)) #예측값과 실제값의 평균을 errors 라는 빈 리스트에 append 
print(errors)

for k, i  in  zip(np.arange(0.001,0.01,0.001), errors):
    print (round(k, 3), '--->', i)  # 0.004 ---> 0.008370260955194485

import matplotlib.pyplot as plt

plt.plot(np.arange(0.001,0.01,0.001), errors, marker='o')
plt.title('Mean error with K-Value')
plt.xlabel('laplace value')
plt.ylabel('mean error')
plt.show()
