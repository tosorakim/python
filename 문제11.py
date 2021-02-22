import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas  as pd


# 1. 데이터 준비
#col_names = ['sepal-length', 'sepal-width','petal-length', 'petal-width','Class']

# csv 파일에서 DataFrame을 생성
#dataset = pd.read_csv('d:\\data\\iris2.csv', encoding='UTF-8', header=None, names=col_names)
#dataset = pd.read_csv("/Users/macbook/data/iris2.csv" , encoding='UTF-8', header=None, names=col_names)
df = pd.read_csv("/Users/macbook/data/wine.csv") #유방암데이터
print(df)

# DataFrame 확인
print(df.shape) #[178 rows x 14 columns]

print(df.info()) # 0   Type             178 non-null    object 
#%%
print(df.describe())
print(df)
#%%
# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:, 1:].to_numpy() 
y = df['Type'].to_numpy()   #암진단
print(X)

print(df.shape)  # (178, 14)
print(len(X))  # 178
print(len(y))  # 178

#데이터 정규화 - 전처리 과정
from sklearn import preprocessing 

#X=preprocessing.StandardScaler().fit(X).transform(X) 
X=preprocessing.MinMaxScaler().fit(X).transform(X)  #MIN/MAX (0~1사이 데이터로 바꿔버리)

from sklearn.model_selection import train_test_split 
                                                                
                     
# 훈련 데이터 : 테스트 데이터 - 9:1
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state = 10) #고정값 10

print(X_train.shape) #(160, 13)
print(y_train.shape) #(160,)


# 스케일링(z-score 표준화 수행 결과 확인)
for col in range(4):
    print(f'평균 = {X_train[:, col].mean()}, 표준편차= {X_train[:, col].std()}')
    
for col in range(4):
    print(f'평균 = {X_test[:, col].mean()}, 표준편차= {X_test[:, col].std()}')



# 학습/예측(Training/Pradiction)
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB

# k-NN 분류기를 생성
classifier = KNeighborsClassifier(n_neighbors=6)

# knn 분류기를 생성
#classifier = GaussianNB() # 

# 분류기 학습
classifier.fit(X_train, y_train)

# 예측
y_pred= classifier.predict(X_test)
print(y_pred)

# 작은 이원교차표
from sklearn.metrics import confusion_matrix
conf_matrix= confusion_matrix(y_test, y_pred)
print(conf_matrix)    

# 정밀도 , 재현율, f1 score 확인 
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

# 정확도 확인하는 코드 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score( y_test, y_pred)

# 정확도 확인하는 코드 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score( y_test, y_pred)
print(accuracy)  # 정확도: 0.9444444444444444
#%%

import numpy as np
    
# 6. 모델 개선 - laplace 값을 변화시킬 때, 에러가 줄어드는 지
errors = []
for i in  np.arange(0.0, 1.0, 0.001):
    model = GaussianNB( var_smoothing= i)
    model.fit(X_train, y_train)
    pred_i = model.predict(X_test)
    errors.append(np.mean(pred_i != y_test))
print(errors)

# 여기서 에러가 가장 적은 것을 선택

import matplotlib.pyplot as plt

plt.plot( np.arange(0.0, 1.0, 0.001), errors, marker='o')
plt.title('Mean error with laplace-Value')
plt.xlabel('laplace')
plt.ylabel('mean error')
plt.show()