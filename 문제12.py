import pandas as pd #데이터 전처리를 위해서
import seaborn as sns #시각화를 위해서

#df = pd.read_csv("d:\\data\\wisc_bc_data.csv") 
#df = open("/Users/macbook/Documents/itwill/python/wisc_bc_data.csv")
#emp = pd.read_csv("/Users/macbook/data/emp3.csv")
df = pd.read_csv("/Users/macbook/data/mushrooms.csv")
#print(df)
df = pd.get_dummies(df)
print(df)

# DataFrame 확인
#print(df.shape) #(8124, 23) ------> (8124, 119)

#print(df.info()) #전부 object(문자형)으로 데이터 구성되어있

#print(df.describe())

#print(df.iloc[0:5, ]) 

#print(df.iloc[-5: ,])

#print(df.iloc[ :, [0,1] ]) 
#print(df.iloc[ :, : ]) 


# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:, 2:].to_numpy() 
y = df.iloc[:, 1].to_numpy()   

print(y)
#%%
#print(df.shape) #(569, 32)
#print(len(X)) #569
#print(len(y)) #569

#데이터 정규화를 수행한다.    (MIN/MAX 적용)                   
#X=preprocessing.StandardScaler().fit(X).transform(X) #scale 함수 적용

#X=preprocessing.MinMaxScaler().fit(X).transform(X) #min/max 함수 적용
#print(X)

# ■ 훈련 데이터와 테스트 데이터를 분리하는 작업
from sklearn.model_selection import train_test_split 
                                                                
                     
# 훈련 데이터 : 테스트 데이터 ---- 75:25
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.25, random_state = 10) #데이터를 7:3으로 나눴습니다(0.3)

print(X_train.shape) #훈련데이터 (6093, 22)
print(y_train.shape) #라벨 (6093,)

# 스케일링(z-score 표준화 수행 결과 확인)
#for col in range(4):
#    print(f'평균 = {X_train[:, col].mean()}, 표준편차= {X_train[:, col].std()}')
    
#for col in range(4):
#    print(f'평균 = {X_test[:, col].mean()}, 표준편차= {X_test[:, col].std()}')

# 학습/예측(Training/Pradiction)
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

# k-NN 분류기를 생성
#classifier = KNeighborsClassifier(n_neighbors=12)

# 나이브베이즈 분류기를 생성
classifier = BernoulliNB() # 

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
print(accuracy)  #  0.9497   MultinomialNB
                 #  0.9615   GaussianNB
                 #  0.9350   BernoulliNB