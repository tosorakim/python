#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:59:13 2021

@author: Sora
"""

import numpy as np
a = np.array([[1,2],[3,4]])
print(a+5)

#%%
import numpy as np
x=np.array([1,2])
w1=np.array([[1,3,5],[2,4,6]])#가중치 행렬1
w2=np.array([[3,4],[5,6],[7,8]])#가중치 행렬2
k=np.dot(x,w1)#1 floor
m=np.dot(k,w2)#2 floor
print(m) #[189 222]

#%%
import numpy as np
x=np.array([[1,2],[3,4]])
print(x)
#%%
x=np.array([ [3,4,2],[4,1,3] ])
w=np.array([ [1,5],[2,3],[4,1] ])
k=np.dot(x,w)
print(k)

#%%
a=[51, 55, 14,19, 0, 4]
b=[]
for i in a:
    if i >= 15:
        b.append(i)
print(b)
#%%
a=[ [51,55], [14,19], [0,4] ] #리스트안에 리스트가 있는 형태
b=np.array(a)
print(b[b>=15]) #[51 55 19]

#%%
import numpy as np
import matplotlib.pyplot as plt

x=np.array([ 0,1,2,3,4,5,6,7,8,9 ])
y=np.array([ 9,8,7,9,8,3,2,4,3,4 ])

plt.plot(x,y,color='red') #산포드일때는 scatter, 라인일때는 plot
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlibimport font_manager,rc

font_path = "/Users/macbook/data/malgun.ttf"   #폰트파일의 위치 실습파일안에있음
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

x=np.array([ 0,1,2,3,4,5,6,7,8,9 ])
y=np.array([ 9,8,7,9,8,3,2,4,3,4 ])

plt.plot(x,y,color='red') #산포드일때는 scatter, 라인일때는 plot
plt.title("신경망 오차 그래프")
plt.show()
#%%
X=np.array([ [0,0], [0,1], [1,0], [1,1] ]) #입력 데이터
y=np.array([ [0], [0], [0], [1] ]) #정답
print(X.shape)
print(y.shape)
#%%

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1+x2*w2 #입력값과 가중치의 곱의 총합
    if tmp <-theta:
        return 0
    elif tmp>theta:
        return 1

print(AND(0,0))
print(AND(1,0))
print(AND(0,1))
print(AND(1,1))



#%%
def NAND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1+x2*w2 #입력값과 가중치의 곱의 총합
    if tmp <=theta: #입력값과 가중치의 총합이 임계치를 넘지 않는다면,
        return 1
    elif tmp>theta:
        return 0

def XOR(x1, x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y

print(XOR(0,0)) #0
print(XOR(0,0)) #1
print(XOR(0,0)) #1
print(XOR(0,0)) #0
#%%
def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    else:
        return 1
    
def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y

print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))
#%%

def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5])  
    b = -0.7
    tmp = np.sum(x*w)+b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5])  
    b = -0.4
    tmp = np.sum(x*w)+b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5])  
    b = -0.7
    tmp = np.sum(x*w)+b
    if tmp <= 0:
        return 1
    else:
        return 0

def XOR(x1, x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y


## X행렬(x2)과 y행렬(label=y2) 생성
x = np.array( [0,0,0,1,1,0,1,1] ) 
x2 = x.reshape(4, -1)
y = np.array( [0,1,1,0] ) 
y2 = y.reshape(4,-1)

# loop 를 사용하여 X행렬의 인자값들을 저장
bin = []
for i in range(x2.shape[0]):
    for j in range(x2.shape[1]):
        bin.append(x2[i][j])
k1 = bin[::2]  #홀수번째 인자
k2 = bin[1::2]  #짝수번째 인자

# XOR 게이트에 각 인자를 대입하여 결과값 출력
for i, j in zip(k1,k2):
    print(XOR(i,j) )

# label column과 비교
print('\n', y)

#%%
x=np.array([ [-1,0,0], [-1,1,0], [-1,0,1], [-1,1,1] ])
w=np.array([ [0.3], [0.4], [0.1] ])

print(x.shape) #(4, 3)
print(w.shape) #(3, 1)


#%%
x=np.array([ [-1,0,0], [-1,1,0], [-1,0,1], [-1,1,1] ]) #4*3
w=np.array([ [0.3], [0.4], [0.1] ]) #3*1
target = np.array([ [0], [0], [0], [1] ])

print(w.T.shape)
print(w.T.shape)

print(x*w.T)
print(np.sum(x*w.T)) #-0.19999999999999993 


print(np.sum(x[0]*w.T))
#%%
for i in range(len(x)):
    print(np.sum(x[i]*w.T))

#%%
k=x*w.T
print(k.sum(axis=1))
#%%
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

print(step_function(0.3)) #1
print(step_function(-2)) #0

#%%
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def predict(x,w):
    a=np.sum(x*w.T)
    return step_function(a) #return할때 step_function값이 바로 리턴되도록 처리한다

target=np.array([ [0], [0], [0], [1] ])

for i in range(len(x)):
    cost = target[i] - predict(x[i],w)
    print(cost)


#%%
def step_func(x):
    if x>0:
        return 1
    else:
        return 0
    
x_data = np.array([-1,0,1]) #신경망의 데이터인 numpy array 형태로 구성
#print(step_func(x_data)) #에러남 ---> 즉 step 함수를 그렇게 짤 수 없다는 것을 의미
print(step_func(3.0)) #1

#설명: 위에서 만든 step_func 은 넘파이 배열을 넣을 수 없습니다.


#%%
def step_func(x):
    y=x>0
    print(y)
step_func(3.0) #True
#%%
def step_func(x):
    y=x>0 #True 또는 False가 출력해라
    return y.astype(np.int) #bool type을 숫자형으로 변환해라(True:1, False:2)

print(step_function(np.array(3.0))) #1
print(step_function(np.array(-3.0))) #0


x_data=np.array([-1,0,1])
print(step_func(x_data)) #[0 0 1]


#%%
import matplotlib.pylab as plt
import numpy as np

def step_func(x):
    y=x>0 #True 또는 False가 출력해라
    return y.astype(np.int) #bool type을 숫자형으로 변환해라(True:1, False:2)

x=np.arange(-5, 5, 0.1)
y=step_func(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()


#%%
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
x_data=np.array([-0.5,1.0,2.0])
print(sigmoid(x_data)) #[0.37754067 0.73105858 0.88079708]

x=np.arange(-5, 5, 0.1)
y=sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

#%%



import matplotlib.pylab as plt
import numpy as np

def odds_ration(x):
    return x/(1-x)

x=np.arange(0,1,0.01)
y=odds_ration(x)
plt.plot(x,y)
plt.ylim(0,100)
plt.show()

#%%

import matplotlib.pylab as plt
import numpy as np

def logit(x):
    return np.log(x/(1-x))

x=np.arange(0.01, 1, 0.01)
y=logit(x)
plt.plot(x,y)
plt.show()


#%%
def AND(x1, x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5]) #실제로는 w, b를 기계(컴퓨터)가 학습해서 알아내야합니다.
    b=-0.7 #우리는 하드코딩해서 값을 집어넣었습니다.
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    else:
        return 1

print(AND(0,0))
print(AND(1,0))
print(AND(0,1))
print(AND(1,1))


#%%
def  AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])   #  실제로는 이 w 와 b 를 컴퓨터(기계)가 학습해서 알아내야합니다.
    b = -0.7
    tmp = np.sum(x*w) + b  
    if  tmp <= 0:
        return  0 
    else:
        return  1 

print (AND(0,0) )
print (AND(1,0) )
print (AND(0,1) )
print (AND(1,1) )
#%%
def  OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])   #  실제로는 이 w 와 b 를 컴퓨터(기계)가 학습해서 알아내야합니다.
    b = -0.4
    tmp = np.sum(x*w) + b  
    if  tmp <= 0:
        return  0 
    else:
        return  1 

print (OR(0,0) )
print (OR(1,0) )
print (OR(0,1) )
print (OR(1,1) )


#%%
def OR(x1, x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5]) #실제로는 w, b를 기계(컴퓨터)가 학습해서 알아내야합니다.
    b=-0.4 #우리는 하드코딩해서 값을 집어넣었습니다.
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    else:
        return 1

print(AND(0,0)) #0
print(AND(1,0)) #0
print(AND(0,1)) #0
print(AND(1,1)) #1








