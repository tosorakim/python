#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:34:02 2021

@author: Sora
"""
import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0,x) #0과 x값중에 큰 값을 출력하시오!

print(relu(-2))
print(relu(0.3))
#%%


x=np.arange(-5,5,0.1)
y=relu(x)

plt.plot(x,y,color='red')
plt.show()

#%%
import numpy as np
b=np.array([    [[1,2],[3,4]],    [[5,6],[7,8]]    ])
print(b[1][0][0])
print(np.ndim(b)) #3차원
#%%
import numpy as np
#방법1
x=np.array([ [1,2], [3,4] ])
y=np.array([ [5,6], [7,8] ])
print(np.dot(x,y))

#방법2
x=np.array([ [1,2], [3,4] ])
y=np.array([ [1,2], [3,4] ])
print(x*y)

#방법3
x=np.array([ [1,2], [3,4] ])
y=np.array([ [1,2], [3,4] ])
print(x@y)
#%%

x=np.array([ [4,5,7,2] ])
y=np.array([ [8,21,1], [4,5,9], [6,34,4], [12,2,5] ])
print(np.dot(x,y))
#%%
import numpy as np
x=np.array([1,2])
w1=np.array([ [1,3,5], [2,4,6] ])
y=np.dot(x,w1)
print(y)
#%%
#시그모이드 통과하려면,
#(0층에서 은닉1층으로 가는것)
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.array([1,2])
w1=np.array([ [1,3,5], [2,4,6] ])
y=np.dot(x,w1)
y_hat=sigmoid(y)
print(y_hat) #[0.99330715 0.9999833  0.99999996]

#%%
#출력층(3층)까지 가기
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

#0층
x=np.array([1,2])

#1층
w1=np.array([ [1,3,5], [2,4,6] ])
y=np.dot(x,w1)
y_hat=sigmoid(y)

#2층
w2=np.array([ [3,4],[5,6],[7,8] ])
z=np.dot(y_hat,w2)
z_hat=sigmoid(z)

#3층
w3=np.array([ [4,5], [6,7] ])
k=np.dot(z_hat,w3)
k_hat=softmax(k)
print(k) #[ 9.99999866 11.99999833]
print(k_hat) #[0.11920296 0.88079704]


#%%
import numpy as np
print(np.exp(10)) #22026.465794806718
print(np.exp(100)) #2.6881171418161356e+43
print(np.exp(1000)) #inf


#%%
import numpy as np
a=np.array([ 1010,1000,990])

def softmax(a):
    C=np.max(a)
    minus=a-C
    np_exp=np.exp(minus)
    return np_exp #분자까지 구현함

print(softmax(a))
#%%
import numpy as np
a=np.array([ 1010,1000,990])

def softmax(a):
    C=np.max(a)
    minus=a-C
    exp_a=np.exp(minus)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

print(softmax(a)) #[9.99954600e-01 4.53978686e-05 2.06106005e-09]
print(np.sum(softmax(a))) #1.0
print(np.argmax(softmax(a))) #0
#%%
import numpy as np
network = {} #비어있는 딕셔너리 생성
network['W1']=np.array([ [1,3,5], [2,4,6] ])
network['W2']=np.array([ [3,4], [5,6], [7,8] ])
network['W3']=np.array([ [4,5], [6,7] ])

print(network['W1'])
print(network['W2'])
print(network['W3'])

#%%
import numpy as np
import common #from my working directory

#가중치 값을 불러온다
network=init_network()
w1,w2,w3=network['W1'],network['W2'],network['W3'] #network가중치를 각 w1,w2,w3에 넣어라

#0층
x=np.array([1,2])

#1층
y=np.dot(x,w1)
y_hat=sigmoid(y)

#2층
z=np.dot(y_hat,w2)
z_hat=sigmoid(z)

#3층
k=np.dot(z_hat,w3)
k_hat=softmax(k)
print(k_hat) #[0.11920296 0.88079704]












