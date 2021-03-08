#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:08:17 2021

@author: Sora
"""
#가중치를 생성하는 함수를 만든다
def init_network():
    network = {} #비어있는 딕셔너리 생성
    network['W1']=np.array([ [1,3,5], [2,4,6] ])
    network['W2']=np.array([ [3,4], [5,6], [7,8] ])
    network['W3']=np.array([ [4,5], [6,7] ])
    return network

#신경망 함수들(시그모이드, 소프트맥스)
def sigmoid(x):
    return 1/(1+np.exp(-x))
 
def softmax(a):
    C=np.max(a)
    minus=a-C
    exp_a=np.exp(minus)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
