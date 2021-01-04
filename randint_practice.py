#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 02:49:10 2021

@author: Sora
"""

import random
import numpy as np

n = 10
point = 1.96

def coin_avg_std(num):
    res = []
    for j in range(num):
        cnt = 0
        for i in range(n):
            cnt += (random.randint(0,1))
        res.append(cnt)    
    c_m = np.mean(res)
    c_s = np.std(res)
    return c_m, c_s

c_m, c_s = coin_avg_std(10000)

print("평균:", c_m)
print("표준편차:", c_s)

def coin_hypo(num):
    c_m, c_s = coin_avg_std(10000)
    if c_m-c_s*point<=num<=c_m+c_s*point:
        return f'동전을 {n}번 던졌을 때 뒷면이 나오는 횟수가 {num}번이 나올 확률은 신뢰구간 95%안에 있습니다.'
    else:
        return f'동전을 {n}번 던졌을 때 뒷면이 나오는 횟수가 {num}번이 나올 확률은 신뢰구간 95%안에 없습니다.'
    
for i in range(11):
    print(coin_hypo(i))
