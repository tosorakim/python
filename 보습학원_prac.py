#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import numpy as np

# CSV 파일을 데이터프레임으로 변환
#dataset = pd.read_csv('D:\\data\\building.csv', encoding='cp949', index_col='Unnamed: 0')
#dataset = pd.read_csv('D:\\data\\building2.csv', encoding='cp949', header=None,names=['병원', '약국', '카페', '휴대폰매장', '일반음식점', '패밀리레스토랑', '당구장', '보습학원', '슈퍼마켓',
#       '은행', '편의점', '화장품'])

dataset = pd.read_csv('/Users/macbook/data/building3.csv', encoding='cp949', header=None,names=['h', 'y', 'c', 'hye', 'r', 'fr', 'd', 'n', 's',
       'e', 'p', 'hya'])
dataset = dataset.fillna(0)
print(dataset)

#%%

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
print(te_ary)
print(te.columns_)
#%%


df = pd.DataFrame(te_ary, columns=te.columns_) #위에서 나온걸 보기 좋게 데이터프레임으로 변경
print(df)

#%%

frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
print(frequent_itemsets)



from mlxtend.frequent_patterns import association_rules
rules1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3) 
print(rules1)

