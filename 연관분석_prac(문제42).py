#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:07:13 2021

@author: Sora
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


dataset=[['사과','치즈','생수'],
['생수','호두','치즈','고등어'],
['수박','사과','생수'],
['생수','호두','치즈','옥수수']]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
print(te.columns_)
print(te_ary)
#%%
df = pd.DataFrame(te_ary, columns=te.columns_) #위에서 나온걸 보기 좋게 데이터프레임으로 변경
print(df)
#%%

frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print(frequent_itemsets )

#%%

from mlxtend.frequent_patterns import association_rules
print( association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3) ) 