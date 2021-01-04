import numpy as np
a = [ [1,2,3,0], [0,1,2,3], [3,0,1,2], [2,3,0,1] ]
a2 = np.array(a)
for i in range(0,4):
    if (a2[0:3, 0:3]):
        (a2[0:3, 1:4])
        (a2[1:4, 0:3])
        (a2[1:4, 1:4])
#%%


import numpy as np

a = [[1,2,3,0], [0,1,2,3], [3,0,1,2], [2,3,0,1]]
a2 = np.array(a)

for i, k in zip(range(0,2), range(3,5)):
    for j, h in zip(range(0,2), range(3,5)):
        print(a2[i:k, j:h])

 
#%%

n = [[2,3,1,4,7],[3,1,6,4,3],[2,1,5,3,1],[6,2,4,1,2],[7,3,1,2,3]]
n1 = np.array(n)
f = [[3,1,4,1],[2,3,3,4],[5,1,2,1],[6,1,3,4]]
f1 = np.array(f)

result = []
          
for i,j in zip(range(0,2),range(4,6)):
    for k,l in zip(range(0,2), range(4,6)):
        result.append(np.sum(n1[i:j,k:l]*f1))
       
result2 = np.array(result).reshape(2,2)
print(result2)