a = int(input('숫자를 입력하세요 ~ '))
b = int(input('숫자를 입력하세요 ~ '))
for i in range(1,b+1):
    for j in range(1,a+1):
        if j == 2:
            break
        print('★' * a)
#%%
a = int(input('숫자를 입력하세요 ~ '))
b = int(input('숫자를 입력하세요 ~ '))
x=0
while x <= b:
    print('★'*a)
    x = x+1

#%%

arr = [1,7,11,12,14,23,33,47,51,64,67,77,139,672,871]
print(arr)

target = int(input('검색할 숫자를 입력하세요 : '))
start = 0
end = len(arr) - 1

cnt = 0 
while start <= end:
    cnt += 1
    mid = (start + end) // 2 #중간점 찾고,
    #찾은 경우 인덱스 중간점 반환
    if arr[mid] == target:
        print("%d은 이진탐색 %d번 만에 검색되었습니다." %(target, cnt))
        break
    elif arr[mid] > target:
        end = mid - 1
    else: 
        start = mid + 1


#%%
a = int(input('숫자를 입력하세요 ~ '))
b = int(input('숫자를 입력하세요 ~ '))
for i in range(1, b+1):
    print('★' * a)

#%%
num = int(input('숫자를 입력하세요 67')) #67

 

import numpy as np

a= [1,7,11,12,14,23,33,47,51,64,67,77,139,672,871]

 

for i in range(len(a) ) : 

    an = np.array(a) # a 리스트를 numpy 모듈을 사용하여 array 형태로 변환 ※

    am = np.median(an) # array 형태로 변환한 a 리스트에서 중앙값 찾기 # 47  ※

    if num in a : # 찾는 숫자 num이 a 리스트에 있다면

        if num == am : # 만약 찾는 숫자num이 a 중앙값(am) 과 같다면

            print(num, '은', i, '번에 나왔습니다')

            break    # 찾았으니 종료

        elif num > am : # 찾고자 하는 숫자가 a 중앙값보다 크다면,  67>47

            del(a[ : a.index(am)+1 ] ) # 1부터 본인포함 47까지 삭제

            print(a) # 삭제 후 남은 리스트 출력

        else: 

            del(a[a.index(am) : ] ) 

            print(a)

    

    else: # 찾는 숫자 num이 a 리스트에 없다면 

        print(num, '은 리스트에 없습니다')

        break