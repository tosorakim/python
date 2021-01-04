b = int(input('검색할 숫자를 입력하세요 ~ '))

a = [15, 11, 1, 3, 8]

for i in a:
    if i == b:
        print('숫자', b, '가 있습니다.')
        break
else:
    print('None')
    #%%
b = int(input('검색할 숫자를 입력하세요 ~ '))
a = [1,7,11,12,14,23,33,47,51,64,67,77,139,672,871]

import numpy as np
a_n = np.array(a)
median = np.median(a_n)
del(a[  : a.index(median) + 1])
median2 = np.median(a)
if median2 > 67:
    del (a[a.index(median2):  ])
else:
    del(a[  :a.index(median2)])

cnt = 1
for i in b:
    a == 67
    if not a == None:

        cnt += 1
        
        
#%%
a=[1,7,11,12,14,23,33,47,51,64,67,77,139,672,871]

import numpy as np
a_n = np.array(a)
median = np.median(a_n)
del(a[  : a.index(median) + 1])
median2 = np.median(a)
if median2 > 67:
    del (a[a.index(median2):  ])
else:
    del(a[  :a.index(median2)])
print(a) #[51, 64, 67]

#%%
import bisect
mylist = [1, 2, 3, 7, 9, 11, 33]
print(bisect.bisect(mylist, 3))

#%%

# movies (tr들) 의 반복문을 돌리기
rank = 1
for movie in movies:
    # movie 안에 a 가 있으면,
    a_tag = movie.select_one('td.title > div > a')
    if not a_tag == None:
        title = a_tag.text
        star = movie.select_one('td.point').text
        
        doc = {
            'rank' : rank,
            'title' : title,
            'star' : star
        }
        db.movies.insert_one(doc)
        rank += 1

#%%
people = [{'name':'bob','age':20},{'name':'carry','age':38},{'name':'john','age':7}]

# 모든 사람의 이름과 나이를 출력해봅시다.
for person in people:
	print (person['name'] + ' / ' + person['age'])

# 이번엔, 반복문과 조건문을 응용한 함수를 만들어봅시다.
# 이름을 받으면, age를 리턴해주는 함수
def get_age(myname):
	for person in people:
		if person['name'] == myname:
			return person['age']
	return '해당하는 이름이 없습니다'

print (get_age('bob'))
print (get_age('kay'))

#%%
import numpy as np

arr = [1,7,11,12,14,23,33,47,51,64,67,77,139,672,871]
print(arr)

target = int(input('검색할 숫자를 입력하세요 : '))
arr.sort()
a_n = np.array(arr)
start = 0
end = len(arr) - 1

cnt = 0
while  start <= end:
    cnt += 1  

    if len(arr)%2 == 1:
        mid = int(np.median(a_n))
    else:
        mid = arr[len(arr) // 2]

    if target == mid:
        print("%d은 이진탐색 %d번 만에 검색되었습니다." %(mid, cnt))
        break
    elif target > mid:
        del arr[:arr.index(mid)]
    else:
        del arr[arr.index(mid):]
#%%
import numpy as np

arr = [1,7,11,12,14,23,33,47,51,64,67,77,139,672,871]
print(arr)

target = int(input('검색할 숫자를 입력하세요 : '))
arr.sort()
start = 0 
end = len(arr) - 1

cnt = 0
while start > end:
    cnt += 1
    a_n = np.array(arr)

    if len(arr)%2 == 1:
        mid = int(np.median(a_n))
    else:
        mid = (start + end) // 2

    if arr[mid] == target:
        print("%d은 이진탐색 %d번 만에 검색되었습니다." %(mid, cnt))
        break
    elif arr[mid] > target:
        end = mid - 1
        print("%d차 시도 :" %cnt, start, end)
    else: 
        start = mid +1
        print("%d차 시도 :" %cnt, start, end)


#%%
import numpy as np

arr = [1,7,11,12,14,23,33,47,51,64,67,77,139,672,871]
print(arr)

target = int(input('검색할 숫자를 입력하세요 : '))
arr.sort()
start = 0 
end = len(arr) - 1

cnt = 0
while True: #무한루프 
    cnt += 1
    a_n = np.array(arr)
    if len(arr)%2 == 1:
        mid = int(np.median(a_n))
    else:
        mid = arr[(start + end) // 2]

    if arr[mid] == target:
        print("%d은 이진탐색 %d번 만에 검색되었습니다." %(mid, cnt))
        break
    elif target < arr[mid]:
        end = mid - 1
        del arr[:arr.index(mid)]
    else:
        del arr[arr.index(mid):]

        #%%
a=[1,7,11,12,14,23,33,47,51,64,67,77,139,672,871]

import numpy as np
a_n = np.array(a)

median = np.median(a_n)
del(a[  : a.index(median) + 1])
median2 = np.median(a)
if median2 > 67:
    del (a[a.index(median2):  ])
else:
    del(a[  :a.index(median2)])
print(a) #[51, 64, 67]


#%%
arr = [1,7,11,12,14,23,33,47,51,64,67,77,139,672,871]
print(arr)

target = int(input('검색할 숫자를 입력하세요 : '))
arr.sort()
start = 0
end = len(arr) - 1

cnt = 0 
while start <= end:
    cnt += 1
    mid = (start + end) // 2 #중간점 찾고,
    #찾은 경우 인덱스 중간점 반환
    if arr[mid] == target:
        print("%d차 시도, 탐색 완료 !! 인덱스 번호 : %d" %(cnt, mid))
        break
    elif arr[mid] > target:
        end = mid - 1
        print("%d차 시도 :" %cnt, start, end)
    else: 
        start = mid +1
        print("%d차 시도 :" %cnt, start, end)
#%%
import numpy as np

arr = [1,7,11,12,14,23,33,47,51,64,67,77,139,672,871]
print(arr)
target = int(input('검색할 숫자를 입력하세요 : '))

start = 0 
end = len(arr) - 1
a_n = np.array(a)

cnt = 0
while start <= end:
    cnt += 1
    
    mid = (start + end) // 2
    if arr[mid] == target:
        print("%d차 시도")
        break
    else:
        start = mid +1
        print("none")

#%%

import numpy as np

a = [1, 7, 11, 12, 14, 23, 33, 47, 51, 64, 67, 77, 139, 672, 871, 990]
print(a)

num = int(input("검색할 숫자를 입력하세요 ~ "))
cnt = 0
for i in range(len(a)):
    if len(a) % 2 == 0:
        medA = a[int(len(a)/2)]
    else:
        medA = np.median(a)
    if num < medA:
        cnt += 1
        del a[a.index(medA):]
    
    elif num > medA:
        cnt += 1
        del a[:a.index(medA) + 1]
    
    elif num == medA:
        cnt += 1
        if cnt == 1:
            print("{0}은 이진탐색 {1}번만에 검색되었습니다.".format(num, cnt))
        elif cnt != 1:
            print("{0}은 이진탐색 {1}번만에 검색되었습니다.".format(num, cnt - 1))
        break
    else:
        print("{}은 리스트 a에 없습니다.".format(num))
        break
#%%
arr = [1,7,11,12,14,23,33,47,51,64,67,77,139,672,871]
print(arr)

target = int(input('검색할 숫자를 입력하세요 : '))
arr.sort()
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
        start = mid +1


