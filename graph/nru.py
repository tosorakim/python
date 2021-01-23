import mysql.connector
import  pandas as  pd
import matplotlib.pyplot as plt

config = { 
    "user": "root", 
    "password": "-", 
    "host": "10.211.55.7", #local 
    "database": "orcl", #Database name 
    "port": "3306" #port는 최초 설치 시 입력한 값(기본값은 3306) 
} 


conn = mysql.connector.connect(**config)

# db select, insert, update, delete 작업 객체
cursor = conn.cursor()

# 실행할 select 문 구성
sql = """
select channel, sum(nru)
from gagong
group by channel;

"""
           
# sql 을 실행해서 cursor (메모리) 에 담는다. 
cursor.execute(sql)

# cursor(메모리)에 담긴 결과 셋 얻어오기
resultList = cursor.fetchall()  # tuple 이 들어있는 list
df = pd.DataFrame(resultList)  # 판다스 데이터 프레임으로 변환 
df.columns = ['channel', 'nru']   #  컬럼명을 만들어 줍니다. 
print(df[['channel','nru']])


# 시각화 
plt.bar(df['channel'], df['nru'], color=['#A3E4D7','#48C9B0'])
plt.xlabel('Channel',fontsize=10)
plt.ylabel('nru',fontsize=10)
plt.xticks(fontsize=10, rotation=0)
plt.yticks(fontsize=10, rotation=0)
plt.show()