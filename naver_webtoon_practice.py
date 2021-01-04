#구글 플레이 웹스크래핑
from selenium import webdriver
driver = webdriver.Chrome('/Users/macbook/data/chromedriver/chromedriver')
from bs4 import BeautifulSoup

keyword = '코끼리'
url = 'https://www.google.com/search?q='+keyword+'&hl=ko&sxsrf=ACYBGNSTW5YFeVU0I4abA6H_bXsmwJ-gag:1582014089814&source=lnms&tbm=isch&sa=X&ved=2ahUKEwj7kune1drnAhXaAYgKHQY3CwkQ_AUoAXoECBUQAw&biw=1440&bih=712'

driver.get(url)

req = driver.page_source

soup = BeautifulSoup(req, 'html.parser')

images = soup.select('#islrg > div.islrc > div')

for count, image in enumerate(images):
    img = image.select_one('img')
    print(img['data-iurl'])
    if count == 5:
        break

driver.close()
#%%
driver = webdriver.Chrome('/Users/macbook/data/chromedriver/chromedriver')
from bs4 import BeautifulSoup

appkey = 'com.nhn.android.webtoon'
url = 'https://play.google.com/store/apps/details?id='+appkey+'&showAllReviews=true'

driver.get(url)

req = driver.page_source

soup = BeautifulSoup(req, 'html.parser')

data = soup.select('#fcxH9b > div.WpDbMd > c-wiz')

for count, div in enumerate(data):
    div = div.select_one('div')
    print(div['data-iurl'])
    if count == 5:
        break

driver.close()


#%%
#네이버 영화 웹스크래핑 - 인턴   
from bs4 import BeautifulSoup
import urllib.request as req

f = open("Documents/itwill/python/sample.txt", "w", encoding = "UTF8")

for i in range(1, 5): #숫자주의
    list_url = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=118917&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page=' + str(i)
    url = req.Request(list_url)
    result = req.urlopen(url).read().decode("UTF-8")   
    soup = BeautifulSoup( result, "html.parser" )
    for i in range(0, 10):
        movie = soup.find_all('span', id = '_filtered_ment_' + str(i))
        for intern in movie:
            f.write(intern.get_text(" ", strip = True) + '\n')

f.close()

#%%
#구글 플레이 웹스크래핑
from selenium import webdriver
driver = webdriver.Chrome('/Users/macbook/data/chromedriver/chromedriver')
from bs4 import BeautifulSoup
import urllib.request as req

f = open("Documents/itwill/python/sample.txt", "w", encoding = "UTF8")

appkey = 'com.nhn.android.webtoon'
url = 'https://play.google.com/store/apps/details?id='+appkey+'&showAllReviews=true'

driver.get(url)
req = driver.page_source
soup = BeautifulSoup(req, 'html.parser')
data = soup.select('#fcxH9b > div.WpDbMd > c-wiz')

for count, image in enumerate(data):
    img = image.select_one('img')
    print(img['data-iurl'])
    if count == 5:
        break

driver.close()