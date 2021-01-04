import  urllib.request #웹 url 을 파이썬이 인식할 수 있게 하는 패키지
from  bs4  import  BeautifulSoup #html 코드에서 원하는 지점을 빨리 찾을 수 있게 만든 모듈
from selenium import webdriver #손으로 클릭하는 것을 컴퓨터가 하게 시키는 모듈
from selenium.webdriver.common.keys import Keys #윗줄과 마찬가지로 셀레니움
import time #이미지 스크롤링할 때는 sleep 을 꼭 써 줘야함 (페이지에서 이미지가 로드되는데 시간이 걸림)

binary = '/Users/macbook/data/chromedriver/chromedriver' #크롬드라이버 위치 지정
browser = webdriver.Chrome(binary) #browser 객체 생성
browser.get("https://search.naver.com/search.naver?where=image&amp;sm=stb_nmr&amp;")
elem = browser.find_element_by_id("nx_query")
#find_elements_by_class_name("")

# 검색어 입력
elem.send_keys("아이언맨")
elem.submit()

# 반복할 횟수
for i in range(1,2):
    browser.find_element_by_xpath("//body").send_keys(Keys.END)
    time.sleep(10) #5초 잠자기

time.sleep(10)

html = browser.page_source #현 페이지의 html 코드 불러와서
soup = BeautifulSoup(html,"lxml") #BeautifulSoup을 이용할 수 있도록 파싱한다

#print(soup)
#print(len(soup))


def fetch_list_url():
    params = []
    imgList = soup.find_all("img", class_="_img") #img 태그의 클래스명 _img 로 접근
    for im in imgList:
        params.append(im["src"]) #src의 값을 가져와서 params에 append 시킴
    return params


def  fetch_detail_url():
    params = fetch_list_url()
    #print(params)
    a = 1
    for p in params:
        # 다운받을 폴더경로 입력
        urllib.request.urlretrieve(p, "/Users/macbook/data/naverimages/"+ str(a) + ".jpg" ) #이미지를 숫자를 누적시켜 경로설정해준 폴더로 저장

        a = a + 1

fetch_detail_url()

browser.quit()