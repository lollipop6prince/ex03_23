# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests , re , json
from collections import Counter

def book_info(div):
 """given a BeautifulSoup <td> Tag representing a book,
 extract the book's details and return a dict"""
 author = div.find("div", "box_mid_billboard_pro").p.text
 title = div.find("h3").a.text
 print(div.find('span','price_sale'))
 #price = div.find("div", "box_mid_billboard").p.text
 # isbn_link = td.find("div", "thumbheader").a.get("href")
 # isbn = re.match("/product/(.*)\.do", isbn_link).groups()[0]
 # date = td.find("span", "directorydate").text.strip()
 return {
 "author" : author,
 "title" : title
 # "isbn" : isbn,
 # "date" : date
 }

url = "http://www.eslite.com/newbook_list.aspx?cate=156&sub=159&page=1"
soup = BeautifulSoup(requests.get(url).text, 'html5lib')


tds = soup('td', 'thumbtext')
print len(tds)
# 30

def is_video(td):
 """it's a video if it has exactly one pricelabel, and if
 the stripped text inside that pricelabel starts with 'Video'"""
 pricelabels = td('span', 'pricelabel')
 return (len(pricelabels) == 1 and
 pricelabels[0].text.strip().startswith("Video"))
#print len([td for td in tds if not is_video(td)])
# 21 for me, might be different for you

from time import sleep
base_url = "http://www.eslite.com/newbook_list.aspx?cate=156&sub=159&page="
books = []
NUM_PAGES = 1 # at the time of writing, probably more by now
for page_num in range(1, NUM_PAGES + 1):
 print "souping page", page_num, ",", len(books), " found so far"
 url = base_url + str(page_num)
 soup = BeautifulSoup(requests.get(url).text, 'html5lib')
 i = 0
 for div in soup('div', 'box_mid_billboard'):
     date = div.find("span" , id = "ctl00_ContentPlaceHolder1_newbook_bookList_ctl0"+str(i)+"_retailPrice").text.strip()
     i += 1
     if i > 10:
         break
     print date
 # now be a good citizen and respect the robots.txt!
# sleep(2)

print(books)

for b in books:
    print("title : " + b['title'])
    print("author : " + b['author'])