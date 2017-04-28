from bs4 import BeautifulSoup
import requests , re , json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


price_list = []

def book_info(div):

 """given a BeautifulSoup <td> Tag representing a book,
 extract the book's details and return a dict"""
 author = div.find("div", "box_mid_billboard_pro").p.text
 title = div.find("h3").a.text
 price = div.find("span", "price").text
 price_list.append(int(price.replace(',', '')))

 return {
 "author" : author,
 "title" : title,
 "price" : price,
 "price_list":price_list
 }


from time import sleep
base_url = "http://www.eslite.com/newbook_list.aspx?cate=156&sub=159&page="
books = []
NUM_PAGES = 10 # at the time of writing, probably more by now
for page_num in range(1, NUM_PAGES + 1):
 url = base_url + str(page_num)
 soup = BeautifulSoup(requests.get(url).text, 'html5lib')
 for div in soup('div', 'box_mid_billboard'):
        books.append(book_info(div))



for b in books:
    print("title : " + b['title'])
    print("author : " + b['author'])
    print("price : "+b['price'])


price_count =[0,0,0,0,0,0,0,0,0]
price_level = ['100','200','300','400','500','600','700','800','900~']

for i in price_list:
   if i >900 :
      price_count[8] += 1
   elif i >800:
      price_count[7]+= 1
   elif i > 700:
      price_count[6] += 1
   elif i > 600:
      price_count[5] += 1
   elif i > 500:
      price_count[4] += 1
   elif i > 400:
      price_count[3] += 1
   elif i > 300:
      price_count[2] += 1
   elif i > 200:
      price_count[1] += 1
   else :
      price_count[0] += 1

x = np.array([0,1,2,3,4,5,6,7,8])
plt.xticks(x, price_level)
plt.plot(x,price_count,'go-')
plt.ylabel("The Number of Prices")
plt.xlabel("Book's Price Level")
plt.grid(True)
plt.fill_between(x,price_count,0,color='#c0f0c0')
plt.title("Book's Price Analysis")
plt.show()
