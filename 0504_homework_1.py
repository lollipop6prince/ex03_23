# encoding: utf-8

import urllib
url = "http://www.thsrc.com.tw/tw/TimeTable/SearchResult"
request = urllib.Request(url)
request.add_header("User-Agent","Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36")

form_data = {
    "StartStation": "977abb69-413a-4ccf-a109-0272c24fd490",
    "EndStation": "f2519629-5973-4d08-913b-479cce78a356",
    "SearchDate": "2016/01/10",
    "SearchTime": "17:00",
    "SearchWay":"DepartureInMandarin",
    "RestTime":"",
    "EarlyOrLater":""
}
form_data = urllib.urlencode(form_data)
response = urllib.urlopen(request,data=form_data)
html = response.read()

file_out = file("0504_homework_1.html",'w')
file_out.write(html)
file_out.close()