from instagram.client import InstagramAPI
access_token = "1680164035.df2dac3.1fe1ae8518f64efab07916b209f8219b"
client_secret = "f0d0b886009945908d70e16d74acd799"

api = InstagramAPI(access_token=access_token, client_secret=client_secret)
user_id = api.user_search('naviyang_0909')[0].id
for i,v in enumerate(api.user_search('naviyang_0909')):
   print(v.id)
recent_media, next_ = api.user_recent_media(user_id=user_id, count=5)

for media in recent_media:
   print (media.caption.text)
   print ('<img src="%s"/>' % media.images['thumbnail'].url)



#if "data" in entry["comments"]:
#	for comment in entry['comments']['data']:
#		new_media.comments.append(Comment.object_from_dictionary(comment))