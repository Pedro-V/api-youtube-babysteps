#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Pedro-V/api-youtube-babysteps/blob/main/YoutubeAPI.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[2]:


get_ipython().system('pip install --upgrade google-api-python-client;')


# In[3]:


from googleapiclient.discovery import build


# In[8]:


#Os passos para obter uma key para a API podem ser vistos nesse vídeo do canal Programação Dinâmica: https://www.youtube.com/watch?v=olDCJ1w3FLM
youtubeApiKey = "AIzaSyCGuWtlgc2rCsQUZWQyJ5W-sVPLvE3O77w"

youtube = build('youtube','v3', developerKey=youtubeApiKey)


# In[9]:


#extraindo dados de uma playlist
playlistId = 'PL2155euLw9DN6EU2zBTlDj8-q572aZv7w' #Foo the Flowerhorn playlist
nextPage_token = None


# In[10]:


playlist_videos = []

# Acessa os vídeos da playlist de Flowerhorn, retorna uma lista de JSONs
res = youtube.playlistItems().list(part='snippet', playlistId = playlistId, maxResults=50).execute()
playlist_videos =res ['items']

#Extrai os ids dos vídeos da playlist
video_ids = list(map(lambda x: x['snippet']['resourceId']['videoId'], playlist_videos))
video_ids;


# In[39]:


#Criar uma nova lista JSON, com estatísticas importantes dos vídeos da playlist.
stats = []

for video_id in video_ids:
  lista_videos = youtube.videos().list(part='statistics', id=video_id).execute()
  stats += lista_videos['items']

stats


# In[12]:


video_titles = list(map(lambda x: x['snippet']['title'], playlist_videos))
thumb_urls = list(map(lambda x: x['snippet']['thumbnails']['high']['url'], playlist_videos))
video_titles = list(map(lambda x: x['snippet']['title'], playlist_videos))
published_date = list(map(lambda x: x['snippet']['publishedAt'], playlist_videos))
video_description = list(map(lambda x: x['snippet']['description'], playlist_videos))


# In[13]:


views = list(map(lambda x: int(x['statistics']['viewCount']), stats))
liked = list(map(lambda x: int(x['statistics']['likeCount']), stats))
#Antes era possível acessar as stats sobre dislikes. Mas o Youtube retirou a visibilidade dessa info e a API refletiu tal mudança.
comments = list(map(lambda x: int(x['statistics']['commentCount']), stats))
liked_view_ratio = list(map(lambda x: 100*(x[0]/x[1]), zip(liked, views)))  #fiz esse por curiosidade, mas o ratio é mtt baixo pra quase todos os vídeos =(


# In[14]:


from datetime import datetime

extraction_date = [str(datetime.now())]*len(video_ids)


# **Finalmente, colocando os dados extraídos da playlist num dataframe usando Pandas**
# 

# In[15]:


import pandas as pd
foo_playlist_df = pd.DataFrame({
    'title':video_titles,
    'video_id':video_ids,
    'published_date':published_date,
    'extraction_date':extraction_date,
    'views':views,
    'likes':liked,
    'comments_count':comments,
    'liked_to_view_ratio':liked_view_ratio,
    'thumbnail':thumb_urls
})

foo_playlist_df.head()


# In[55]:


import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import seaborn as sns

# Criar as arrays
length = len(foo_playlist_df)

x = foo_playlist_df.views.values
y = foo_playlist_df.likes.values
print(x)


# In[56]:


# Arranjar as arrays para possibilitar o fit linear
X = x[:, np.newaxis]
y = y[:, np.newaxis]
regr = linear_model.LinearRegression()
regr.fit(x, y)


# In[18]:


x_1dim = np.concatenate(x.reshape(1, -1))
y_1dim = np.concatenate(y.reshape(1, -1))
data = pd.DataFrame({'views':x_1dim, 'likes':y_1dim})


# In[19]:


# Plotar a regressão linear com sua incerteza
sns.set_style('darkgrid')
sns.lmplot(data=data, x='views', y='likes', aspect=2);


# In[20]:


r_sq = regr.score(x, y)
print('Coefficient of determination:', r_sq)


# Ou, melhor ainda, limitar os vídeos da playlist para aqueles com menos de 500k views.

# In[27]:


limited_videos_df = data[data['views'] < (0.5 * 1e7)]


# In[30]:


sns.lmplot(data=limited_videos_df, x='views',
          y='likes', aspect=1.5);


# Vamos agora os outros vídeos do canal e que não são da playlist usada pra construir o modelo

# In[152]:


playlistId2 = 'PL2155euLw9DMQrIHTkB4aLZBpMsbmc7QS'

res = youtube.playlistItems().list(part='snippet', playlistId=playlistId2, maxResults=50).execute()
playlist_videos2 = res['items']

video_ids2 = list(map(lambda x: x['snippet']['resourceId']['videoId'], playlist_videos2))

stats2 = []
for video_id in video_ids2:
    lista_videos = youtube.videos().list(part='statistics', id=video_id).execute()
    stats += lista_videos['items']


# In[164]:


views2 = list(map(lambda x: int(x['statistics']['viewCount']), stats))
views2 = np.array(views2)
views2 = views2[views2 < (0.5 * 1e7)]

Xfit = views2[:, np.newaxis]

liked2 = list(map(lambda x: int(x['statistics']['likeCount']), stats))
liked2 = np.sort(np.array(liked2))[:226]


# In[167]:


yfit = regr.predict(Xfit)


# In[170]:


sns.relplot(data=limited_videos_df, x='views', y='likes', aspect=1.5)
plt.plot(views2, yfit);

