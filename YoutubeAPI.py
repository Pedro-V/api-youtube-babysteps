#!/usr/bin/env python
# coding: utf-8
# Esse arquivo é uma versão bruta do notebook principal

# <a href="https://colab.research.google.com/github/Pedro-V/api-youtube-babysteps/blob/main/YoutubeAPI.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


get_ipython().system('pip install --upgrade google-api-python-client;')


# In[2]:


from googleapiclient.discovery import build


# In[3]:


#Os passos para obter uma key para a API podem ser vistos nesse vídeo do canal Programação Dinâmica: https://www.youtube.com/watch?v=olDCJ1w3FLM
youtubeApiKey = "XXXXXXXXXX"

youtube = build('youtube','v3', developerKey=youtubeApiKey)


# In[4]:


#extraindo dados de uma playlist
playlistId = 'PL2155euLw9DN6EU2zBTlDj8-q572aZv7w' #Foo the Flowerhorn playlist
nextPage_token = None


# In[5]:


playlist_videos = []

# Acessa os vídeos da playlist de Flowerhorn, retorna uma lista de JSONs
res = youtube.playlistItems().list(part='snippet', playlistId = playlistId, maxResults=50).execute()
playlist_videos =res ['items']

#Extrai os ids dos vídeos da playlist
video_ids = list(map(lambda x: x['snippet']['resourceId']['videoId'], playlist_videos))
video_ids;


# In[6]:


#Criar uma nova lista JSON, com estatísticas importantes dos vídeos da playlist.
stats = []

for video_id in video_ids:

  lista_videos = youtube.videos().list(part='statistics', id=video_id).execute()
  stats += lista_videos['items']

stats


# In[7]:


video_titles = list(map(lambda x: x['snippet']['title'], playlist_videos))
thumb_urls = list(map(lambda x: x['snippet']['thumbnails']['high']['url'], playlist_videos))
video_titles = list(map(lambda x: x['snippet']['title'], playlist_videos))
published_date = list(map(lambda x: x['snippet']['publishedAt'], playlist_videos))
video_description = list(map(lambda x: x['snippet']['description'], playlist_videos))


# In[8]:


views = list(map(lambda x: int(x['statistics']['viewCount']), stats))
liked = list(map(lambda x: int(x['statistics']['likeCount']), stats))
#Antes era possível acessar as stats sobre dislikes. Mas o Youtube retirou a visibilidade dessa info e a API refletiu tal mudança.
comments = list(map(lambda x: int(x['statistics']['commentCount']), stats))
liked_view_ratio = list(map(lambda x: 100*(x[0]/x[1]), zip(liked, views)))  #fiz esse por curiosidade, mas o ratio é mtt baixo pra quase todos os vídeos =(


# In[9]:


from datetime import datetime

extraction_date = [str(datetime.now())]*len(video_ids)


# **Finalmente, colocando os dados extraídos da playlist num dataframe usando Pandas**
# 

# In[10]:


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


# In[11]:


import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import seaborn as sns

# Criar as arrays
length = len(foo_playlist_df)

x = foo_playlist_df.views.values
y = foo_playlist_df.likes.values
print(x)


# In[12]:


# Arranjar as arrays para possibilitar o fit linear
x = x.reshape(length,1)
y = y.reshape(length,1)
regr = linear_model.LinearRegression()
regr.fit(x, y)


# In[13]:


x_1dim = np.concatenate(x.reshape(1, -1))
y_1dim = np.concatenate(y.reshape(1, -1))
data = pd.DataFrame({'views':x_1dim, 'likes':y_1dim})


# In[14]:


# Plotar a regressão linear com sua incerteza
sns.set_style('darkgrid')
sns.lmplot(data=data, x='views', y='likes', aspect=2);


# In[15]:


r_sq = regr.score(x, y)
print('Coefficient of determination:', r_sq)

