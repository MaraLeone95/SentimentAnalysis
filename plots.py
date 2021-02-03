import pandas as pd
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import *

#shows null data (if present)
sns.heatmap(tweets_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")

#plots data according to lable (first is histogram, second counterplot)
tweets_df.hist(bins = 30, figsize = (13,5), color = 'r')
sns.countplot(data = tweets_df, x = 'label', color = 'r')

# Let's get the length of the messages
tweets_df['length'] = tweets_df['tweet'].apply(len)

#plots the length of the tweets
tweets_df['length'].plot(bins=100, kind='hist')

# Let's see the shortest message
tweets_df[tweets_df['length'] == 11]['tweet'].iloc[0]

#positive tweets
positive = tweets_df[tweets_df['label']==0]

#negative tweets
negative = tweets_df[tweets_df['label']==1]

#sentences as a list
sentences = tweets_df['tweet'].tolist()

#sentences as a string
sentences_as_one_string = " ".join(sentences)

#plots a Word Cloud
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))
