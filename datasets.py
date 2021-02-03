
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


# Load the data
tweets_df = pd.read_csv('C:/Users/maral/Downloads/twitter.csv')

#drops the id axis (we don't need that data)
tweets_df = tweets_df.drop(['id'], axis = 1)

# Pipeline to clean up all the messages: (1) with list comprehension, (2) with classic for loop
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean

def message_cleaning2(message):
    Test_punc_removed = []
    Test_punc_removed_join_clean = []
    for char in message:
        if char not in string.punctuation:
            Test_punc_removed.append(char)

    Test_punc_removed_join = ''.join(Test_punc_removed)

    for word in Test_punc_removed_join.split():
        if word.lower() not in stopwords.words('english'):
            Test_punc_removed_join_clean.append(word)

    return Test_punc_removed_join_clean



# Define the cleaning pipeline we defined earlier - returns vectorized, cleaned tweets
vectorizer = CountVectorizer(analyzer = message_cleaning)
tweets_countvectorizer = CountVectorizer(analyzer = message_cleaning).fit_transform(tweets_df['tweet']).toarray()


#final data
X = tweets_countvectorizer
y = tweets_df['label']