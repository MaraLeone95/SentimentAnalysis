
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from datasets import *

#splits the dataset into trainset and testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#trains the model with Naive Bayes classifier model
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

