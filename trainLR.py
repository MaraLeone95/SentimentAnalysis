
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from datasets import *

#splits the dataset into trainset and testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#trains the model with logistic regression classifier
LR_classifier = LogisticRegression()
LR_classifier.fit(X_train, y_train)