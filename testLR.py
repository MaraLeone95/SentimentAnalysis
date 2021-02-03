
from sklearn.metrics import classification_report, confusion_matrix
from trainLR import *
import seaborn as sns

# Predicting the Test set results
y_predict_test = LR_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)

#plots the results
sns.heatmap(cm, annot=True)

#classification report
print(classification_report(y_test, y_predict_test))