import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

train = pd.read_csv('UNSW_2018_IoT_Botnet_Final_10_best_Training.csv')
test = pd.read_csv('UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv')

X_train = train.iloc[:, 6:16]
y_train = train.iloc[:, 17:18]
X_test = test.iloc[:, 6:16]
y_test = test.iloc[:, 17:18]

DT_clf = DecisionTreeClassifier()
DT_clf = DT_clf.fit(X_train,y_train)
DT_y_pred = DT_clf.predict(X_test)
acc = accuracy_score(y_test, DT_y_pred)
print("Accuracy:", acc)
print("Classification Report:")
print(classification_report(y_test, DT_y_pred))

from sklearn.neighbors import KNeighborsClassifier

KNN_clf = KNeighborsClassifier(n_neighbors=5)
KNN_clf = KNN_clf.fit(X_train,y_train.values.ravel()) #using ravel to change column vector to a 1-D array
KNN_y_pred = KNN_clf.predict(X_test)
KNN_acc = accuracy_score(y_test, KNN_y_pred)
print("Accuracy:", KNN_acc)
print("Classification Report:")
print(classification_report(y_test, KNN_y_pred))

from sklearn.naive_bayes import GaussianNB

NB_clf = GaussianNB()
NB_clf = NB_clf.fit(X_train,y_train.values.ravel())  #using ravel to change column vector to a 1-D array
NB_y_pred = NB_clf.predict(X_test)
NB_acc = accuracy_score(y_test, NB_y_pred)
print("Accuracy:", NB_acc)
print("Classification Report:")
print(classification_report(y_test, NB_y_pred))