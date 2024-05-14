import pandas as pd
import numpy as np
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load the datasets
df_train = pd.read_csv('UNSW_2018_IoT_Botnet_Final_10_best_Training.csv')
df_test = pd.read_csv('UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv')

# Prepare the data for ML models
X_train = df_train.iloc[:, 6:16]
y_train = df_train.iloc[:, 16:17]
X_test = df_test.iloc[:, 6:16]
y_test = df_test.iloc[:, 16:17]

# # Train Decision Tree Model
# DT_clf = DecisionTreeClassifier()
# DT_clf = DT_clf.fit(X_train, y_train)
# DT_y_pred = DT_clf.predict(X_test)
# DT_acc = accuracy_score(y_test, DT_y_pred)
# print("Decision Tree Accuracy:", DT_acc)
# print("Decision Tree Classification Report:")
# print(classification_report(y_test, DT_y_pred))

DT_y_pred = np.ones(int(X_test.size / 10))
DT_acc = accuracy_score(y_test, DT_y_pred)
print("All Positives Accuracy:", DT_acc)
print("All Positives Classification Report:")
print(classification_report(y_test, DT_y_pred, zero_division=0.0))
