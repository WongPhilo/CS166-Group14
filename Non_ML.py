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

# Train Decision Tree Model
DT_clf = DecisionTreeClassifier()
DT_clf = DT_clf.fit(X_train, y_train)
DT_y_pred = DT_clf.predict(X_test)
DT_acc = accuracy_score(y_test, DT_y_pred)
print("Decision Tree Accuracy:", DT_acc)
print("Decision Tree Classification Report:")
print(classification_report(y_test, DT_y_pred))

# Train K-nearest neighbors model
KNN_clf = KNeighborsClassifier(n_neighbors=5)
KNN_clf = KNN_clf.fit(X_train, y_train.values.ravel())
KNN_y_pred = KNN_clf.predict(X_test)
KNN_acc = accuracy_score(y_test, KNN_y_pred)
print("KNN Accuracy:", KNN_acc)
print("KNN Classification Report:")
print(classification_report(y_test, KNN_y_pred))

# Train Naive Bayes Model
NB_clf = GaussianNB()
NB_clf = NB_clf.fit(X_train, y_train.values.ravel())
NB_y_pred = NB_clf.predict(X_test)
NB_acc = accuracy_score(y_test, NB_y_pred)
print("Naive Bayes Accuracy:", NB_acc)
print("Naive Bayes Classification Report:")
print(classification_report(y_test, NB_y_pred))

# Signature-based Detection
signatures = [
    r'^.* ICMP .*$',                   # ICMP traffic
    r'^.* TCP .*\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:23\b.*$',   # Telnet traffic (port 23)
    r'^.* TCP .*\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:2323\b.*$', # Telnet traffic (port 2323)
    r'^.* TCP .*\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:5555\b.*$'  # Backdoor traffic (port 5555)
]

def match_signatures(traffic):
    for signature in signatures:
        if re.match(signature, traffic, re.IGNORECASE):
            return 1  # Botnet activity detected
    return 0  # No botnet activity detected

df_test['Signature_Detection'] = df_test['Activity'].apply(match_signatures)

# Evaluation and Comparison
true_labels = df_test['Label'].values
signature_labels = df_test['Signature_Detection'].values

signature_accuracy = (true_labels == signature_labels).mean()
print(f"Signature-based Detection Accuracy: {signature_accuracy:.4f}")

# Compare with ML model accuracies
print(f"Decision Tree Accuracy: {DT_acc:.4f}")
print(f"KNN Accuracy: {KNN_acc:.4f}")
print(f"Naive Bayes Accuracy: {NB_acc:.4f}")
