import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

train = pd.read_csv('UNSW_2018_IoT_Botnet_Final_10_best_Training.csv')
test = pd.read_csv('UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv')

X_train = train.iloc[:, 6:16]
y_train = train.iloc[:, 16:17]
X_test = test.iloc[:, 6:16]
y_test = test.iloc[:, 16:17]

classifier = HistGradientBoostingClassifier(learning_rate=0.1)
classifier = classifier.fit(X_train, y_train.values.ravel())
classifier_y_pred = classifier.predict(X_test)
classifier_acc = accuracy_score(y_test, classifier_y_pred)
print("Gradient Boosting Classifier Accuracy:", classifier_acc)
print("Gradient Boosting Classifier Classification Report:")
print(classification_report(y_test, classifier_y_pred))

# I know that n_estimators ought to be 100, but with the runtime, I think that we can throw fair comparisons to the wind
forest = RandomForestClassifier(n_estimators=10)
forest = forest.fit(X_train, y_train.values.ravel())
forest_y_pred = forest.predict(X_test)
forest_acc = accuracy_score(y_test, forest_y_pred)
print("Random Forest Accuracy:", forest_acc)
print("Random Forest Classification Report:")
print(classification_report(y_test, forest_y_pred))

# Ditto as above
extra = ExtraTreesClassifier(n_estimators=10)
extra = extra.fit(X_train, y_train.values.ravel())
extra_y_pred = extra.predict(X_test)
extra_acc = accuracy_score(y_test, extra_y_pred)
print("Extra Trees Accuracy:", extra_acc)
print("Extra Trees Classification Report:")
print(classification_report(y_test, extra_y_pred))

XGB = XGBClassifier()
XGB = XGB.fit(X_train, y_train.values.ravel())
XGB_y_pred = XGB.predict(X_test)
XGB_acc = accuracy_score(y_test, XGB_y_pred)
print("XGB Accuracy:", XGB_acc)
print("XGB Classification Report:")
print(classification_report(y_test, XGB_y_pred))

ada = AdaBoostClassifier(n_estimators = 10, algorithm='SAMME')
ada = ada.fit(X_train, y_train.values.ravel())
ada_y_pred = ada.predict(X_test)
ada_acc = accuracy_score(y_test, ada_y_pred)
print("AdaBoost Accuracy:", ada_acc)
print("AdaBoost Classification Report:")
print(classification_report(y_test, ada_y_pred))

DT_y_pred = np.ones(int(X_test.size / 10))
DT_acc = accuracy_score(y_test, DT_y_pred)
print("All Positives Accuracy:", DT_acc)
print("All Positives Classification Report:")
print(classification_report(y_test, DT_y_pred, zero_division=0.0))
