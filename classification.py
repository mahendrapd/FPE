import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv("trainfpe.csv")
cols=len(data.columns)
nfeatures = cols-1
X = data.iloc[:,0:nfeatures]  
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#clf = svm.SVC(kernel='rbf')
#clf = svm.SVC(kernel="linear")
#clf = LogisticRegression(solver='liblinear')
#clf = RandomForestClassifier(max_depth=5, random_state=0)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
report = classification_report(y_test, y_pred)
print(report)

print("Mahendra")
