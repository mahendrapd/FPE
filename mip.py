import pandas as pd
import numpy as np
import random
#import time
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

data = pd.read_csv("normdata.csv")
data = data.drop_duplicates()
size=len(data)
nf=len(data.columns)
print(nf)
print(size)
nfeatures = nf-1
X = data.iloc[:,0:nfeatures]  
y = data.iloc[:,-1]

#start_time = time.perf_counter()

mi = mutual_info_classif(X, y, random_state=42)

#mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
mi_scores = pd.Series(mi).sort_values(ascending=False)
print("Mutual Information Scores:")
print(mi_scores)
#df=igvalue.sort_values()
idx=list(mi_scores.index)
print(idx)
print(idx[:39])
X_new = X.iloc[:, idx[:39]]
y_new=pd.Series(y)
trainMI=pd.concat([X_new,y_new],axis=1)
trainMI.to_csv('trainMI.csv',header=False, index=False)
print(trainMI.head())
print(trainMI.tail())

# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"Execution time: {elapsed_time:.6f} seconds")

# Select top 2 features
# selector = SelectKBest(mutual_info_classif, k=39)
# X_selected = selector.fit_transform(X, y)

# print("\nSelected Top 39 Features:")
# P=list(X.columns[selector.get_support()])
# print(P)
print("Mahendra")
