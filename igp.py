import pandas as pd
import numpy as np
from math import log2
# import time

# start_time = time.perf_counter()

def entropy(column):
    values, counts = np.unique(column, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(data, feature, target):
    total_entropy = entropy(data[target])

    # Values and counts for the feature
    values, counts = np.unique(data[feature], return_counts=True)

    # Weighted entropy after the split
    weighted_entropy = np.sum([
        (counts[i] / np.sum(counts)) * entropy(data[data[feature] == values[i]][target])
        for i in range(len(values))
    ])

    # Information Gain = entropy before - weighted entropy after
    ig = total_entropy - weighted_entropy
    return ig

data = pd.read_csv("normdata.csv")
data = data.drop_duplicates()
size=len(data)
nf=len(data.columns)
print(nf)
print(size)
print(type(nf))
X = data.iloc[:,0:nf-1]  
y = data.iloc[:,-1]
result = data.to_dict()
P=list(result.keys())

features=P[:69]
target=P[-1]
# print(features)
# print(target)

print("Information Gain for each feature:\n")
igvalue=[]
for feature in features:
    ig_value = information_gain(data, feature, target)
    print(f"{feature:12s}: {ig_value:.4f}")
    ival=round(ig_value,4)
    igvalue.append(ival)
#print(igvalue)
features=pd.Series(features)
igvalue=pd.Series(igvalue)
InfoGain=pd.concat([features,igvalue],axis=1)
#print(InfoGain)
df=igvalue.sort_values(ascending=False)
print(df)
idx=list(df.index)
print(idx)
print(idx[:39])
X_new = X.iloc[:, idx[:39]]
y_new=pd.Series(y)
trainIG=pd.concat([X_new,y_new],axis=1)
trainIG.to_csv('trainIG.csv',header=False, index=False)
print(trainIG.head())
print(trainIG.tail())

# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"Execution time: {elapsed_time:.6f} seconds")

print("Mahendra")
