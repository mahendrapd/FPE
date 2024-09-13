import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

##data = pd.read_csv("train.csv")
###data.dropna(inplace=True)
##X = data.iloc[:,0:79]  
##y = data.iloc[:,-1]
##a=np.array(X)
##size=len(X)
##nf=79
##X=round(((X-X.min())/(X.max()-X.min())),2)
#print(X.head())
##for i in range(size):
##    for j in range(nf):
##        a[i][j]=float(a[i][j])
##for j in range(nf):
##    xmin=a[j].min()
##    xmax=a[j].max()
##    for i in range(size):
##        a[i][j]=round(float((a[i][j]-xmin)/(xmax-xmin)),2)
##    print(a[0][j])
#X=a
##P=pd.concat([X,y],axis=1)
##print(P.head())
##P.to_csv('data.csv', header=True, index=False)


##data = pd.read_csv("normdata.csv")
##print(len(data))
##data = data.drop_duplicates()
##data.to_csv('datanoduplicate.csv', header=True, index=False)
##print(len(data))

data = pd.read_csv("datanoduplicate.csv")
nfeatures = 69
X = data.iloc[:,0:nfeatures]  
y = data.iloc[:,-1]
'''
fd=pd.read_csv("FPE.CSV")
dd=fd.iloc[:,-1]
idx=[]
for i in range(len(dd)):
    if(dd.values[i]>0.6):
        idx.append(i)
#print(idx)
#print(len(idx))
X_new = X.iloc[:, idx]
y_new=pd.Series(y)
trainfpe=pd.concat([X_new,y_new],axis=1)
trainfpe.to_csv('trainfpe.csv',header=False, index=False)
print(trainfpe.head())
'''
#KBest Feature Selection Method (Method 1)
'''
bestfeatures = SelectKBest(score_func=chi2, k=4)
fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfscores = round(dfscores,2)
#dfcolumns = pd.DataFrame(X.columns)
#featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#featureScores.to_csv('Kbestudata.csv', header=False, index=False)
df=dfscores.sort_values(by=[0],ascending=False)
idx=list(df.index)
#print(idx[:39])
X_new = X.iloc[:, idx[:39]]
y_new=pd.Series(y)
trainkbest=pd.concat([X_new,y_new],axis=1)
trainkbest.to_csv('trainkbest.csv',header=False, index=False)
print(trainkbest.head())
'''


#Feature Probability Estimation based Feature Selection Method (FPE)(Method_2)
'''
def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
size=len(X)
label=unique(y)
numberOfLabel=len(label)
Prob=[]
for j in range(nfeatures):
    ulist=[]
    mu=0
    ncount=0
    Pvalue=0
    ulist=unique(X.iloc[:,j])
    sizeulist=len(ulist)
    for m in range(sizeulist):
        mlist=[]
        for i in range(size):
            if(ulist[m] == X.iloc[i,j]):
                if y[i] not in mlist:
                    mlist.append(y[i])
                if(numberOfLabel == len(mlist)):
                    break
        nc=len(mlist)
        if(nc<numberOfLabel):
            mu=mu+float(nc)/numberOfLabel
            ncount=ncount+1
    if(ncount==0):
        Pvalue=0
    else:
        Pvalue=round((1-(float(1)/2)*((float(sizeulist-ncount)/sizeulist)+(float(mu)/ncount))),2)
    Prob.append(Pvalue)
    print(j,mu,ncount,sizeulist,Pvalue)
Prob=pd.Series(Prob)
dfcolumns = pd.DataFrame(X.columns)
estimatedProbability = pd.concat([dfcolumns,Prob],axis=1)
estimatedProbability.to_csv('FPE.csv', header=False, index=False)
'''


#Recursive Feature Elimination (RFE) Method (Method_3)
'''
model=LogisticRegression(max_iter=1000)
rfe=RFE(model, n_features_to_select=1)
fit=rfe.fit(X,y)
frank=fit.ranking_
frank=pd.Series(frank)

#dfcolumns = pd.DataFrame(X.columns)
#featurerank=pd.concat([dfcolumns,frank],axis=1)
#featurerank.to_csv('RFE.csv', header=False, index=False)
#print(frank)
df=frank.sort_values()
idx=list(df.index)
#print(idx)
#print(idx[:39])
X_new = X.iloc[:, idx[:39]]
y_new=pd.Series(y)
trainrfe=pd.concat([X_new,y_new],axis=1)
trainrfe.to_csv('trainrfe.csv',header=False, index=False)
print(trainrfe.head())
'''

#Feature Importance method (Method_4)
'''
model=ExtraTreesClassifier()
model.fit(X,y)
fimp=model.feature_importances_
fimp=np.round(fimp,3)
fimp=pd.Series(fimp)
#dfcolumns = pd.DataFrame(X.columns)
#featureimportance=pd.concat([dfcolumns,fimp],axis=1)
#featureimportance.to_csv('Fimportance.csv', header=False, index=False)
print(fimp)
df=fimp.sort_values(ascending=False)
#print(df)
idx=list(df.index)
#print(idx)
X_new = X.iloc[:, idx[:39]]
y_new=pd.Series(y)
trainfimp=pd.concat([X_new,y_new],axis=1)
trainfimp.to_csv('trainfimp.csv',header=False, index=False)
print(trainfimp.head())
print(idx[:39])
'''
#PCA (Method_5)
'''
nf = 39
pca = PCA(n_components=nf)
pca.fit(X)
X_new = pca.transform(X)
X_new=pd.DataFrame(X_new)
y_new=pd.Series(y)
trainpca=pd.concat([X_new,y_new],axis=1)
trainpca.to_csv('trainpca.csv',header=False, index=False)
#print(X_new)
'''
print("Mahendra")
