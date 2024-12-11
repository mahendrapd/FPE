#Feature Probability Estimation based Feature Selection Method (FPE)
#This code is faster than previous code.
#Here, We have used Python Dictionary Concept to make less comparisions and iterations
#It works for numeric and String data

import pandas as pd
import numpy as np
import numbers
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
data = pd.read_csv("train.csv")

def fpefs(data):
    df=pd.DataFrame(data)
    col=len(df.columns)
    nfeatures=col-1
    del df #memory is released
    X = data.iloc[:,0:nfeatures]  
    y = data.iloc[:,-1]
    size=len(X)
    for f in range (nfeatures):
        K=X.iloc[:,f]
        da=K.iloc[0]
        if ((isinstance(da,numbers.Number) or isinstance(da,numbers.Real)) and (K.max()-K.min())>0) :
            K=round(((K-K.min())/(K.max()-K.min())),2)
            X.iloc[:,f]=pd.Series(K)
    #print(col,size) #print columns and rows of training set
    #print(X.head()) #print first five rows data 
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
        #ulist=unique(X.iloc[:,j])
        dp=pd.Series(X.iloc[:,j])
        md=dp.groupby(dp).apply(lambda px: px.index.tolist()).to_dict()
        ulist=md.keys()
        sizeulist=len(ulist)
        for m in ulist:
            mlist=[]
            mindex=md[m]
            msize=len(mindex)
            for i in range(msize):
                if y[mindex[i]] not in mlist:
                    mlist.append(y[mindex[i]])
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
        print(j,mu,ncount,sizeulist,Pvalue) #print feature idex, mu value, distinct values of features, and probability 
    Prob=pd.Series(Prob) #Estimated probability features-wise
    dfcolumns = pd.DataFrame(X.columns)
    estimatedProbability = pd.concat([dfcolumns,Prob],axis=1)
    return estimatedProbability #returns feature names and estimated probability
P=fpefs(data);
print(P)

