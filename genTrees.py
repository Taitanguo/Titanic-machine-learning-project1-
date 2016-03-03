#!/usr/bin/python

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pickle

#depth is the depth of the tree to to be used
def makeTree(depth,x,y):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=0)


    ### Fit tree6 to training data ###
    tree.fit(x,y)

    ### Displaying results ###
    from sklearn.tree import export_graphviz
    from sklearn.externals.six import StringIO
    import pydot
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data,feature_names=['pclass','sex','age','sibsp','parch'])
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("/Users/Uvaguy/Desktop/ml/titannic/genfamtree.pdf")

    # Now we calculate our accuracy and create a confusion matrix of our results

#    from sklearn.metrics import accuracy_score
#    print('Accuracy: %.2f' % accuracy_score(Y_test,y_pred))
#    from sklearn.metrics import confusion_matrix
#    confmat=confusion_matrix(y_true=Y_test, y_pred=y_pred)
#    print(confmat)


### master script ###
data=pd.read_excel('/Users/Uvaguy/Desktop/ml/titannic/titanic3.xls',header=0)

#Remove question marks and replace with NaN
#bc=bc.replace(to_replace='?',value=0)

#Impute medians to address NaN
#imput=preprocessing.Imputer(missing_values='NaN',strategy='median')
#bc=imput.fit_transform(bc)
fam=pickle.load(open('fam.pickle','rt'))
nonfam=pickle.load(open('nonfam.pickle','rt'))

#get row indexes to agree for dataframe
for i in range(len(fam)):
    fam[i]=fam[i]-1

for i in range(len(nonfam)):
    nonfam[i]=nonfam[i]-1

famData=pd.DataFrame(data,index=fam)
nonfamData=pd.DataFrame(data,index=nonfam)

#training data
xfamData=pd.DataFrame(famData,columns=['pclass','sex','age','sibsp','parch'])
for i in range(len(xfamData)):
    if xfamData['sex'][fam[i]]=='male':
        xfamData['sex'][fam[i]]=0
    else:
        xfamData['sex'][fam[i]]=1

#target values
yfamData=pd.DataFrame(famData,columns=['survived'])

xnonfamData=pd.DataFrame(nonfamData,columns=['pclass','sex','age','sibsp','parch'])
ynonfamData=pd.DataFrame(nonfamData,columns=['survived'])

print famData
print nonfamData
makeTree(3,xfamData,yfamData)


