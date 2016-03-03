import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

class decisionTree(object):
    def __init__(self, fileName, header = 0):
        #Pull in data
        self.data=pd.read_excel(fileName,header=header)
        self.data=pd.DataFrame(self.data)

    def replace(self,col,tobe,be):
        self.data[col]=self.data[col].replace(to_replace=tobe,value=be)

    def dupCol(self,oldCol,newCol):
        self.data[newCol] = self.data[oldCol]
    
    def impute(self, col):
        #Impute medians to address NaN
        imput=preprocessing.Imputer(missing_values='NaN',strategy='median',axis = 1)
        tmp=imput.fit_transform(self.data[col]).T
        self.data[col]=tmp
	
    def printData(self):
        print self.data.head()

    def dropCol(self,col):
        #Drop first column
        self.data=self.data.drop(col,axis=1)

    def createTestSet(self,xSet,y):
	    #Create trainning and testing sets
        self.X=self.data.ix[:,xSet]
        self.Y=self.data.ix[:,y]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=0)


    def scale(self):
        """ The data is scaled in preparation for creating the DT.  WHY ?  
        WHAT IS THE DIFFERENCE BETWEEN SCALING AND NORMALIZATION?"""
        sc = StandardScaler()
        sc=sc.fit(X_train)
        self.X_train = sc.transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

    def buildTree(self,depth):
        #Here, we define the parameters of our tree and use a feature selection algorithm (RFE) to pick out the strongest features.

        self.tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=depth, random_state=0)
        selector = RFE(self.tree, 2, step=1)
        selector = selector.fit(self.X_train, self.Y_train)
        selector.support_
        selector.ranking_

    def fitTree(self):
        # We then fit the tree to our training data 

        self.tree.fit(self.X_train, self.Y_train)

    def visTree(self):
        # Now we visualize our tree

        export_graphviz(self.tree, out_file='tree.dot',feature_names=['pclass', 'age','sibsp','parch','fare','male','female'])

    def pred(self):
        # Let's make a prediction

        self.Y_pred=self.tree.predict(self.X_test)

    def confusionMatrix(self):
        # Now we calculate our accuracy and create a confusion matrix of our results

        print('Accuracy: %.2f' % accuracy_score(self.Y_test,self.Y_pred))
        
        confmat=confusion_matrix(y_true=self.Y_test, y_pred=self.Y_pred)
        print(confmat)



##################################################
#  feature selection
##################################################

fileName = "data/titanic3.xls"
d = decisionTree(fileName)
d.impute('age')
d.impute('fare')

d.dropCol('home.dest')
d.dropCol('boat')
d.dropCol('embarked')
d.dropCol('cabin')
d.dropCol('name')
d.dropCol('body')

#preprocessing.OneHotEncoder().fit(d.data)

d.dupCol('sex','male')
d.dupCol('sex','female')
d.replace('male','male',1)
d.replace('male','female',0)
d.replace('female','male',0)
d.replace('female','female',1)
d.printData()

xSet = [0,3,4,5,7,8,9]
y = 1
d.createTestSet(xSet,y)

d.buildTree(3)
d.fitTree()
d.visTree()



'''
X, y = iris.data, iris.target
print X.shape
lr = LogisticRegression(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lr, prefit=True)
X_new = model.transform(X)
print X_new.shape
print(X_new)
print(X)
'''



"""Now you explore on your own.  Take a look at the following link:
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
Experiment with the various parameters.  What do they mean?  How do they change your
results?"""

"""Try loading another dataset from the UCI dataset site:
(you will need pandas imported for this)
ds = pd.read_csv('https://archive.ics.uci.edu/ml/'
machine-learning-databases/yourDataChoice/yourDataChoice.data', header=None)
"""

""" The following code will allow you to experiment with pruning a tree with your chosen
dataset:


#In order to find the optimal number of leaves we can use cross validated scores on the data:

scores = tree.prune_path(clf, X_train_std, y_train,max_n_leaves=20, n_iterations=10, random_state=0)

#In order to plot the scores one can use the following function:

def plot_pruned_path(scores, with_std=True)

#Plots the cross validated scores versus the number of leaves of trees:

import matplotlib.pyplot as plt
means = np.array([np.mean(s) for s in scores])
stds = np.array([np.std(s) for s in scores]) / np.sqrt(len(scores[1]))

x = range(len(scores) + 1, 1, -1)

plt.plot(x, means)
if with_std:
    plt.plot(x, means + 2 * stds, lw=1, c='0.7')
    plt.plot(x, means - 2 * stds, lw=1, c='0.7')

    plt.xlabel('Number of leaves')
    plt.ylabel('Cross validated score')
"""
