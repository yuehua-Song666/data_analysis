from matplotlib.pyplot import subplots
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

filename = 'Iris\\Iris.csv'
#names = ['separ-length', 'separ-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename).drop(labels='Id',axis=1)
dataset_with_y = dataset
#print('Dimension: row: %s, colum: %s'%dataset.shape)
#print(dataset.head(10))
#print(dataset.describe())
#print(dataset.groupby('Species').size())
dataset = dataset.drop(labels='Species',axis=1)
#boxplot = dataset.boxplot(column=['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm', 'PetalLengthCm'])
#histplot = dataset.hist(column=['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm', 'PetalLengthCm'])
#scatter_matrix(dataset)

array = dataset_with_y.values
#print(array)
X = array[: , 0:4]
#print(X)
Y = array[: , 4]
#print(Y)
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=validation_size, random_state=seed)

models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()

results = []
for key in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(models[key], X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    print('%s: %f (%f)' %(key, cv_results.mean(), cv_results.std()))
    



