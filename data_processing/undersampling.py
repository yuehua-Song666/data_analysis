# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:26:09 2022

@author: yuehua
"""
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
# check version number
from imblearn.under_sampling import NearMiss

X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# print(X[0])
# (10000, 2) (10000,)
# Counter({0: 9900, 1: 100})
counter= Counter(y)
# print(counter.items())
for label,_ in counter.items():
    row = where(y==label)[0]
    pyplot.scatter(X[row,0],X[row,1],label=str(label))
pyplot.legend()
pyplot.show()

'''
undersample = NearMiss(version=1, n_neighbors=3)
X, y = undersample.fit_resample(X, y)
reCounter = Counter(y)
for label,_ in reCounter.items():
    row = where(y==label)[0]
    pyplot.scatter(X[row,0],X[row,1],label=str(label))
pyplot.legend()
pyplot.show()


undersample = NearMiss(version=2, n_neighbors=3)
X, y = undersample.fit_resample(X, y)
reCounter = Counter(y)
for label,_ in reCounter.items():
    row = where(y==label)[0]
    pyplot.scatter(X[row,0],X[row,1],label=str(label))
pyplot.legend()
pyplot.show()

'''
undersample = NearMiss(version=3, n_neighbors=3)
X, y = undersample.fit_resample(X, y)
reCounter = Counter(y)
for label,_ in reCounter.items():
    row = where(y==label)[0]
    pyplot.scatter(X[row,0],X[row,1],label=str(label))
pyplot.legend()
pyplot.show()

    