# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:19:49 2022

@author: yuehua
"""

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import CondensedNearestNeighbour
from matplotlib import pyplot
from numpy import where

X,y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
counter = Counter(y)
print(counter)
undersample = CondensedNearestNeighbour(n_neighbors=1)
X,y = undersample.fit_resample(X, y)
counter = Counter(y)

for label,_ in counter.items():
    row = where(label==y)[0]
    pyplot.scatter(X[row,0], X[row,1], label=str(label))
pyplot.legend()
pyplot.show()