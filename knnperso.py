from tp_knn_source import *

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as mp
import scipy.stats as st

############################################################################
#     K-NN
############################################################################

class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=1, weights=None, h=1e3):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.h = h

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        # TODO : Compute all pairwise distances between X and self.X_
        d = mp.pairwise_distances(self.X_,X)

        # TODO : Find the predicted labels y for each entry in X
        y_predict = np.zeros(len(X))
        
        if self.weights is None:
            self.weights = np.ones_like(d)
        else:
            self.weights = get_weights(d,self.h)
        
        for i in range(len(X)):
            #y_predict[i] = st.mode(self.y_[d[:,i].argsort()[0:self.n_neighbors]])[0]
            y_neigh = np.array(self.y_[d[:,i].argsort()[0:self.n_neighbors]],'int')
            w_neigh = self.weights[d[:,i].argsort()[0:self.n_neighbors],i]
            y_predict[i] = np.bincount(y_neigh,weights=w_neigh).argmax()
        # You can use the scipy.stats.mode function

        return y_predict

def get_weights(dist,h):
    """Returns an array of weights, exponentially decreasing in the square
    of the distance.

    Parameters
    ----------
    dist : a one-dimensional array of distances.

    Returns
    -------
    weight : array of the same size as dist
    """
    w = np.exp(-dist*dist/h)
    return w

#test
x,y = rand_bi_gauss(n1=100, n2=100, mu1=[0.5, 0.5], mu2=[-0.5, -0.5], sigma1=[1, 1], sigma2=[1, 1])
#x,y = rand_tri_gauss()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3)
my = KNNClassifier(n_neighbors=10,weights='function')
#my = KNNClassifier(n_neighbors=10)
my.fit(x_train,y_train)
y_model = my.predict(x_test)
test = sum(y_model==y_test)/len(y_test)

#check
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(x_train, y_train)
y_model2 = neigh.predict(x_test)
test2 = sum(y_model2==y_test)/len(y_test)