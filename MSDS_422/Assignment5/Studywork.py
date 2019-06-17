# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:19:34 2019

@author: vtika
"""
import numpy as np
import sklearn
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import tft
from tensorflow.examples.tutorials.mnist import input_data
import sklearn.decomposition

from sklearn.datasets import fetch_openml
mnist = fetch_openml('MNIST original', version = 1)


mnist = input_data.read_data_sets('MNIST original', one_hot=True)

mnist = tf.keras.datasets.mnist.load_data()

X = mnist

tf.p

np.array(X, dtype = object)

pca = PCA(n_components=.95) ##SETING NUMBER OF COMPONENTS
X2D = pca.fit_transform(X)

pca.components_.T[:,0] ##describes first principal component

xx = pca.explained_variance_ratio_ ##explains variance of each principal component

#NP way of doing PCA
X_centered = X - X.mean( axis = 0) 
U, s, Vt = np.linalg.svd( X_centered) 
c1 = Vt.T[:, 0] 
c2 = Vt.T[:, 1]

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= .95) + 1

#best way is to set the number of components equal to the ratio variance of what
#you want
pca = PCA(n_components=.95)
X_reduced = pca.fit_transform(X)