
# coding: utf-8

# In[ ]:


# Jump-Start for the Bank Marketing Study
# as described in Marketing Data Science: Modeling Techniques
# for Predictive Analytics with R and Python (Miller 2015)

# jump-start code revised by Thomas W. Milller (2018/10/07)

# Scikit Learn documentation for this assignment:
# http://scikit-learn.org/stable/auto_examples/classification/
#   plot_classifier_comparison.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB.score
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.LogisticRegression.html
# http://scikit-learn.org/stable/modules/model_evaluation.html 
# http://scikit-learn.org/stable/modules/generated/
#  sklearn.model_selection.KFold.html

# prepare for Python version 3x features and functions
# comment out for Python 3.x execution
# from __future__ import division, print_function
# from future_builtins import ascii, filter, hex, map, oct, zip


# In[126]:


# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# import base packages into the namespace for this program
import numpy as np
import pandas as pd


# In[127]:


# initial work with the smaller data set
bank = pd.read_csv('bank.csv', sep = ';')  # start with smaller data set
# examine the shape of original input data
print(bank.shape)


# In[128]:


# drop observations with missing data, if any
bank.dropna()
# examine the shape of input data after dropping missing data
print(bank.shape)


# In[129]:


# look at the list of column names, note that y is the response
list(bank.columns.values)


# In[130]:


# look at the beginning of the DataFrame
bank.head()


# In[131]:


# mapping function to convert text no/yes to integer 0/1
convert_to_binary = {'no' : 0, 'yes' : 1}


# In[132]:


# define binary variable for having credit in default
default = bank['default'].map(convert_to_binary)


# In[133]:


# define binary variable for having a mortgage or housing loan
housing = bank['housing'].map(convert_to_binary)


# In[134]:


# define binary variable for having a personal loan
loan = bank['loan'].map(convert_to_binary)

# define response variable to use in the model
response = bank['response'].map(convert_to_binary)


# In[135]:


# gather three explanatory variables and response into a numpy array 
# here we use .T to obtain the transpose for the structure we want
model_data = np.array([np.array(default), np.array(housing), np.array(loan), 
    np.array(response)]).T


# In[136]:


# examine the shape of model_data, which we will use in subsequent modeling
print(model_data.shape)


# In[137]:


# the rest of the program should set up the modeling methods
# and evaluation within a cross-validation design


# In[138]:


features = model_data[:,:3]
output = model_data[:,-1]


# # Logistic Regression Modelling method

# In[139]:


# importing the libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[140]:


# spliting the dataset between train and test. 
# training dataset is 70% and test data is 30%
X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.30, random_state=42)


# In[141]:


# Logistic Regression classifier using scikit learn
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
clf.fit(X_train, y_train)


# In[142]:


# k fold cross validation where k=5
# validation set is prepared from the training data internally by scikit learn.

k=5
logistic_scores = cross_val_score(clf, X_train, y_train, cv=k)


# In[144]:


# calculate the accuracy and standard deviation in the accuracy
logistic_accuracy = logistic_scores.mean()
logistic_std = logistic_scores.std()

print("Training Accuracy on logistic regression: %0.2f (+/- %0.2f)" % (logistic_accuracy, logistic_std * 2))


# This means that logsitic regression model is 88% accurately trained and with
# 0 standard deviation on the training data.

# In[145]:


# test predictions
y_predict = clf.predict(X_test)


# In[146]:


print('Test accuracy on logistic regression: %0.2f' % clf.score(X_test, y_test))


# In[147]:


# ROC plot
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=2)


# the warning tells us that model is too much biased but since the accuracy is good, 
# that means training data itself is biased. there are 4000 records with no as the 
# response and 500 records with yes as response.

# In[148]:


print('no count: ', bank[bank['response']=='no'].count()[0])
print('yes count: ', bank[bank['response']=='yes'].count()[0])


# # Naive Bayes Classifier modelling technique

# In[149]:


# importing the naive bayes module
from sklearn.naive_bayes import GaussianNB


# In[150]:


# gaussian naive bayes model instance
gnb = GaussianNB()

# training on the training data
gnb.fit(X_train, y_train)


# In[151]:


# k fold cross validation where k=5
# validation set is prepared from the training data internally by scikit learn.

k=5
gnb_scores = cross_val_score(gnb, X_train, y_train, cv=k)


# In[153]:


# calculate the accuracy and standard deviation in the accuracy
gnb_accuracy = gnb_scores.mean()
gnb_std = gnb_scores.std()

print("Training Accuracy on gnb: %0.2f (+/- %0.2f)" % (gnb_accuracy, gnb_std * 2))


# In[154]:


# test predictions
y_predict = gnb.predict(X_test)


# In[155]:


print('Test accuracy on gnb: %0.2f' % gnb.score(X_test, y_test))


# In[156]:


# ROC plot
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=2)


# the warning tells us that model is too much biased but since the accuracy is good, 
# that means training data itself is biased. there are 4000 records with no as the 
# response and 500 records with yes as response.

# In[157]:


print('no count: ', bank[bank['response']=='no'].count()[0])
print('yes count: ', bank[bank['response']=='yes'].count()[0])


# Since ROC cant be compared as the training data is heavily biased for one class,
# training accuracy can be a good criteria to evaluate the model. Training accuracy 
# on logistic regression is 88% and 87% on gaussian naive bayes. hence logistic
# regression is just 1% better than naive bayes. Also logistic regression performs 
# much better on test data, but we cant use test data for comparison. training 
# accuracy is actually the mean of training + validation accuracy
