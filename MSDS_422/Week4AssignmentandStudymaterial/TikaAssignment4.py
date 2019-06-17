# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:15:01 2019

@author: vtika
"""

model = pd.DataFrame(model_data)

model.columns = ["mv","crim","zn","indus","chas","nox","rooms","age","dis","rad",
"tax","ptratio","lstat"]


corr_matrix = model.corr()
corr_matrix

import seaborn as sns
corr = corr_matrix
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)