# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 13:34:03 2019

@author: vtika
"""

import os
import pandas as pd
import glob


####################### APRIL INVOICES ###################
path = r'C:\Users\vtika\Downloads\invoiceDetails1'                     # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f) for f in all_files)
invoice   = pd.concat(df_from_each_file, ignore_index=True)

invoicesum = invoice.groupby(['DocumentDate'])[["ExtendedPrice"]].sum()

print(invoicesum)


invoice['ProductDescription'].str.contains('l').value_counts()

count = invoice['ProductDescription'].value_counts()

lime = invoice['ProductDescription'].filter(like='lime')