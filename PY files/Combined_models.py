#!/usr/bin/env python
# coding: utf-8

# In[31]:


from sklearn.externals import joblib
from sklearn.metrics import *
import warnings
warnings.filterwarnings('ignore')


# In[48]:


XGB=joblib.load('xgb_modelmodel.sav')


# In[54]:


GBC=joblib.load('GBCFINALMODELTEST.sav')


# In[34]:


RFC=joblib.load('final_rfc78.sav')


# In[35]:


import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
train = "https://raw.githubusercontent.com/12ani/Taarifa_water_table/master/pump_train_for_models%20(1).csv" 
test =  "https://raw.githubusercontent.com/12ani/Taarifa_water_table/master/water_table_test.csv"

train = pd.read_csv(train)
test = pd.read_csv(test)


# In[36]:


dummy_cols = ['funder', 'installer', 'basin', 'public_meeting', 'scheme_management', 'permit',
              'construction_year', 'management_group', 'payment', 'water_quality',
              'quantity', 'source_class', 'waterpoint_type']

train = pd.get_dummies(train, columns = dummy_cols)

train = train.sample(frac=1).reset_index(drop=True)


# In[37]:


test = pd.get_dummies(test, columns = dummy_cols)


# In[38]:


target = train.status_group
features = train.drop('status_group', axis=1)

X_train, X_val, y_train, y_val = train_test_split(features, target, train_size=0.8)


# In[49]:


XGB_acc = XGB.score(X_val,y_val)
print("Accuracy obtained by XGB model is",XGB_acc)


# In[55]:


GBC_acc = GBC.score(X_val,y_val)
print("Accuracy obtained by GBC model is",GBC_acc)


# In[53]:


RFC_acc = RFC.score(X_val,y_val)
print("Accuracy obtained by RFC model is",RFC_acc)


# In[ ]:





# In[ ]:




