#!/usr/bin/env python
# coding: utf-8

# In[46]:


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


# In[2]:


test


# In[3]:


dummy_cols = ['funder', 'installer', 'basin', 'public_meeting', 'scheme_management', 'permit',
              'construction_year', 'management_group', 'payment', 'water_quality',
              'quantity', 'source_class', 'waterpoint_type']

train = pd.get_dummies(train, columns = dummy_cols)

train = train.sample(frac=1).reset_index(drop=True)


# In[4]:


test = pd.get_dummies(test, columns = dummy_cols)
#test


# In[5]:


target = train.status_group
features = train.drop('status_group', axis=1)

X_train, X_val, y_train, y_val = train_test_split(features, target, train_size=0.8)


# In[6]:


target


# In[7]:


rfc = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=20)
rfc


# In[8]:


rfc_model = rfc.fit(X_train,y_train)


# In[9]:


val_acc = rfc_model.score(X_val,y_val)
val_acc


# In[10]:


pred = rfc_model.predict(X_train)
pred


# In[22]:


ytrain=y_train.values
ytrain


# In[23]:


from sklearn import metrics


# In[24]:


metrics.confusion_matrix(ytrain,pred)


# In[25]:


scores = rfc_model.score(X_train,y_train)
scores


# In[26]:


metrics.accuracy_score(ytrain,pred)


# In[27]:


from sklearn.metrics import classification_report


# In[28]:


from sklearn.metrics import classification_report
#target_names = ['functional', 'functional needs repair', 'non functional']
print(classification_report(ytrain, pred))


# In[29]:


from sklearn.externals import joblib


# In[30]:


filename = 'rfc78.sav'
joblib.dump(rfc_model,filename)


# In[31]:


loaded_model_rfc = joblib.load(filename)


# In[32]:


loaded_model_rfc


# In[33]:


final_rfc=loaded_model_rfc.fit(features, target)     


# In[34]:


predictions = loaded_model_rfc.predict(test)


# In[35]:


final_filename = 'final_rfc78.sav'


# In[36]:


joblib.dump(final_rfc,final_filename)


# In[37]:


final_loaded = joblib.load(final_filename)
final_loaded


# In[38]:


predictions


# In[39]:


importances = final_loaded.feature_importances_
importances


# In[40]:


indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_val.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print(X_val.columns[indices[f]],end=',')
    print()


# In[41]:


predictions=pd.DataFrame(predictions)
predictions


# In[42]:


#https://raw.githubusercontent.com/12ani/Taarifa_water_table/master/test_set_values.csv
pred_id = pd.read_csv('https://raw.githubusercontent.com/12ani/Taarifa_water_table/master/test_set_values.csv', usecols=['id'])
pred_id


# In[43]:


final=pd.concat([pred_id,predictions],axis=1)
final


# In[44]:


final.columns=['id','status_group']
final


# In[45]:


final.to_csv('predictions_RFCTEST78.csv', index=False)


# In[ ]:




