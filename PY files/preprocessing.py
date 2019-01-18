# -*- coding: utf-8 -*-
"""PREPROCESSING.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18f-XQl5JnxjTUGS_ZFmokU6E1ApSqC6M
"""

!pip3 install -U seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data.
df = pd.read_csv("https://raw.githubusercontent.com/12ani/Taarifa_water_table/master/training_set_values.csv")
labels = pd.read_csv("https://raw.githubusercontent.com/12ani/Taarifa_water_table/master/training_set_labels.csv")

# Merge data and labels together in one dataframe.
df = pd.merge(df, labels, on='id')
del labels

# Data set description 

df.info()

# Check for null values 

df.apply(lambda x: sum(x.isnull()))

#visual representation to get better picture about the dataset 

#count of status_group (functional, non-functional, needs repair)
plt.figure()
sns.countplot(df.status_group)
plt.show()
df.groupby(['status_group']).size()

# Status_group with respect to region 
plt.figure(figsize=(20,20))
sns.countplot(data=df,x='region',hue='status_group')
df.region.value_counts()

#Stus_group with respect to area 
plt.figure()
sns.scatterplot(data=df,x='latitude', y='longitude',hue='status_group')
plt.ylim(25,50)
plt.show()

# water qaulity w.r.t status group 
plt.figure(figsize=(15,15))
sns.countplot(data=df,x='water_quality',hue='status_group')
df.water_quality.value_counts()

#Since, majority of pumps lie in 'soft' category. There is not much variance 
#in this column. 
#We might drop this column.

plt.figure(figsize=(15,15))
sns.countplot(data=df,x='district_code',hue='status_group')
df.district_code.value_counts()

plt.figure(figsize=(15,15))
sns.countplot(data=df,x='ward',hue='status_group')
plt.ylim(0,150)
df.ward.value_counts()

#There are some features that we can remve right away just by looking at the data.
#Apart from these, other features are evaluated later in the code

#Drop redundant or similar columns 
df = df.drop(columns = ['wpt_name','extraction_type','extraction_type_group','extraction_type_class','scheme_name','payment_type','quantity_group','source_type','waterpoint_type_group','quality_group','source','management'])

#Drop location features 
df= df.drop(columns = ['gps_height', 'longitude', 'latitude', 'region_code', 'district_code','region', 'lga', 'ward'])

#Drop columns with insignificant values
#both the columns have single value
df= df.drop(columns = ['recorded_by','num_private'])

#Keeping only funders above 500 count
def funder_top500(row):  
    '''Keep top 5 values and set the rest to 'other'''

    if row['funder']=='Government Of Tanzania':
        return 'gov'
    elif row['funder']=='Danida':
        return 'danida'
    elif row['funder']=='Hesawa':
        return 'hesawa'
    elif row['funder']=='Rwssp':
        return 'rwssp'
    elif row['funder']=='World Bank':
        return 'world_bank' 
    elif row['funder']=='Kkkt':
        return 'Kkkt'    
    elif row['funder']=='World Vision':
        return 'World Vision'
    elif row['funder']=='Unicef':
        return 'Unicef' 
    elif row['funder']=='Tasaf':
        return 'Tasaf' 
    elif row['funder']=='Dhv':
        return 'Dhv' 
    elif row['funder']=='Private Individual':
        return 'Private Individual' 
    elif row['funder']=='Dwsp':
        return 'Dwsp' 
    elif row['funder']=='Norad':
        return 'Norad' 
    elif row['funder']=='Germany Republi':
        return 'Germany Republi'
    elif row['funder']=='Ministry Of Water':
        return 'Ministry Of Water'
    elif row['funder']=='Tcrs':
        return 'Tcrs'
    elif row['funder']=='Water':
        return 'Water'
    else:
        return 'other'
    
df['funder'] = df.apply(lambda row: funder_top500(row), axis=1)

plt.figure(figsize=(20,20))
sns.countplot(data=df,x='funder',hue='status_group')
df.funder.value_counts()

#Addiing new column called 'status_group_int' to make pivot table in order to 
# see status of pumps between the different funders.

vals_to_replace = {'functional':2, 'functional needs repair':1,
                   'non functional':0}

df['status_group_int']  = df.status_group.replace(vals_to_replace)

piv_table = pd.pivot_table(df,index=['funder','status_group'],
                           values='status_group_int', aggfunc='count')
piv_table

# For the feature 'installer'


df.installer.value_counts()

# Create a function to reduce the amount of dummy columns needed whilst maintaining the 
# information contained in the column.

def installer_top500(row):
    '''Keep top 5 values and set the rest to 'other'''
    if row['installer']=='DWE':
        return 'dwe'
    elif row['installer']=='Government':
        return 'gov'
    elif row['installer']=='RWE':
        return 'rwe'
    elif row['installer']=='Commu':
        return 'commu'
    elif row['installer']=='DANIDA':
        return 'danida'
    elif row['installer']=='KKKT':
        return 'KKKT'
    elif row['installer']=='Hesawa' or row['installer']=='HESAWA':
        return 'Hesawa'
    elif row['installer']=='TCRS':
        return 'TCRS'
    elif row['installer']=='Central government':
        return 'Central government'
    elif row['installer']=='CES':
        return 'CES'
    elif row['installer']=='Community':
        return 'Community'
    elif row['installer']=='DANID':
        return 'DANID'
    elif row['installer']=='District Council':
        return 'District Council'
     
    else:
        return 'other'  

df['installer'] = df.apply(lambda row: installer_top500(row), axis=1)

plt.figure(figsize=(20,20))
sns.countplot(data=df,x='installer',hue='status_group')
df.installer.value_counts()

piv_table = pd.pivot_table(df,index=['installer','status_group'],
                           values='status_group_int', aggfunc='count')
piv_table

# The next feature to inspect is 'subvillage'.
print(df.subvillage.value_counts())
#We can see that there is not much variance in the subvillage column, hence we can drop it.

#To verify our assumption, we can compute the the unique values in this feature. 
print('Unique villages: ', len(df.subvillage.value_counts()))

#since, it has 19287 unique values, it won't carry much variance in the dataset

df = df.drop('subvillage', axis=1)

# Next feature to process public_meeting

df.public_meeting.value_counts()

#Let's keep this column since it has only two values. Also, subsituting the NA values with 'unknown'
df.public_meeting = df.public_meeting.fillna('Unknown')

# Next fearure is 'scheme_management'
df.scheme_management.value_counts()

# Lets keep values above 1000 and assign the rest as 'other' 

def scheme_top1000(row):
    if row['scheme_management']=='VWC':
        return 'vwc'
    elif row['scheme_management']=='WUG':
        return 'wug'
    elif row['scheme_management']=='Water authority':
        return 'wtr_auth'
    elif row['scheme_management']=='WUA':
        return 'wua'
    elif row['scheme_management']=='Water Board':
        return 'wtr_brd'
    elif row['scheme_management']=='Parastatal':
        return 'Parastatal'
    elif row['scheme_management']=='Private operator':
        return 'Private operator'
    elif row['scheme_management']=='Company':
        return 'Company'
    else:
        return 'other'

      
df['scheme_management'] = df.apply(lambda row: scheme_top1000(row), axis=1)

plt.figure()
sns.countplot(data=df,x='scheme_management',hue='status_group')
df.scheme_management.value_counts()

piv_table = pd.pivot_table(df, index=['scheme_management', 'status_group'],
                           values='status_group_int', aggfunc='count')
piv_table

#'permit' is the last column with null values 
df.permit.value_counts()

#Just like the feature 'public_meeting' it has only 2 values but with better count ratio
#We will keeo this column and replace the NA values with null

df.permit = df.permit.fillna('Unknown')

#checking for null values in the columns 
df.apply(lambda x: sum(x.isnull()))

# string values and modify or remove them as we see fit.

str_cols = df.select_dtypes(include = ['object'])
str_cols.apply(lambda x: len(x.unique()))

# 'Date recorded'

df.date_recorded.describe()

#Converting 'date_recorded' into Datetime format. We assume that the most recently recorded data might
#have higher probability of having functional pump. 
#We can obtain this by subtracting all dates with the lastest recorded date


df.date_recorded = pd.to_datetime(df.date_recorded)
df.date_recorded.describe()

# The most recent data is 2013-12-03. Subtract each date from this point to obtain a 
# 'days_since_recorded' column. This 

df.date_recorded = pd.datetime(2013, 12, 3) - pd.to_datetime(df.date_recorded)
df.columns = ['days_since_recorded' if x=='date_recorded' else x for x in df.columns]
df.days_since_recorded = df.days_since_recorded.astype('timedelta64[D]').astype(int)
df.days_since_recorded.describe()

df.basin.value_counts()

piv_table = pd.pivot_table(df, index=['basin', 'status_group'],
                           values=['status_group_int'], aggfunc='count')
piv_table

# Most of the basins value have high number of functional pumps, therefore it seems
# like a good feature to keep.

str_cols.apply(lambda x: len(x.unique()))

#construction_year feature only has the year if construction of the pump, we assume 
# the more old the construction year is the more likely the pump will be non-functional or will need repair
df.construction_year.value_counts()

# changing the values of construction year into 60s, 70s, 80s,.....10s to eliminate 
# scattered values like 1960, 1964 can fall under 60s

def construction_00(row):
    if row['construction_year'] >= 1960 and row['construction_year'] < 1970:
        return '60s'
    elif row['construction_year'] >= 1970 and row['construction_year'] < 1980:
        return '70s'
    elif row['construction_year'] >= 1980 and row['construction_year'] < 1990:
        return '80s'
    elif row['construction_year'] >= 1990 and row['construction_year'] < 2000:
        return '90s'
    elif row['construction_year'] >= 2000 and row['construction_year'] < 2010:
        return '00s'
    elif row['construction_year'] >= 2010:
        return '10s'
    else:
        return 'unknown'
    
df['construction_year'] = df.apply(lambda row: construction_00(row), axis=1)

plt.figure()
sns.countplot(data=df,x='construction_year',hue='status_group')
df.construction_year.value_counts()

sns.distplot(df.population, bins = 40)
plt.show()

df.info()

sns.distplot(df.amount_tsh, bins = 40)
plt.show()

df.population.describe()

df.amount_tsh.describe()

# There is enough variation between the two, hence we can keep it in our model.
# Let's save the dataframe to a new csv file. We'll start creating models in the next notebooks.
df = df.drop('status_group_int', 1)
df.to_csv('water_table_train.csv', index=False)

# We'll also need to perform the same modifications to the test set.

test = pd.read_csv('https://raw.githubusercontent.com/12ani/Taarifa_water_table/master/test_set_values.csv')

test = test.drop(['wpt_name','extraction_type','extraction_type_group','extraction_type_class',
                  'scheme_name','payment_type','quantity_group','source_type','waterpoint_type_group',
                  'quality_group','source','management','gps_height', 'longitude', 'latitude', 
                  'region_code', 'district_code','region', 'lga', 'ward','recorded_by','num_private',
                 'subvillage'], axis=1)

test.date_recorded = pd.datetime(2013, 12, 3) - pd.to_datetime(test.date_recorded)
test.columns = ['days_since_recorded' if x=='date_recorded' else x for x in test.columns]
test.days_since_recorded = test.days_since_recorded.astype('timedelta64[D]').astype(int)

test.permit = test.permit.fillna('Unknown')
test.public_meeting = test.public_meeting.fillna('Unknown')

test['scheme_management'] = test.apply(lambda row: scheme_top1000(row), axis=1)
test['construction_year'] = test.apply(lambda row: construction_00(row), axis=1)
test['installer'] = test.apply(lambda row: installer_top500(row), axis=1)
test['funder'] = test.apply(lambda row: funder_top500(row), axis=1)

df.shape

test.shape

test.to_csv('water_table_test.csv', index=False)

from google.colab import files
files.download('water_table_test.csv','pump_train_for_models.csv')

