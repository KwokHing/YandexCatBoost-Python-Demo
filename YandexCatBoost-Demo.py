
# coding: utf-8

# ## CI6227 Assignment ##

# Installing the open source Yandex CatBoost package

# In[2]:


get_ipython().system(u'pip install catboost')


# Importing the required packaged: Numpy, Pandas, Matplotlib, Seaborn, Scikit-learn and CatBoost

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('ggplot') 
import seaborn as sns
from catboost import Pool, CatBoostClassifier, cv, CatboostIpythonWidget
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold


# Loading of [IBM HR Dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset/data) into pandas dataframe

# In[4]:


ibm_hr_df = pd.read_csv("/home/nbuser/library/IBM-HR-Employee-Attrition.csv")


# ### Part 1a: Data Exploration - Summary Statistics ###
# 
# Getting the summary statistics of the IBM HR dataset

# In[5]:


ibm_hr_df.describe()


# Zooming in on the summary statistics of irrelevant attributes __*EmployeeCount*__ and __*StandardHours*__

# In[6]:


irrList = ['EmployeeCount', 'StandardHours'] 
ibm_hr_df[irrList].describe()


# Zooming in on the summary statistics of irrelevant attribute __*Over18*__ 

# In[7]:


ibm_hr_df["Over18"].value_counts()


# From the summary statistics, one could see that attributes __*EmployeeCount*__, __*StandardHours*__ and __*Over18*__ holds only one single value for all of the 1470 records <br>
# 
# __*EmployeeCount*__ only holds a single value - 1.0 <br>
# __*StandardHours*__ only holds a single value - 80.0 <br>
# __*Over18*__        only holds a single value - 'Y'  <br>
# 
# These irrelevant attributes are duely dropped from the dataset

# ### Part 1b: Data Exploration - Missing Values and Duplicate Records ###
# 
# Checking for 'NA' and missing values in the dataset.

# In[8]:


ibm_hr_df.isnull().sum(axis=0)


# Well, we got lucky here, there isn't any missing values in this dataset
# 
# Next, let's check for the existence of duplicate records in the dataset

# In[9]:


ibm_hr_df.duplicated().sum()


# There are also no duplicate records in the dataset
# 
# Converting __*OverTime*__ binary categorical attribute to {1, 0}

# In[10]:


ibm_hr_df['OverTime'].replace(to_replace=dict(Yes=1, No=0), inplace=True)


# ### Part 2a: Data Preprocessing - Removal of Irrelevant Attributes ###

# In[12]:


ibm_hr_df = ibm_hr_df.drop(['EmployeeCount', 'StandardHours', 'Over18'], axis=1)


# ### Part 2b: Data Preprocessing - Feature Subset Selection - Low Variance Filter  ###
# 
# Performing variance analysis

# Performing Pearson correlation analysis between attributes to aid in dimension reduction

# In[15]:


plt.figure(figsize=(16,16))
sns.heatmap(ibm_hr_df.corr(), annot=True, fmt=".2f")

plt.show()


# Performing variance analysis to aid in dimension reduction

# In[16]:


variance_x = ibm_hr_df.drop('Attrition', axis=1)
variance_one_hot = pd.get_dummies(variance_x)


# In[17]:


#Normalise the dataset. This is required for getting the variance threshold
scaler = MinMaxScaler()
scaler.fit(variance_one_hot)
MinMaxScaler(copy=True, feature_range=(0, 1))
scaled_variance_one_hot = scaler.transform(variance_one_hot)


# In[18]:


#Set the threshold values and run VarianceThreshold 
thres = .85* (1 - .85)
sel = VarianceThreshold(threshold=thres)
sel.fit(scaled_variance_one_hot)
variance = sel.variances_


# In[19]:


#Sorting of the score in acsending orders for plotting
indices = np.argsort(variance)[::-1]
feature_list = list(variance_one_hot)
sorted_feature_list = []
thres_list = []
for f in range(len(variance_one_hot.columns)):
    sorted_feature_list.append(feature_list[indices[f]])
    thres_list.append(thres)


# In[20]:


plt.figure(figsize=(14,6))
plt.title("Feature Variance: %f" %(thres), fontsize = 14)
plt.bar(range(len(variance_one_hot.columns)), variance[indices], color="c")
plt.xticks(range(len(variance_one_hot.columns)), sorted_feature_list, rotation = 90)
plt.xlim([-0.5, len(variance_one_hot.columns)])
plt.plot(range(len(variance_one_hot.columns)), thres_list, "k-", color="r")
plt.tight_layout()
plt.show()


# Performing Pearson correlation analysis between attributes to aid in dimension reduction

# ### Part 3 ###

# In[21]:


rAttrList = ['Department', 'OverTime', 'HourlyRate',
             'StockOptionLevel', 'DistanceFromHome',
             'YearsInCurrentRole', 'Age']


# In[22]:


#keep only the attribute list on rAttrList
label_hr_df = ibm_hr_df[rAttrList]


# In[23]:


#convert continous attribute DistanceFromHome to Catergorical
#: 1: near, 2: mid distance, 3: far
maxValues = label_hr_df['DistanceFromHome'].max()
minValues = label_hr_df['DistanceFromHome'].min()
intervals = (maxValues - minValues)/3
bins = [0, (minValues + intervals), (maxValues - intervals), maxValues]
groupName = [1, 2, 3]
label_hr_df['CatDistanceFromHome'] = pd.cut(label_hr_df['DistanceFromHome'], bins, labels = groupName)


# In[24]:


# convert col type from cat to int64
label_hr_df['CatDistanceFromHome'] = pd.to_numeric(label_hr_df['CatDistanceFromHome']) 
label_hr_df.drop(['DistanceFromHome'], axis = 1, inplace = True)


# In[25]:


#replace department into 0 & 1, 0: R&D, and 1: Non-R&D
label_hr_df['Department'].replace(['Research & Development', 'Human Resources', 'Sales'],
                                  [0, 1, 1], inplace = True)


# In[26]:


#normalise data
label_hr_df_norm = (label_hr_df - label_hr_df.min()) / (label_hr_df.max() - label_hr_df.min())


# In[27]:


#create a data frame for the function value and class labels
value_df = pd.DataFrame(columns = ['ClassValue'])


# In[28]:


#compute the class value
for row in range (0, ibm_hr_df.shape[0]):
    if label_hr_df_norm['Department'][row] == 0:
        value = 0.3 * label_hr_df_norm['HourlyRate'][row] - 0.2 * label_hr_df_norm['OverTime'][row] +             - 0.2 * label_hr_df_norm['CatDistanceFromHome'][row] + 0.15 * label_hr_df_norm['StockOptionLevel'][row] +             0.1 * label_hr_df_norm['Age'][row] - 0.05 * label_hr_df_norm['YearsInCurrentRole'][row]
    
    else:
        value = 0.2 * label_hr_df_norm['HourlyRate'][row] - 0.3 * label_hr_df_norm['OverTime'][row] +             - 0.15 * label_hr_df_norm['CatDistanceFromHome'][row] + 0.2 * label_hr_df_norm['StockOptionLevel'][row] +             0.05 * label_hr_df_norm['Age'][row] - 0.1 * label_hr_df_norm['YearsInCurrentRole'][row]
    value_df.loc[row] = value


# In[29]:


# top 500 highest class value is satisfied with their job
v1 = value_df.sort_values('ClassValue', ascending = False).reset_index(drop = True)        ['ClassValue'][499]
# next top 500 is neutral
v2 = value_df.sort_values('ClassValue', ascending = False).reset_index(drop = True)        ['ClassValue'][999]
# rest is unsatisfied


# In[30]:


label_df = pd.DataFrame(columns = ['ClassLabel'])


# In[31]:


#compute the classlabel
for row in range (0, value_df.shape[0]):
    if value_df['ClassValue'][row] >= v1:
        cat = "Satisfied"
    elif value_df['ClassValue'][row] >= v2:
        cat = "Neutral"
    else:
        cat = "Unsatisfied"
    label_df.loc[row] = cat


# In[32]:


df = pd.concat([ibm_hr_df, label_df], axis = 1)


# ### Part 3: Classification with CatBoost ###

# In[26]:


#df = pd.read_csv("/home/nbuser/library/HR_dataset_generated_label.csv")


# In[33]:


df = df[['Age', 'Department', 'DistanceFromHome', 'HourlyRate', 'OverTime', 'StockOptionLevel', 
         'MaritalStatus', 'YearsInCurrentRole', 'EmployeeNumber', 'ClassLabel']]


# Split dataset into attributes/features __*X*__ and label/class __*y*__

# In[34]:


X = df.drop('ClassLabel', axis=1)
y = df.ClassLabel


# Replacing label/class value from __*'Satisfied'*__, __*'Neutral'*__ and *__'Unsatisfied'__* to *__2__*, __*1*__ and __*0*__

# In[35]:


y.replace(to_replace=dict(Satisfied=2, Neutral=1, Unsatisfied=0), inplace=True)


# Performing __'one hot encoding'__ method

# In[36]:


one_hot = pd.get_dummies(X)


# Normalisation of features

# In[37]:


one_hot = (one_hot - one_hot.mean()) / (one_hot.max() - one_hot.min())


# In[38]:


categorical_features_indices = np.where(one_hot.dtypes != np.float)[0]


# ### Part 3a: Model training with CatBoost ###
# Now lets split our data to train (70%) and test (30%) set:

# In[39]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(one_hot, y, train_size=0.7, random_state=1234)


# In[44]:


model = CatBoostClassifier(
    custom_loss = ['Accuracy'],
    random_seed = 100,
    loss_function = 'MultiClass'
)


# In[51]:


model.fit(
    X_train, y_train,
    cat_features = categorical_features_indices,
    verbose = True,  # you can uncomment this for text output
    #plot = True
)


# In[48]:


feature_score = pd.DataFrame(list(zip(one_hot.dtypes.index, model.get_feature_importance(Pool(one_hot, label=y, cat_features=categorical_features_indices)))),
                columns=['Feature','Score'])
feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')


# In[49]:


plt.rcParams["figure.figsize"] = (12,7)
ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
ax.set_xlabel('')

rects = ax.patches

# get feature score as labels round to 2 decimal
labels = feature_score['Score'].round(2)

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')

plt.show()


# In[50]:


model.score(X_test, y_test)


# ### Part 4: CatBoost Classifier Tuning ###

# In[40]:


model = CatBoostClassifier(
    l2_leaf_reg = 3,
    iterations = 1000,
    fold_len_multiplier = 1.05,
    learning_rate = 0.05,
    custom_loss = ['Accuracy'],
    random_seed = 100,
    loss_function = 'MultiClass'
)


# In[41]:


model.fit(
    X_train, y_train,
    cat_features = categorical_features_indices,
    verbose = True,  # you can uncomment this for text output
    #plot = True
)


# In[42]:


feature_score = pd.DataFrame(list(zip(one_hot.dtypes.index, model.get_feature_importance(Pool(one_hot, label=y, cat_features=categorical_features_indices)))),
                columns=['Feature','Score'])


# In[43]:


feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')


# In[44]:


plt.rcParams["figure.figsize"] = (12,7)
ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
ax.set_xlabel('')

rects = ax.patches

# get feature score as labels round to 2 decimal
labels = feature_score['Score'].round(2)

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')

plt.show()
#plt.savefig("image.png")


# In[61]:


cm = pd.DataFrame()
cm['Satisfaction'] = y_test
cm['Predict'] = model.predict(X_test)


# In[63]:


mappingSatisfaction = {0:'Unsatisfied', 1: 'Neutral', 2: 'Satisfied'}
mappingPredict = {0.0:'Unsatisfied', 1.0: 'Neutral', 2.0: 'Satisfied'}
cm = cm.replace({'Satisfaction': mappingSatisfaction, 'Predict': mappingPredict})


# In[64]:


pd.crosstab(cm['Satisfaction'], cm['Predict'], margins=True)


# In[65]:


model.score(X_test, y_test)

