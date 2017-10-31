
## C6227 Assignment ##

Installing the open source Yandex CatBoost package


```python
!pip install catboost
```

    Collecting catboost
      Downloading catboost-0.2.5-cp27-none-manylinux1_x86_64.whl (2.8MB)
    [K    100% |################################| 2.8MB 467kB/s eta 0:00:01
    [?25hRequirement already satisfied: six in /home/nbcommon/anaconda2_410/lib/python2.7/site-packages (from catboost)
    Requirement already satisfied: numpy in /home/nbcommon/anaconda2_410/lib/python2.7/site-packages (from catboost)
    Installing collected packages: catboost
    Successfully installed catboost-0.2.5


Importing the required packaged: Numpy, Pandas, Matplotlib, Seaborn, Scikit-learn and CatBoost


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('ggplot') 
import seaborn as sns
from catboost import Pool, CatBoostClassifier, cv, CatboostIpythonWidget
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
```

Loading of [IBM HR Dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset/data) into pandas dataframe


```python
ibm_hr_df = pd.read_csv("/home/nbuser/library/IBM-HR-Employee-Attrition.csv")
```

### Part 1: Feature Selection ###

Getting dataset summary statistics


```python
ibm_hr_df.describe()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.0</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>...</td>
      <td>1470.000000</td>
      <td>1470.0</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36.923810</td>
      <td>802.485714</td>
      <td>9.192517</td>
      <td>2.912925</td>
      <td>1.0</td>
      <td>1024.865306</td>
      <td>2.721769</td>
      <td>65.891156</td>
      <td>2.729932</td>
      <td>2.063946</td>
      <td>...</td>
      <td>2.712245</td>
      <td>80.0</td>
      <td>0.793878</td>
      <td>11.279592</td>
      <td>2.799320</td>
      <td>2.761224</td>
      <td>7.008163</td>
      <td>4.229252</td>
      <td>2.187755</td>
      <td>4.123129</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.135373</td>
      <td>403.509100</td>
      <td>8.106864</td>
      <td>1.024165</td>
      <td>0.0</td>
      <td>602.024335</td>
      <td>1.093082</td>
      <td>20.329428</td>
      <td>0.711561</td>
      <td>1.106940</td>
      <td>...</td>
      <td>1.081209</td>
      <td>0.0</td>
      <td>0.852077</td>
      <td>7.780782</td>
      <td>1.289271</td>
      <td>0.706476</td>
      <td>6.126525</td>
      <td>3.623137</td>
      <td>3.222430</td>
      <td>3.568136</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>102.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.000000</td>
      <td>465.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.0</td>
      <td>491.250000</td>
      <td>2.000000</td>
      <td>48.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36.000000</td>
      <td>802.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>1.0</td>
      <td>1020.500000</td>
      <td>3.000000</td>
      <td>66.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>3.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.000000</td>
      <td>1157.000000</td>
      <td>14.000000</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>1555.750000</td>
      <td>4.000000</td>
      <td>83.750000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>15.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>60.000000</td>
      <td>1499.000000</td>
      <td>29.000000</td>
      <td>5.000000</td>
      <td>1.0</td>
      <td>2068.000000</td>
      <td>4.000000</td>
      <td>100.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>3.000000</td>
      <td>40.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>40.000000</td>
      <td>18.000000</td>
      <td>15.000000</td>
      <td>17.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>




```python
irrList = ['EmployeeCount', 'StandardHours'] 
ibm_hr_df[irrList].describe()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmployeeCount</th>
      <th>StandardHours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1470.0</td>
      <td>1470.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.0</td>
      <td>80.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
ibm_hr_df["Over18"].value_counts()
```




    Y    1470
    Name: Over18, dtype: int64



From the summary statistics, one could see that attribute __*EmployeeCount*__, __*StandardHours*__ and __*Over18*__ holds only one single value for all of the 1470 records <br>

We will proceed to drop these attributes as they are irrelevant features


```python
ibm_hr_df = ibm_hr_df.drop(['EmployeeCount', 'StandardHours', 'Over18'], axis=1)
```

Checking for 'NA' and missing values in the dataset.


```python
ibm_hr_df.isnull().sum(axis=0)
```




    Age                         0
    Attrition                   0
    BusinessTravel              0
    DailyRate                   0
    Department                  0
    DistanceFromHome            0
    Education                   0
    EducationField              0
    EmployeeNumber              0
    EnvironmentSatisfaction     0
    Gender                      0
    HourlyRate                  0
    JobInvolvement              0
    JobLevel                    0
    JobRole                     0
    JobSatisfaction             0
    MaritalStatus               0
    MonthlyIncome               0
    MonthlyRate                 0
    NumCompaniesWorked          0
    OverTime                    0
    PercentSalaryHike           0
    PerformanceRating           0
    RelationshipSatisfaction    0
    StockOptionLevel            0
    TotalWorkingYears           0
    TrainingTimesLastYear       0
    WorkLifeBalance             0
    YearsAtCompany              0
    YearsInCurrentRole          0
    YearsSinceLastPromotion     0
    YearsWithCurrManager        0
    dtype: int64



Well, we got lucky here, there isn't any missing values in this dataset.

Next, let's check for the existence of duplicate records


```python
ibm_hr_df.duplicated().sum()
```




    0



Converting __*OverTime*__ binary categorical attribute to {1, 0}


```python
ibm_hr_df['OverTime'].replace(to_replace=dict(Yes=1, No=0), inplace=True)
```

Performing Pearson correlation analysis between attributes to aid in dimension reduction


```python
plt.figure(figsize=(16,16))
sns.heatmap(ibm_hr_df.corr(), annot=True, fmt=".2f")

plt.show()
```


![png](output_20_0.png)


Performing variance analysis to aid in dimension reduction


```python
variance_x = ibm_hr_df.drop('Attrition', axis=1)
variance_one_hot = pd.get_dummies(variance_x)
```


```python
#Normalise the dataset. This is required for getting the variance threshold
scaler = MinMaxScaler()
scaler.fit(variance_one_hot)
MinMaxScaler(copy=True, feature_range=(0, 1))
scaled_variance_one_hot = scaler.transform(variance_one_hot)
```


```python
#Set the threshold values and run VarianceThreshold 
thres = .85* (1 - .85)
sel = VarianceThreshold(threshold=thres)
sel.fit(scaled_variance_one_hot)
variance = sel.variances_
```


```python
#Sorting of the score in acsending orders for plotting
indices = np.argsort(variance)[::-1]
feature_list = list(variance_one_hot)
sorted_feature_list = []
thres_list = []
for f in range(len(variance_one_hot.columns)):
    sorted_feature_list.append(feature_list[indices[f]])
    thres_list.append(thres)
```


```python
plt.figure(figsize=(14,6))
plt.title("Feature Variance: %f" %(thres), fontsize = 14)
plt.bar(range(len(variance_one_hot.columns)), variance[indices], color="c")
plt.xticks(range(len(variance_one_hot.columns)), sorted_feature_list, rotation = 90)
plt.xlim([-0.5, len(variance_one_hot.columns)])
plt.plot(range(len(variance_one_hot.columns)), thres_list, "k-", color="r")
plt.tight_layout()
plt.show()
```


![png](output_26_0.png)


### Part 2: Labels Generation ###


```python
rAttrList = ['Department', 'OverTime', 'HourlyRate',
             'StockOptionLevel', 'DistanceFromHome',
             'YearsInCurrentRole', 'Age']
```


```python
#keep only the attribute list on rAttrList
label_hr_df = ibm_hr_df[rAttrList]
```


```python
#convert continous attribute DistanceFromHome to Catergorical
#: 1: near, 2: mid distance, 3: far
maxValues = label_hr_df['DistanceFromHome'].max()
minValues = label_hr_df['DistanceFromHome'].min()
intervals = (maxValues - minValues)/3
bins = [0, (minValues + intervals), (maxValues - intervals), maxValues]
groupName = [1, 2, 3]
label_hr_df['CatDistanceFromHome'] = pd.cut(label_hr_df['DistanceFromHome'], bins, labels = groupName)
```

    /home/nbcommon/anaconda2_410/lib/python2.7/site-packages/ipykernel/__main__.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
# convert col type from cat to int64
label_hr_df['CatDistanceFromHome'] = pd.to_numeric(label_hr_df['CatDistanceFromHome']) 
label_hr_df.drop(['DistanceFromHome'], axis = 1, inplace = True)
```

    /home/nbcommon/anaconda2_410/lib/python2.7/site-packages/ipykernel/__main__.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from ipykernel import kernelapp as app
    /home/nbcommon/anaconda2_410/lib/python2.7/site-packages/ipykernel/__main__.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      app.launch_new_instance()



```python
#replace department into 0 & 1, 0: R&D, and 1: Non-R&D
label_hr_df['Department'].replace(['Research & Development', 'Human Resources', 'Sales'],
                                  [0, 1, 1], inplace = True)
```

    /home/nbuser/anaconda2_410/lib/python2.7/site-packages/pandas/core/generic.py:3554: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self._update_inplace(new_data)



```python
#normalise data
label_hr_df_norm = (label_hr_df - label_hr_df.min()) / (label_hr_df.max() - label_hr_df.min())
```


```python
#create a data frame for the function value and class labels
value_df = pd.DataFrame(columns = ['ClassValue'])
```


```python
#compute the class value
for row in range (0, ibm_hr_df.shape[0]):
    if label_hr_df_norm['Department'][row] == 0:
        value = 0.3 * label_hr_df_norm['HourlyRate'][row] - 0.2 * label_hr_df_norm['OverTime'][row] + \
            - 0.2 * label_hr_df_norm['CatDistanceFromHome'][row] + 0.15 * label_hr_df_norm['StockOptionLevel'][row] + \
            0.1 * label_hr_df_norm['Age'][row] - 0.05 * label_hr_df_norm['YearsInCurrentRole'][row]
    
    else:
        value = 0.2 * label_hr_df_norm['HourlyRate'][row] - 0.3 * label_hr_df_norm['OverTime'][row] + \
            - 0.15 * label_hr_df_norm['CatDistanceFromHome'][row] + 0.2 * label_hr_df_norm['StockOptionLevel'][row] + \
            0.05 * label_hr_df_norm['Age'][row] - 0.1 * label_hr_df_norm['YearsInCurrentRole'][row]
    value_df.loc[row] = value
```


```python
# top 500 highest class value is satisfied with their job
v1 = value_df.sort_values('ClassValue', ascending = False).reset_index(drop = True)\
        ['ClassValue'][499]
# next top 500 is neutral
v2 = value_df.sort_values('ClassValue', ascending = False).reset_index(drop = True)\
        ['ClassValue'][999]
# rest is unsatisfied
```


```python
label_df = pd.DataFrame(columns = ['ClassLabel'])
```


```python
#compute the classlabel
for row in range (0, value_df.shape[0]):
    if value_df['ClassValue'][row] >= v1:
        cat = "Satisfied"
    elif value_df['ClassValue'][row] >= v2:
        cat = "Neutral"
    else:
        cat = "Unsatisfied"
    label_df.loc[row] = cat
```


```python
df = pd.concat([ibm_hr_df, label_df], axis = 1)
```

### Part 3: Classification with CatBoost ###


```python
#df = pd.read_csv("/home/nbuser/library/HR_dataset_generated_label.csv")
```


```python
df = df[['Age', 'Department', 'DistanceFromHome', 'HourlyRate', 'OverTime', 'StockOptionLevel', 
         'MaritalStatus', 'YearsInCurrentRole', 'EmployeeNumber', 'ClassLabel']]
```

Split dataset into attributes/features __*X*__ and label/class __*y*__


```python
X = df.drop('ClassLabel', axis=1)
y = df.ClassLabel
```

Replacing label/class value from __*'Satisfied'*__, __*'Neutral'*__ and *__'Unsatisfied'__* to *__2__*, __*1*__ and __*0*__


```python
y.replace(to_replace=dict(Satisfied=2, Neutral=1, Unsatisfied=0), inplace=True)
```

Performing __'one hot encoding'__ method


```python
one_hot = pd.get_dummies(X)
```

Normalisation of features


```python
one_hot = (one_hot - one_hot.mean()) / (one_hot.max() - one_hot.min())
```


```python
categorical_features_indices = np.where(one_hot.dtypes != np.float)[0]
```

### Part 3a: Model training with CatBoost ###
Now lets split our data to train (70%) and test (30%) set:


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(one_hot, y, train_size=0.7, random_state=1234)
```


```python
model = CatBoostClassifier(
    custom_loss = ['Accuracy'],
    random_seed = 100,
    loss_function = 'MultiClass'
)
```


```python
model.fit(
    X_train, y_train,
    cat_features = categorical_features_indices,
    verbose = True,  # you can uncomment this for text output
    #plot = True
)
```




    '\nmodel.fit(\n    X_train, y_train,\n    cat_features = categorical_features_indices,\n    verbose = True,  # you can uncomment this for text output\n    #plot = True\n)\n'




```python
feature_score = pd.DataFrame(list(zip(one_hot.dtypes.index, model.get_feature_importance(Pool(one_hot, label=y, cat_features=categorical_features_indices)))),
                columns=['Feature','Score'])
feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
```


```python
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
```


![png](output_57_0.png)



```python
model.score(X_test, y_test)
```




    0.92517006802721091



### Part 4: CatBoost Classifier Tuning ###


```python
model = CatBoostClassifier(
    l2_leaf_reg = 3,
    iterations = 1000,
    fold_len_multiplier = 1.05,
    learning_rate = 0.05,
    custom_loss = ['Accuracy'],
    random_seed = 100,
    loss_function = 'MultiClass'
)
```


```python
model.fit(
    X_train, y_train,
    cat_features = categorical_features_indices,
    verbose = True,  # you can uncomment this for text output
    #plot = True
)
```

    Borders for float features generated
    0:	learn -1.064919059	total: 88.1ms	remaining: 1m 28s
    1:	learn -1.02506179	total: 136ms	remaining: 1m 7s
    2:	learn -0.987600438	total: 179ms	remaining: 59.6s
    3:	learn -0.9521302748	total: 229ms	remaining: 56.9s
    4:	learn -0.9212409658	total: 265ms	remaining: 52.8s
    5:	learn -0.8914720843	total: 312ms	remaining: 51.8s
    6:	learn -0.8608445021	total: 361ms	remaining: 51.1s
    7:	learn -0.8334836706	total: 401ms	remaining: 49.7s
    8:	learn -0.8113218202	total: 442ms	remaining: 48.7s
    9:	learn -0.7887240798	total: 483ms	remaining: 47.8s
    10:	learn -0.7654734708	total: 525ms	remaining: 47.2s
    11:	learn -0.7441723847	total: 563ms	remaining: 46.3s
    12:	learn -0.7273213481	total: 616ms	remaining: 46.7s
    13:	learn -0.7117720226	total: 633ms	remaining: 44.6s
    14:	learn -0.6943642033	total: 675ms	remaining: 44.3s
    15:	learn -0.6781659875	total: 709ms	remaining: 43.6s
    16:	learn -0.6638900187	total: 747ms	remaining: 43.2s
    17:	learn -0.6465621556	total: 790ms	remaining: 43.1s
    18:	learn -0.6321577853	total: 826ms	remaining: 42.6s
    19:	learn -0.6202549704	total: 864ms	remaining: 42.3s
    20:	learn -0.6077138272	total: 907ms	remaining: 42.3s
    21:	learn -0.5958833071	total: 939ms	remaining: 41.7s
    22:	learn -0.5852145971	total: 982ms	remaining: 41.7s
    23:	learn -0.5756196088	total: 1.02s	remaining: 41.4s
    24:	learn -0.5631845468	total: 1.06s	remaining: 41.3s
    25:	learn -0.5540928464	total: 1.1s	remaining: 41.4s
    26:	learn -0.5457839628	total: 1.13s	remaining: 40.8s
    27:	learn -0.5347697765	total: 1.17s	remaining: 40.7s
    28:	learn -0.5267175066	total: 1.21s	remaining: 40.6s
    29:	learn -0.5185703811	total: 1.25s	remaining: 40.4s
    30:	learn -0.510959972	total: 1.29s	remaining: 40.3s
    31:	learn -0.5056846241	total: 1.31s	remaining: 39.6s
    32:	learn -0.498472873	total: 1.35s	remaining: 39.6s
    33:	learn -0.4910463635	total: 1.39s	remaining: 39.6s
    34:	learn -0.4826664548	total: 1.43s	remaining: 39.5s
    35:	learn -0.4780480385	total: 1.46s	remaining: 39.1s
    36:	learn -0.474712874	total: 1.48s	remaining: 38.4s
    37:	learn -0.4694496884	total: 1.5s	remaining: 38s
    38:	learn -0.4627494117	total: 1.54s	remaining: 38s
    39:	learn -0.4569057348	total: 1.58s	remaining: 37.9s
    40:	learn -0.4499596618	total: 1.62s	remaining: 37.9s
    41:	learn -0.4446105537	total: 1.66s	remaining: 37.8s
    42:	learn -0.4404631218	total: 1.68s	remaining: 37.4s
    43:	learn -0.4353369328	total: 1.71s	remaining: 37.2s
    44:	learn -0.4290587595	total: 1.75s	remaining: 37.2s
    45:	learn -0.4238886513	total: 1.79s	remaining: 37.1s
    46:	learn -0.4190830301	total: 1.83s	remaining: 37.2s
    47:	learn -0.4139584868	total: 1.88s	remaining: 37.2s
    48:	learn -0.4090822897	total: 1.92s	remaining: 37.2s
    49:	learn -0.4037027637	total: 1.96s	remaining: 37.2s
    50:	learn -0.399022889	total: 1.99s	remaining: 37.1s
    51:	learn -0.3935642171	total: 2.03s	remaining: 37s
    52:	learn -0.3891947383	total: 2.07s	remaining: 37.1s
    53:	learn -0.3857445723	total: 2.11s	remaining: 37s
    54:	learn -0.3813998819	total: 2.15s	remaining: 36.9s
    55:	learn -0.3772274215	total: 2.19s	remaining: 36.9s
    56:	learn -0.3750588988	total: 2.21s	remaining: 36.6s
    57:	learn -0.3710840794	total: 2.25s	remaining: 36.5s
    58:	learn -0.3677455369	total: 2.29s	remaining: 36.5s
    59:	learn -0.3648528617	total: 2.32s	remaining: 36.4s
    60:	learn -0.3613582634	total: 2.37s	remaining: 36.4s
    61:	learn -0.3569212788	total: 2.41s	remaining: 36.5s
    62:	learn -0.3529146286	total: 2.45s	remaining: 36.5s
    63:	learn -0.3506190727	total: 2.5s	remaining: 36.5s
    64:	learn -0.346397955	total: 2.54s	remaining: 36.6s
    65:	learn -0.3434285199	total: 2.58s	remaining: 36.5s
    66:	learn -0.3393180162	total: 2.63s	remaining: 36.6s
    67:	learn -0.3374250148	total: 2.67s	remaining: 36.5s
    68:	learn -0.3337058074	total: 2.71s	remaining: 36.6s
    69:	learn -0.3308358264	total: 2.75s	remaining: 36.6s
    70:	learn -0.328603359	total: 2.8s	remaining: 36.6s
    71:	learn -0.325508245	total: 2.85s	remaining: 36.7s
    72:	learn -0.3219641352	total: 2.89s	remaining: 36.7s
    73:	learn -0.3193129471	total: 2.93s	remaining: 36.7s
    74:	learn -0.3172465951	total: 2.97s	remaining: 36.6s
    75:	learn -0.314080606	total: 3.01s	remaining: 36.6s
    76:	learn -0.3118133574	total: 3.05s	remaining: 36.6s
    77:	learn -0.3093414059	total: 3.1s	remaining: 36.6s
    78:	learn -0.3068826059	total: 3.14s	remaining: 36.7s
    79:	learn -0.3048011291	total: 3.19s	remaining: 36.7s
    80:	learn -0.3029028387	total: 3.23s	remaining: 36.6s
    81:	learn -0.3009953633	total: 3.26s	remaining: 36.5s
    82:	learn -0.2993538854	total: 3.29s	remaining: 36.4s
    83:	learn -0.2971211239	total: 3.34s	remaining: 36.4s
    84:	learn -0.2958156529	total: 3.36s	remaining: 36.2s
    85:	learn -0.2940580795	total: 3.4s	remaining: 36.1s
    86:	learn -0.292025725	total: 3.44s	remaining: 36.1s
    87:	learn -0.2893513265	total: 3.47s	remaining: 36s
    88:	learn -0.2874445631	total: 3.51s	remaining: 35.9s
    89:	learn -0.285252546	total: 3.55s	remaining: 35.9s
    90:	learn -0.2836729837	total: 3.58s	remaining: 35.8s
    91:	learn -0.2818612245	total: 3.62s	remaining: 35.7s
    92:	learn -0.2797052397	total: 3.9s	remaining: 38.1s
    93:	learn -0.2781346026	total: 3.98s	remaining: 38.4s
    94:	learn -0.2762940522	total: 4.16s	remaining: 39.7s
    95:	learn -0.2748735255	total: 4.19s	remaining: 39.5s
    96:	learn -0.2732801779	total: 4.23s	remaining: 39.4s
    97:	learn -0.2716223742	total: 4.41s	remaining: 40.6s
    98:	learn -0.2704744319	total: 4.45s	remaining: 40.5s
    99:	learn -0.2695670942	total: 4.5s	remaining: 40.5s
    100:	learn -0.2680545144	total: 4.53s	remaining: 40.3s
    101:	learn -0.2671006284	total: 4.58s	remaining: 40.3s
    102:	learn -0.2651207394	total: 4.61s	remaining: 40.2s
    103:	learn -0.2636927635	total: 4.65s	remaining: 40.1s
    104:	learn -0.2618787866	total: 4.69s	remaining: 40s
    105:	learn -0.2601758783	total: 4.74s	remaining: 39.9s
    106:	learn -0.2589556286	total: 4.76s	remaining: 39.7s
    107:	learn -0.2571597063	total: 4.8s	remaining: 39.6s
    108:	learn -0.2555791379	total: 4.84s	remaining: 39.6s
    109:	learn -0.2551986449	total: 4.86s	remaining: 39.3s
    110:	learn -0.2541919008	total: 4.89s	remaining: 39.2s
    111:	learn -0.2535816531	total: 4.91s	remaining: 38.9s
    112:	learn -0.2521656405	total: 4.94s	remaining: 38.8s
    113:	learn -0.250069953	total: 4.98s	remaining: 38.7s
    114:	learn -0.2487478807	total: 5.01s	remaining: 38.6s
    115:	learn -0.2479207829	total: 5.05s	remaining: 38.5s
    116:	learn -0.2460463995	total: 5.11s	remaining: 38.6s
    117:	learn -0.2441098113	total: 5.15s	remaining: 38.5s
    118:	learn -0.2424386564	total: 5.19s	remaining: 38.4s
    119:	learn -0.2411296081	total: 5.24s	remaining: 38.4s
    120:	learn -0.2397065984	total: 5.28s	remaining: 38.3s
    121:	learn -0.2380772449	total: 5.33s	remaining: 38.3s
    122:	learn -0.2364429243	total: 5.37s	remaining: 38.3s
    123:	learn -0.2350014422	total: 5.42s	remaining: 38.3s
    124:	learn -0.2338238592	total: 5.47s	remaining: 38.3s
    125:	learn -0.2323065653	total: 5.52s	remaining: 38.3s
    126:	learn -0.2307859996	total: 5.56s	remaining: 38.2s
    127:	learn -0.2293584574	total: 5.61s	remaining: 38.2s
    128:	learn -0.2276276687	total: 5.65s	remaining: 38.1s
    129:	learn -0.2265455006	total: 5.68s	remaining: 38.1s
    130:	learn -0.2252006002	total: 5.73s	remaining: 38s
    131:	learn -0.2235186214	total: 5.77s	remaining: 38s
    132:	learn -0.222260683	total: 5.81s	remaining: 37.9s
    133:	learn -0.2207232508	total: 5.85s	remaining: 37.8s
    134:	learn -0.2194646251	total: 5.89s	remaining: 37.7s
    135:	learn -0.2184196376	total: 5.92s	remaining: 37.6s
    136:	learn -0.2172121687	total: 5.96s	remaining: 37.6s
    137:	learn -0.2156172419	total: 6s	remaining: 37.5s
    138:	learn -0.2144237719	total: 6.03s	remaining: 37.4s
    139:	learn -0.2129187075	total: 6.07s	remaining: 37.3s
    140:	learn -0.2122661232	total: 6.11s	remaining: 37.2s
    141:	learn -0.2116047391	total: 6.13s	remaining: 37.1s
    142:	learn -0.2103724132	total: 6.17s	remaining: 37s
    143:	learn -0.2096114799	total: 6.21s	remaining: 36.9s
    144:	learn -0.2085972047	total: 6.24s	remaining: 36.8s
    145:	learn -0.2072735917	total: 6.28s	remaining: 36.8s
    146:	learn -0.2064480531	total: 6.32s	remaining: 36.7s
    147:	learn -0.2052949756	total: 6.36s	remaining: 36.6s
    148:	learn -0.2043907766	total: 6.4s	remaining: 36.6s
    149:	learn -0.2032722592	total: 6.45s	remaining: 36.6s
    150:	learn -0.2023054684	total: 6.49s	remaining: 36.5s
    151:	learn -0.2012657403	total: 6.53s	remaining: 36.4s
    152:	learn -0.199818604	total: 6.57s	remaining: 36.4s
    153:	learn -0.1990187017	total: 6.6s	remaining: 36.3s
    154:	learn -0.1981845147	total: 6.64s	remaining: 36.2s
    155:	learn -0.1974233261	total: 6.69s	remaining: 36.2s
    156:	learn -0.1961223881	total: 6.73s	remaining: 36.2s
    157:	learn -0.195301182	total: 6.78s	remaining: 36.1s
    158:	learn -0.1939784544	total: 6.82s	remaining: 36.1s
    159:	learn -0.1929118232	total: 6.87s	remaining: 36s
    160:	learn -0.192322512	total: 6.9s	remaining: 36s
    161:	learn -0.1912299223	total: 6.94s	remaining: 35.9s
    162:	learn -0.1900594826	total: 6.99s	remaining: 35.9s
    163:	learn -0.189036068	total: 7.03s	remaining: 35.8s
    164:	learn -0.1883333476	total: 7.07s	remaining: 35.8s
    165:	learn -0.1873241923	total: 7.11s	remaining: 35.7s
    166:	learn -0.1863098329	total: 7.16s	remaining: 35.7s
    167:	learn -0.1851933039	total: 7.2s	remaining: 35.6s
    168:	learn -0.1841890443	total: 7.24s	remaining: 35.6s
    169:	learn -0.1832924598	total: 7.29s	remaining: 35.6s
    170:	learn -0.1825378591	total: 7.33s	remaining: 35.5s
    171:	learn -0.1815643674	total: 7.37s	remaining: 35.5s
    172:	learn -0.1805714473	total: 7.42s	remaining: 35.5s
    173:	learn -0.1796025278	total: 7.46s	remaining: 35.4s
    174:	learn -0.1787351063	total: 7.5s	remaining: 35.4s
    175:	learn -0.1775916069	total: 7.55s	remaining: 35.4s
    176:	learn -0.1769596988	total: 7.59s	remaining: 35.3s
    177:	learn -0.1759436467	total: 7.64s	remaining: 35.3s
    178:	learn -0.1749403754	total: 7.67s	remaining: 35.2s
    179:	learn -0.1740011059	total: 7.71s	remaining: 35.1s
    180:	learn -0.1729617704	total: 7.75s	remaining: 35.1s
    181:	learn -0.1720770941	total: 7.79s	remaining: 35s
    182:	learn -0.1715138778	total: 7.84s	remaining: 35s
    183:	learn -0.1707912022	total: 7.87s	remaining: 34.9s
    184:	learn -0.170221366	total: 7.91s	remaining: 34.8s
    185:	learn -0.1696443088	total: 7.95s	remaining: 34.8s
    186:	learn -0.1689622171	total: 7.99s	remaining: 34.7s
    187:	learn -0.1682385353	total: 8.03s	remaining: 34.7s
    188:	learn -0.1675811664	total: 8.06s	remaining: 34.6s
    189:	learn -0.1670455526	total: 8.1s	remaining: 34.5s
    190:	learn -0.16632195	total: 8.14s	remaining: 34.5s
    191:	learn -0.1658193095	total: 8.18s	remaining: 34.4s
    192:	learn -0.1650683599	total: 8.22s	remaining: 34.4s
    193:	learn -0.1646099862	total: 8.26s	remaining: 34.3s
    194:	learn -0.1639893048	total: 8.31s	remaining: 34.3s
    195:	learn -0.1635597446	total: 8.36s	remaining: 34.3s
    196:	learn -0.1628309202	total: 8.41s	remaining: 34.3s
    197:	learn -0.1620345547	total: 8.45s	remaining: 34.2s
    198:	learn -0.1613660111	total: 8.49s	remaining: 34.2s
    199:	learn -0.1607837243	total: 8.53s	remaining: 34.1s
    200:	learn -0.1601555693	total: 8.56s	remaining: 34s
    201:	learn -0.159554856	total: 8.6s	remaining: 34s
    202:	learn -0.1587321522	total: 8.65s	remaining: 33.9s
    203:	learn -0.158346815	total: 8.68s	remaining: 33.9s
    204:	learn -0.1579261214	total: 8.72s	remaining: 33.8s
    205:	learn -0.1571832558	total: 8.76s	remaining: 33.8s
    206:	learn -0.156561676	total: 8.8s	remaining: 33.7s
    207:	learn -0.1556981079	total: 8.84s	remaining: 33.7s
    208:	learn -0.1551409429	total: 8.88s	remaining: 33.6s
    209:	learn -0.1542685097	total: 8.93s	remaining: 33.6s
    210:	learn -0.1536777326	total: 8.96s	remaining: 33.5s
    211:	learn -0.1528186463	total: 9.01s	remaining: 33.5s
    212:	learn -0.1523265466	total: 9.06s	remaining: 33.5s
    213:	learn -0.1517784803	total: 9.11s	remaining: 33.5s
    214:	learn -0.1511418982	total: 9.16s	remaining: 33.4s
    215:	learn -0.1505122545	total: 9.21s	remaining: 33.4s
    216:	learn -0.1501287486	total: 9.24s	remaining: 33.4s
    217:	learn -0.1493921055	total: 9.29s	remaining: 33.3s
    218:	learn -0.1488366189	total: 9.33s	remaining: 33.3s
    219:	learn -0.1483716629	total: 9.37s	remaining: 33.2s
    220:	learn -0.1480547555	total: 9.42s	remaining: 33.2s
    221:	learn -0.1474448884	total: 9.46s	remaining: 33.1s
    222:	learn -0.1465489534	total: 9.5s	remaining: 33.1s
    223:	learn -0.1458748266	total: 9.54s	remaining: 33s
    224:	learn -0.1451766609	total: 9.57s	remaining: 33s
    225:	learn -0.1444786567	total: 9.6s	remaining: 32.9s
    226:	learn -0.1440186643	total: 9.64s	remaining: 32.8s
    227:	learn -0.1435177415	total: 9.67s	remaining: 32.7s
    228:	learn -0.142994284	total: 9.7s	remaining: 32.7s
    229:	learn -0.14239633	total: 9.73s	remaining: 32.6s
    230:	learn -0.1418838787	total: 9.77s	remaining: 32.5s
    231:	learn -0.1413470653	total: 9.8s	remaining: 32.4s
    232:	learn -0.1410250504	total: 9.83s	remaining: 32.4s
    233:	learn -0.140290986	total: 9.87s	remaining: 32.3s
    234:	learn -0.1398324574	total: 9.91s	remaining: 32.3s
    235:	learn -0.1393551399	total: 9.95s	remaining: 32.2s
    236:	learn -0.1388484932	total: 9.98s	remaining: 32.1s
    237:	learn -0.1381177395	total: 10s	remaining: 32.1s
    238:	learn -0.137498535	total: 10.1s	remaining: 32s
    239:	learn -0.1372076013	total: 10.1s	remaining: 31.9s
    240:	learn -0.1368250697	total: 10.1s	remaining: 31.9s
    241:	learn -0.1364220747	total: 10.2s	remaining: 31.8s
    242:	learn -0.1357307614	total: 10.2s	remaining: 31.8s
    243:	learn -0.1353506489	total: 10.2s	remaining: 31.7s
    244:	learn -0.1348738795	total: 10.3s	remaining: 31.6s
    245:	learn -0.1343902838	total: 10.3s	remaining: 31.6s
    246:	learn -0.1338385505	total: 10.3s	remaining: 31.5s
    247:	learn -0.1333346292	total: 10.4s	remaining: 31.5s
    248:	learn -0.1327839746	total: 10.4s	remaining: 31.4s
    249:	learn -0.1323092432	total: 10.4s	remaining: 31.3s
    250:	learn -0.1319523706	total: 10.5s	remaining: 31.3s
    251:	learn -0.1315847898	total: 10.5s	remaining: 31.3s
    252:	learn -0.1311730863	total: 10.6s	remaining: 31.2s
    253:	learn -0.1307830555	total: 10.6s	remaining: 31.2s
    254:	learn -0.13038984	total: 10.7s	remaining: 31.1s
    255:	learn -0.1298973397	total: 10.7s	remaining: 31.1s
    256:	learn -0.1293672399	total: 10.7s	remaining: 31s
    257:	learn -0.1287118684	total: 10.8s	remaining: 31s
    258:	learn -0.1281442487	total: 10.8s	remaining: 31s
    259:	learn -0.127708775	total: 10.9s	remaining: 30.9s
    260:	learn -0.1273743377	total: 10.9s	remaining: 30.9s
    261:	learn -0.1270781848	total: 10.9s	remaining: 30.8s
    262:	learn -0.1265865329	total: 11s	remaining: 30.7s
    263:	learn -0.1262394203	total: 11s	remaining: 30.7s
    264:	learn -0.1256360761	total: 11s	remaining: 30.6s
    265:	learn -0.1253674264	total: 11.1s	remaining: 30.6s
    266:	learn -0.1250248516	total: 11.1s	remaining: 30.5s
    267:	learn -0.1246679333	total: 11.2s	remaining: 30.5s
    268:	learn -0.1242148198	total: 11.2s	remaining: 30.4s
    269:	learn -0.1238395235	total: 11.2s	remaining: 30.4s
    270:	learn -0.1235405634	total: 11.3s	remaining: 30.3s
    271:	learn -0.1230781336	total: 11.3s	remaining: 30.2s
    272:	learn -0.1228126465	total: 11.3s	remaining: 30.2s
    273:	learn -0.1225238352	total: 11.4s	remaining: 30.1s
    274:	learn -0.122064182	total: 11.4s	remaining: 30.1s
    275:	learn -0.1217306479	total: 11.5s	remaining: 30s
    276:	learn -0.1213002443	total: 11.5s	remaining: 30s
    277:	learn -0.1209914122	total: 11.5s	remaining: 29.9s
    278:	learn -0.1206161194	total: 11.6s	remaining: 29.9s
    279:	learn -0.1201918437	total: 11.6s	remaining: 29.8s
    280:	learn -0.1197712505	total: 11.6s	remaining: 29.8s
    281:	learn -0.119375131	total: 11.7s	remaining: 29.7s
    282:	learn -0.1190093505	total: 11.7s	remaining: 29.7s
    283:	learn -0.1186489028	total: 11.7s	remaining: 29.6s
    284:	learn -0.1183244038	total: 11.8s	remaining: 29.5s
    285:	learn -0.1178278633	total: 11.8s	remaining: 29.5s
    286:	learn -0.1174568216	total: 11.8s	remaining: 29.4s
    287:	learn -0.1172157735	total: 11.9s	remaining: 29.4s
    288:	learn -0.1168172468	total: 11.9s	remaining: 29.3s
    289:	learn -0.1165187796	total: 12s	remaining: 29.3s
    290:	learn -0.1162323716	total: 12s	remaining: 29.2s
    291:	learn -0.1159951648	total: 12s	remaining: 29.2s
    292:	learn -0.1155136214	total: 12.1s	remaining: 29.1s
    293:	learn -0.1151703045	total: 12.1s	remaining: 29.1s
    294:	learn -0.1148356055	total: 12.1s	remaining: 29s
    295:	learn -0.1144435874	total: 12.2s	remaining: 29s
    296:	learn -0.1140429024	total: 12.2s	remaining: 28.9s
    297:	learn -0.1136166939	total: 12.3s	remaining: 28.9s
    298:	learn -0.1130786224	total: 12.3s	remaining: 28.8s
    299:	learn -0.1125907956	total: 12.3s	remaining: 28.8s
    300:	learn -0.1123740904	total: 12.4s	remaining: 28.7s
    301:	learn -0.1119774616	total: 12.4s	remaining: 28.7s
    302:	learn -0.1115616474	total: 12.5s	remaining: 28.7s
    303:	learn -0.111273853	total: 12.5s	remaining: 28.6s
    304:	learn -0.1108387943	total: 12.5s	remaining: 28.6s
    305:	learn -0.1106088016	total: 12.6s	remaining: 28.5s
    306:	learn -0.1101382349	total: 12.6s	remaining: 28.4s
    307:	learn -0.109822005	total: 12.6s	remaining: 28.4s
    308:	learn -0.1094806185	total: 12.7s	remaining: 28.3s
    309:	learn -0.1090662369	total: 12.7s	remaining: 28.3s
    310:	learn -0.1087769668	total: 12.7s	remaining: 28.2s
    311:	learn -0.1085211927	total: 12.8s	remaining: 28.2s
    312:	learn -0.1081999593	total: 12.8s	remaining: 28.1s
    313:	learn -0.1078410783	total: 12.9s	remaining: 28.1s
    314:	learn -0.1075478269	total: 12.9s	remaining: 28.1s
    315:	learn -0.1070670709	total: 12.9s	remaining: 28s
    316:	learn -0.1068168936	total: 13s	remaining: 28s
    317:	learn -0.1063309216	total: 13s	remaining: 27.9s
    318:	learn -0.1059473894	total: 13.1s	remaining: 27.9s
    319:	learn -0.1057537934	total: 13.1s	remaining: 27.8s
    320:	learn -0.1053118474	total: 13.1s	remaining: 27.8s
    321:	learn -0.1048831964	total: 13.2s	remaining: 27.7s
    322:	learn -0.1044660297	total: 13.2s	remaining: 27.7s
    323:	learn -0.1042507772	total: 13.3s	remaining: 27.7s
    324:	learn -0.1039393046	total: 13.3s	remaining: 27.6s
    325:	learn -0.1036737873	total: 13.3s	remaining: 27.6s
    326:	learn -0.1032659809	total: 13.4s	remaining: 27.5s
    327:	learn -0.1030257514	total: 13.4s	remaining: 27.5s
    328:	learn -0.1026658425	total: 13.5s	remaining: 27.4s
    329:	learn -0.1023654225	total: 13.5s	remaining: 27.4s
    330:	learn -0.1021359056	total: 13.5s	remaining: 27.3s
    331:	learn -0.1018757241	total: 13.6s	remaining: 27.3s
    332:	learn -0.1015557871	total: 13.6s	remaining: 27.2s
    333:	learn -0.1012206614	total: 13.6s	remaining: 27.2s
    334:	learn -0.1008258815	total: 13.7s	remaining: 27.1s
    335:	learn -0.100550177	total: 13.7s	remaining: 27.1s
    336:	learn -0.1002887494	total: 13.7s	remaining: 27s
    337:	learn -0.1000267518	total: 13.8s	remaining: 27s
    338:	learn -0.0998605875	total: 13.8s	remaining: 27s
    339:	learn -0.09960962567	total: 13.9s	remaining: 26.9s
    340:	learn -0.09916075247	total: 13.9s	remaining: 26.9s
    341:	learn -0.0988493924	total: 13.9s	remaining: 26.8s
    342:	learn -0.09862457427	total: 14s	remaining: 26.8s
    343:	learn -0.09827989922	total: 14s	remaining: 26.7s
    344:	learn -0.09798383833	total: 14.1s	remaining: 26.7s
    345:	learn -0.09760790605	total: 14.1s	remaining: 26.6s
    346:	learn -0.09730796899	total: 14.1s	remaining: 26.6s
    347:	learn -0.09699659011	total: 14.2s	remaining: 26.5s
    348:	learn -0.09683393486	total: 14.2s	remaining: 26.5s
    349:	learn -0.09656730955	total: 14.3s	remaining: 26.5s
    350:	learn -0.09634307786	total: 14.3s	remaining: 26.4s
    351:	learn -0.0960876915	total: 14.3s	remaining: 26.4s
    352:	learn -0.09582205214	total: 14.4s	remaining: 26.3s
    353:	learn -0.09554055774	total: 14.4s	remaining: 26.3s
    354:	learn -0.0952060017	total: 14.4s	remaining: 26.2s
    355:	learn -0.09488818237	total: 14.5s	remaining: 26.2s
    356:	learn -0.09460548089	total: 14.5s	remaining: 26.1s
    357:	learn -0.0943005059	total: 14.5s	remaining: 26.1s
    358:	learn -0.09410764408	total: 14.6s	remaining: 26s
    359:	learn -0.09390410968	total: 14.6s	remaining: 26s
    360:	learn -0.09370060774	total: 14.6s	remaining: 25.9s
    361:	learn -0.09342774541	total: 14.7s	remaining: 25.9s
    362:	learn -0.09319007395	total: 14.7s	remaining: 25.8s
    363:	learn -0.09295875457	total: 14.8s	remaining: 25.8s
    364:	learn -0.09278737893	total: 14.8s	remaining: 25.7s
    365:	learn -0.09261907082	total: 14.8s	remaining: 25.7s
    366:	learn -0.09239342524	total: 14.9s	remaining: 25.7s
    367:	learn -0.09205641198	total: 14.9s	remaining: 25.6s
    368:	learn -0.09185672408	total: 15s	remaining: 25.6s
    369:	learn -0.09156928712	total: 15s	remaining: 25.5s
    370:	learn -0.09139071065	total: 15s	remaining: 25.5s
    371:	learn -0.09117501833	total: 15.1s	remaining: 25.4s
    372:	learn -0.09091573921	total: 15.1s	remaining: 25.4s
    373:	learn -0.09062915405	total: 15.2s	remaining: 25.4s
    374:	learn -0.09045569345	total: 15.2s	remaining: 25.3s
    375:	learn -0.09026057938	total: 15.2s	remaining: 25.3s
    376:	learn -0.09003770981	total: 15.3s	remaining: 25.2s
    377:	learn -0.08979365082	total: 15.3s	remaining: 25.2s
    378:	learn -0.08959731282	total: 15.3s	remaining: 25.1s
    379:	learn -0.08933546051	total: 15.4s	remaining: 25.1s
    380:	learn -0.08916688711	total: 15.4s	remaining: 25.1s
    381:	learn -0.08901968768	total: 15.5s	remaining: 25s
    382:	learn -0.08890263401	total: 15.5s	remaining: 24.9s
    383:	learn -0.08866539072	total: 15.5s	remaining: 24.9s
    384:	learn -0.08841535589	total: 15.6s	remaining: 24.9s
    385:	learn -0.08825015059	total: 15.6s	remaining: 24.8s
    386:	learn -0.08809621225	total: 15.6s	remaining: 24.8s
    387:	learn -0.08782229479	total: 15.7s	remaining: 24.7s
    388:	learn -0.0875357113	total: 15.7s	remaining: 24.7s
    389:	learn -0.08737481247	total: 15.7s	remaining: 24.6s
    390:	learn -0.0871857442	total: 15.8s	remaining: 24.6s
    391:	learn -0.08692390385	total: 15.8s	remaining: 24.5s
    392:	learn -0.08673648443	total: 15.8s	remaining: 24.5s
    393:	learn -0.08650536659	total: 15.9s	remaining: 24.4s
    394:	learn -0.08634350535	total: 15.9s	remaining: 24.4s
    395:	learn -0.08614948599	total: 16s	remaining: 24.3s
    396:	learn -0.08589104237	total: 16s	remaining: 24.3s
    397:	learn -0.08566966709	total: 16s	remaining: 24.2s
    398:	learn -0.08545388427	total: 16.1s	remaining: 24.2s
    399:	learn -0.08520551326	total: 16.1s	remaining: 24.1s
    400:	learn -0.085032007	total: 16.1s	remaining: 24.1s
    401:	learn -0.08491052813	total: 16.2s	remaining: 24.1s
    402:	learn -0.08473178096	total: 16.2s	remaining: 24s
    403:	learn -0.08459370289	total: 16.2s	remaining: 24s
    404:	learn -0.08440621345	total: 16.3s	remaining: 23.9s
    405:	learn -0.08427459235	total: 16.3s	remaining: 23.9s
    406:	learn -0.08407451802	total: 16.4s	remaining: 23.8s
    407:	learn -0.08385277537	total: 16.4s	remaining: 23.8s
    408:	learn -0.08363555934	total: 16.4s	remaining: 23.7s
    409:	learn -0.08350709526	total: 16.5s	remaining: 23.7s
    410:	learn -0.08337444256	total: 16.5s	remaining: 23.7s
    411:	learn -0.08319314317	total: 16.5s	remaining: 23.6s
    412:	learn -0.08296773419	total: 16.6s	remaining: 23.6s
    413:	learn -0.08275835359	total: 16.6s	remaining: 23.5s
    414:	learn -0.08252031678	total: 16.6s	remaining: 23.5s
    415:	learn -0.0823665671	total: 16.7s	remaining: 23.4s
    416:	learn -0.08203133354	total: 16.7s	remaining: 23.4s
    417:	learn -0.0818110467	total: 16.8s	remaining: 23.3s
    418:	learn -0.08170984721	total: 16.8s	remaining: 23.3s
    419:	learn -0.08157167901	total: 16.8s	remaining: 23.3s
    420:	learn -0.08143676014	total: 16.9s	remaining: 23.2s
    421:	learn -0.08130425889	total: 16.9s	remaining: 23.2s
    422:	learn -0.08115921268	total: 16.9s	remaining: 23.1s
    423:	learn -0.08100700311	total: 17s	remaining: 23.1s
    424:	learn -0.08079846475	total: 17s	remaining: 23s
    425:	learn -0.08066566261	total: 17.1s	remaining: 23s
    426:	learn -0.08047401548	total: 17.1s	remaining: 22.9s
    427:	learn -0.08030313435	total: 17.1s	remaining: 22.9s
    428:	learn -0.08014513974	total: 17.2s	remaining: 22.9s
    429:	learn -0.07998405976	total: 17.2s	remaining: 22.8s
    430:	learn -0.07976197718	total: 17.3s	remaining: 22.8s
    431:	learn -0.0796064726	total: 17.3s	remaining: 22.7s
    432:	learn -0.07942479716	total: 17.3s	remaining: 22.7s
    433:	learn -0.07924550669	total: 17.4s	remaining: 22.7s
    434:	learn -0.07913122706	total: 17.4s	remaining: 22.6s
    435:	learn -0.07886776849	total: 17.4s	remaining: 22.6s
    436:	learn -0.07875078504	total: 17.5s	remaining: 22.5s
    437:	learn -0.07847734909	total: 17.5s	remaining: 22.5s
    438:	learn -0.07836034383	total: 17.6s	remaining: 22.4s
    439:	learn -0.07815463486	total: 17.6s	remaining: 22.4s
    440:	learn -0.07799778529	total: 17.6s	remaining: 22.3s
    441:	learn -0.07773183065	total: 17.7s	remaining: 22.3s
    442:	learn -0.07761212543	total: 17.7s	remaining: 22.3s
    443:	learn -0.07744869905	total: 17.7s	remaining: 22.2s
    444:	learn -0.07729191599	total: 17.8s	remaining: 22.2s
    445:	learn -0.07709181952	total: 17.8s	remaining: 22.1s
    446:	learn -0.07699973349	total: 17.8s	remaining: 22.1s
    447:	learn -0.07681806387	total: 17.9s	remaining: 22s
    448:	learn -0.07660529953	total: 17.9s	remaining: 22s
    449:	learn -0.07645858059	total: 17.9s	remaining: 21.9s
    450:	learn -0.07635112981	total: 18s	remaining: 21.9s
    451:	learn -0.07618885677	total: 18s	remaining: 21.8s
    452:	learn -0.07607675948	total: 18s	remaining: 21.8s
    453:	learn -0.07596893939	total: 18.1s	remaining: 21.7s
    454:	learn -0.07578983052	total: 18.1s	remaining: 21.7s
    455:	learn -0.07557445569	total: 18.2s	remaining: 21.7s
    456:	learn -0.07544529231	total: 18.2s	remaining: 21.6s
    457:	learn -0.07522483386	total: 18.2s	remaining: 21.6s
    458:	learn -0.07512421819	total: 18.3s	remaining: 21.5s
    459:	learn -0.07498559148	total: 18.3s	remaining: 21.5s
    460:	learn -0.07484184625	total: 18.3s	remaining: 21.4s
    461:	learn -0.074725304	total: 18.4s	remaining: 21.4s
    462:	learn -0.07453723056	total: 18.4s	remaining: 21.3s
    463:	learn -0.07437966917	total: 18.4s	remaining: 21.3s
    464:	learn -0.07424954439	total: 18.5s	remaining: 21.2s
    465:	learn -0.07414933944	total: 18.5s	remaining: 21.2s
    466:	learn -0.07406094322	total: 18.5s	remaining: 21.2s
    467:	learn -0.07394300078	total: 18.6s	remaining: 21.1s
    468:	learn -0.07373492331	total: 18.6s	remaining: 21.1s
    469:	learn -0.07352786582	total: 18.7s	remaining: 21s
    470:	learn -0.0733408863	total: 18.7s	remaining: 21s
    471:	learn -0.07319340323	total: 18.7s	remaining: 21s
    472:	learn -0.07294114512	total: 18.8s	remaining: 20.9s
    473:	learn -0.07285316329	total: 18.8s	remaining: 20.9s
    474:	learn -0.07263725254	total: 18.8s	remaining: 20.8s
    475:	learn -0.0725250007	total: 18.9s	remaining: 20.8s
    476:	learn -0.0723298634	total: 18.9s	remaining: 20.7s
    477:	learn -0.07211513796	total: 18.9s	remaining: 20.7s
    478:	learn -0.07201194962	total: 19s	remaining: 20.6s
    479:	learn -0.07186576347	total: 19s	remaining: 20.6s
    480:	learn -0.0717137996	total: 19.1s	remaining: 20.6s
    481:	learn -0.07161013258	total: 19.1s	remaining: 20.5s
    482:	learn -0.07147187973	total: 19.2s	remaining: 20.5s
    483:	learn -0.07132591353	total: 19.2s	remaining: 20.5s
    484:	learn -0.07113266392	total: 19.2s	remaining: 20.4s
    485:	learn -0.07103504649	total: 19.3s	remaining: 20.4s
    486:	learn -0.07093261701	total: 19.3s	remaining: 20.4s
    487:	learn -0.07079949502	total: 19.4s	remaining: 20.3s
    488:	learn -0.070623346	total: 19.4s	remaining: 20.3s
    489:	learn -0.07048948202	total: 19.5s	remaining: 20.2s
    490:	learn -0.07033569425	total: 19.5s	remaining: 20.2s
    491:	learn -0.07021483352	total: 19.5s	remaining: 20.2s
    492:	learn -0.07010366802	total: 19.6s	remaining: 20.1s
    493:	learn -0.06996657197	total: 19.6s	remaining: 20.1s
    494:	learn -0.06979304685	total: 19.7s	remaining: 20.1s
    495:	learn -0.06972788975	total: 19.7s	remaining: 20s
    496:	learn -0.06958027123	total: 19.7s	remaining: 20s
    497:	learn -0.06943975148	total: 19.8s	remaining: 19.9s
    498:	learn -0.06934168649	total: 19.8s	remaining: 19.9s
    499:	learn -0.06921174165	total: 19.9s	remaining: 19.9s
    500:	learn -0.0690198619	total: 19.9s	remaining: 19.8s
    501:	learn -0.06889527294	total: 19.9s	remaining: 19.8s
    502:	learn -0.06873290733	total: 20s	remaining: 19.7s
    503:	learn -0.06862478394	total: 20s	remaining: 19.7s
    504:	learn -0.06846176611	total: 20s	remaining: 19.6s
    505:	learn -0.06834438779	total: 20.1s	remaining: 19.6s
    506:	learn -0.06818498172	total: 20.1s	remaining: 19.6s
    507:	learn -0.0680437677	total: 20.2s	remaining: 19.5s
    508:	learn -0.06793373002	total: 20.2s	remaining: 19.5s
    509:	learn -0.06776631451	total: 20.3s	remaining: 19.5s
    510:	learn -0.06763658558	total: 20.3s	remaining: 19.5s
    511:	learn -0.06746919095	total: 20.4s	remaining: 19.4s
    512:	learn -0.0672423994	total: 20.4s	remaining: 19.4s
    513:	learn -0.06709140276	total: 20.4s	remaining: 19.3s
    514:	learn -0.06692668831	total: 20.5s	remaining: 19.3s
    515:	learn -0.06678027581	total: 20.5s	remaining: 19.3s
    516:	learn -0.06667541082	total: 20.6s	remaining: 19.2s
    517:	learn -0.06657599615	total: 20.6s	remaining: 19.2s
    518:	learn -0.06650929465	total: 20.7s	remaining: 19.1s
    519:	learn -0.06640819285	total: 20.7s	remaining: 19.1s
    520:	learn -0.06631457991	total: 20.7s	remaining: 19.1s
    521:	learn -0.06623071493	total: 20.8s	remaining: 19s
    522:	learn -0.06613925036	total: 20.8s	remaining: 19s
    523:	learn -0.06599976725	total: 20.9s	remaining: 18.9s
    524:	learn -0.06589013099	total: 20.9s	remaining: 18.9s
    525:	learn -0.06579345334	total: 20.9s	remaining: 18.9s
    526:	learn -0.06568805028	total: 21s	remaining: 18.8s
    527:	learn -0.06553841377	total: 21s	remaining: 18.8s
    528:	learn -0.06538446684	total: 21.1s	remaining: 18.7s
    529:	learn -0.06523591063	total: 21.1s	remaining: 18.7s
    530:	learn -0.06504399998	total: 21.1s	remaining: 18.7s
    531:	learn -0.06494301571	total: 21.2s	remaining: 18.6s
    532:	learn -0.06479000516	total: 21.2s	remaining: 18.6s
    533:	learn -0.06471841924	total: 21.2s	remaining: 18.5s
    534:	learn -0.06457257271	total: 21.3s	remaining: 18.5s
    535:	learn -0.06442668317	total: 21.3s	remaining: 18.5s
    536:	learn -0.06436899886	total: 21.4s	remaining: 18.4s
    537:	learn -0.06419290899	total: 21.4s	remaining: 18.4s
    538:	learn -0.06412206799	total: 21.4s	remaining: 18.3s
    539:	learn -0.06397807804	total: 21.5s	remaining: 18.3s
    540:	learn -0.06378228159	total: 21.5s	remaining: 18.3s
    541:	learn -0.06366415803	total: 21.6s	remaining: 18.2s
    542:	learn -0.06356049527	total: 21.6s	remaining: 18.2s
    543:	learn -0.063431437	total: 21.7s	remaining: 18.2s
    544:	learn -0.06334789479	total: 21.7s	remaining: 18.1s
    545:	learn -0.06321405304	total: 21.8s	remaining: 18.1s
    546:	learn -0.06312467474	total: 21.8s	remaining: 18.1s
    547:	learn -0.06305035878	total: 21.9s	remaining: 18s
    548:	learn -0.06297205435	total: 21.9s	remaining: 18s
    549:	learn -0.06280273036	total: 22s	remaining: 18s
    550:	learn -0.06266648428	total: 22s	remaining: 17.9s
    551:	learn -0.06246178618	total: 22s	remaining: 17.9s
    552:	learn -0.06233073922	total: 22.1s	remaining: 17.9s
    553:	learn -0.0621899193	total: 22.1s	remaining: 17.8s
    554:	learn -0.06204269141	total: 22.2s	remaining: 17.8s
    555:	learn -0.06189532282	total: 22.2s	remaining: 17.8s
    556:	learn -0.06176243879	total: 22.3s	remaining: 17.7s
    557:	learn -0.06157646509	total: 22.3s	remaining: 17.7s
    558:	learn -0.06148353075	total: 22.4s	remaining: 17.6s
    559:	learn -0.06135140973	total: 22.4s	remaining: 17.6s
    560:	learn -0.06127522928	total: 22.4s	remaining: 17.6s
    561:	learn -0.06116872218	total: 22.5s	remaining: 17.5s
    562:	learn -0.06105165064	total: 22.5s	remaining: 17.5s
    563:	learn -0.06090950507	total: 22.6s	remaining: 17.4s
    564:	learn -0.0607640995	total: 22.6s	remaining: 17.4s
    565:	learn -0.06065599963	total: 22.6s	remaining: 17.4s
    566:	learn -0.06051387988	total: 22.7s	remaining: 17.3s
    567:	learn -0.06037833377	total: 22.7s	remaining: 17.3s
    568:	learn -0.06025650852	total: 22.7s	remaining: 17.2s
    569:	learn -0.06015831561	total: 22.8s	remaining: 17.2s
    570:	learn -0.06006958964	total: 22.8s	remaining: 17.1s
    571:	learn -0.05994571401	total: 22.9s	remaining: 17.1s
    572:	learn -0.05981946283	total: 22.9s	remaining: 17.1s
    573:	learn -0.05973570849	total: 22.9s	remaining: 17s
    574:	learn -0.05959777638	total: 23s	remaining: 17s
    575:	learn -0.05945660747	total: 23s	remaining: 17s
    576:	learn -0.05934416978	total: 23.1s	remaining: 16.9s
    577:	learn -0.05924193305	total: 23.1s	remaining: 16.9s
    578:	learn -0.05917690717	total: 23.2s	remaining: 16.8s
    579:	learn -0.05910006107	total: 23.2s	remaining: 16.8s
    580:	learn -0.05900414245	total: 23.2s	remaining: 16.8s
    581:	learn -0.05892482049	total: 23.3s	remaining: 16.7s
    582:	learn -0.05885352466	total: 23.3s	remaining: 16.7s
    583:	learn -0.05874708269	total: 23.4s	remaining: 16.6s
    584:	learn -0.05867157443	total: 23.4s	remaining: 16.6s
    585:	learn -0.05853667584	total: 23.4s	remaining: 16.6s
    586:	learn -0.05837239161	total: 23.5s	remaining: 16.5s
    587:	learn -0.05826150437	total: 23.5s	remaining: 16.5s
    588:	learn -0.05818768754	total: 23.5s	remaining: 16.4s
    589:	learn -0.05810895721	total: 23.6s	remaining: 16.4s
    590:	learn -0.05800579849	total: 23.6s	remaining: 16.3s
    591:	learn -0.05792458727	total: 23.6s	remaining: 16.3s
    592:	learn -0.05783107614	total: 23.7s	remaining: 16.3s
    593:	learn -0.05774763501	total: 23.7s	remaining: 16.2s
    594:	learn -0.05765436673	total: 23.8s	remaining: 16.2s
    595:	learn -0.05758865899	total: 23.8s	remaining: 16.1s
    596:	learn -0.05752621308	total: 23.8s	remaining: 16.1s
    597:	learn -0.05739949674	total: 23.9s	remaining: 16s
    598:	learn -0.05729703101	total: 23.9s	remaining: 16s
    599:	learn -0.05715586524	total: 23.9s	remaining: 16s
    600:	learn -0.05707287745	total: 24s	remaining: 15.9s
    601:	learn -0.05696794132	total: 24s	remaining: 15.9s
    602:	learn -0.05684770508	total: 24.1s	remaining: 15.8s
    603:	learn -0.05677687635	total: 24.1s	remaining: 15.8s
    604:	learn -0.05668617161	total: 24.1s	remaining: 15.8s
    605:	learn -0.05660233093	total: 24.2s	remaining: 15.7s
    606:	learn -0.056463992	total: 24.2s	remaining: 15.7s
    607:	learn -0.05638962555	total: 24.3s	remaining: 15.6s
    608:	learn -0.05621446395	total: 24.3s	remaining: 15.6s
    609:	learn -0.05617060228	total: 24.3s	remaining: 15.6s
    610:	learn -0.0560103787	total: 24.4s	remaining: 15.5s
    611:	learn -0.05589711932	total: 24.4s	remaining: 15.5s
    612:	learn -0.05578348203	total: 24.4s	remaining: 15.4s
    613:	learn -0.05568102269	total: 24.5s	remaining: 15.4s
    614:	learn -0.05557250193	total: 24.5s	remaining: 15.3s
    615:	learn -0.0555328576	total: 24.6s	remaining: 15.3s
    616:	learn -0.05542420604	total: 24.6s	remaining: 15.3s
    617:	learn -0.05534587312	total: 24.6s	remaining: 15.2s
    618:	learn -0.05529497757	total: 24.7s	remaining: 15.2s
    619:	learn -0.05516434598	total: 24.7s	remaining: 15.1s
    620:	learn -0.05499118593	total: 24.7s	remaining: 15.1s
    621:	learn -0.05486684934	total: 24.8s	remaining: 15.1s
    622:	learn -0.05479936079	total: 24.8s	remaining: 15s
    623:	learn -0.05470669232	total: 24.8s	remaining: 15s
    624:	learn -0.05460392865	total: 24.9s	remaining: 14.9s
    625:	learn -0.05447300374	total: 24.9s	remaining: 14.9s
    626:	learn -0.05437403874	total: 25s	remaining: 14.8s
    627:	learn -0.0542838952	total: 25s	remaining: 14.8s
    628:	learn -0.05423268183	total: 25s	remaining: 14.8s
    629:	learn -0.0541400869	total: 25.1s	remaining: 14.7s
    630:	learn -0.05405350031	total: 25.1s	remaining: 14.7s
    631:	learn -0.05398017242	total: 25.1s	remaining: 14.6s
    632:	learn -0.05391624194	total: 25.2s	remaining: 14.6s
    633:	learn -0.05384218486	total: 25.2s	remaining: 14.5s
    634:	learn -0.0536738999	total: 25.2s	remaining: 14.5s
    635:	learn -0.05359776793	total: 25.3s	remaining: 14.5s
    636:	learn -0.05354378345	total: 25.3s	remaining: 14.4s
    637:	learn -0.05342851588	total: 25.3s	remaining: 14.4s
    638:	learn -0.05335268054	total: 25.4s	remaining: 14.3s
    639:	learn -0.05321740637	total: 25.4s	remaining: 14.3s
    640:	learn -0.05311666071	total: 25.4s	remaining: 14.3s
    641:	learn -0.05300001753	total: 25.5s	remaining: 14.2s
    642:	learn -0.05292652275	total: 25.5s	remaining: 14.2s
    643:	learn -0.05284865274	total: 25.6s	remaining: 14.1s
    644:	learn -0.05281180334	total: 25.6s	remaining: 14.1s
    645:	learn -0.05264952074	total: 25.6s	remaining: 14s
    646:	learn -0.05250968551	total: 25.7s	remaining: 14s
    647:	learn -0.05239789907	total: 25.7s	remaining: 14s
    648:	learn -0.05228616948	total: 25.7s	remaining: 13.9s
    649:	learn -0.05217698466	total: 25.8s	remaining: 13.9s
    650:	learn -0.05207954352	total: 25.8s	remaining: 13.8s
    651:	learn -0.0519733934	total: 25.8s	remaining: 13.8s
    652:	learn -0.05188081897	total: 25.9s	remaining: 13.8s
    653:	learn -0.05181940544	total: 25.9s	remaining: 13.7s
    654:	learn -0.05176069636	total: 26s	remaining: 13.7s
    655:	learn -0.0516932853	total: 26s	remaining: 13.6s
    656:	learn -0.05160919217	total: 26s	remaining: 13.6s
    657:	learn -0.05153671928	total: 26.1s	remaining: 13.5s
    658:	learn -0.05142668152	total: 26.1s	remaining: 13.5s
    659:	learn -0.05133802888	total: 26.1s	remaining: 13.5s
    660:	learn -0.05125096807	total: 26.2s	remaining: 13.4s
    661:	learn -0.05109955466	total: 26.2s	remaining: 13.4s
    662:	learn -0.0510001744	total: 26.2s	remaining: 13.3s
    663:	learn -0.05093760315	total: 26.3s	remaining: 13.3s
    664:	learn -0.05086331129	total: 26.3s	remaining: 13.3s
    665:	learn -0.05080999869	total: 26.4s	remaining: 13.2s
    666:	learn -0.05069732077	total: 26.4s	remaining: 13.2s
    667:	learn -0.05063239505	total: 26.4s	remaining: 13.1s
    668:	learn -0.05055791076	total: 26.5s	remaining: 13.1s
    669:	learn -0.05047603325	total: 26.5s	remaining: 13.1s
    670:	learn -0.05040820868	total: 26.5s	remaining: 13s
    671:	learn -0.05034044533	total: 26.6s	remaining: 13s
    672:	learn -0.05026114825	total: 26.6s	remaining: 12.9s
    673:	learn -0.05019700145	total: 26.7s	remaining: 12.9s
    674:	learn -0.05014377143	total: 26.7s	remaining: 12.9s
    675:	learn -0.05007267208	total: 26.7s	remaining: 12.8s
    676:	learn -0.0500168193	total: 26.8s	remaining: 12.8s
    677:	learn -0.04990881311	total: 26.8s	remaining: 12.7s
    678:	learn -0.04985426259	total: 26.8s	remaining: 12.7s
    679:	learn -0.04979362867	total: 26.9s	remaining: 12.6s
    680:	learn -0.04972894703	total: 26.9s	remaining: 12.6s
    681:	learn -0.0496465228	total: 26.9s	remaining: 12.6s
    682:	learn -0.04955200618	total: 27s	remaining: 12.5s
    683:	learn -0.04941582096	total: 27s	remaining: 12.5s
    684:	learn -0.04937423097	total: 27.1s	remaining: 12.4s
    685:	learn -0.04928368883	total: 27.1s	remaining: 12.4s
    686:	learn -0.04918075361	total: 27.1s	remaining: 12.4s
    687:	learn -0.04910897222	total: 27.2s	remaining: 12.3s
    688:	learn -0.04901775664	total: 27.2s	remaining: 12.3s
    689:	learn -0.04895555928	total: 27.2s	remaining: 12.2s
    690:	learn -0.04887932085	total: 27.3s	remaining: 12.2s
    691:	learn -0.04880616137	total: 27.3s	remaining: 12.1s
    692:	learn -0.04867781536	total: 27.3s	remaining: 12.1s
    693:	learn -0.04858353344	total: 27.4s	remaining: 12.1s
    694:	learn -0.04850394158	total: 27.4s	remaining: 12s
    695:	learn -0.0484429728	total: 27.4s	remaining: 12s
    696:	learn -0.04832351439	total: 27.5s	remaining: 11.9s
    697:	learn -0.04824199762	total: 27.5s	remaining: 11.9s
    698:	learn -0.04816377587	total: 27.5s	remaining: 11.9s
    699:	learn -0.04809962746	total: 27.6s	remaining: 11.8s
    700:	learn -0.04804584961	total: 27.6s	remaining: 11.8s
    701:	learn -0.04799251223	total: 27.6s	remaining: 11.7s
    702:	learn -0.04791848243	total: 27.7s	remaining: 11.7s
    703:	learn -0.04782352953	total: 27.7s	remaining: 11.7s
    704:	learn -0.04771477221	total: 27.8s	remaining: 11.6s
    705:	learn -0.04764123923	total: 27.8s	remaining: 11.6s
    706:	learn -0.0475868749	total: 27.8s	remaining: 11.5s
    707:	learn -0.04750723036	total: 27.9s	remaining: 11.5s
    708:	learn -0.04744553206	total: 27.9s	remaining: 11.5s
    709:	learn -0.04732614889	total: 28s	remaining: 11.4s
    710:	learn -0.04724941181	total: 28s	remaining: 11.4s
    711:	learn -0.047178689	total: 28.1s	remaining: 11.3s
    712:	learn -0.04712026438	total: 28.1s	remaining: 11.3s
    713:	learn -0.04707303854	total: 28.1s	remaining: 11.3s
    714:	learn -0.04701391429	total: 28.2s	remaining: 11.2s
    715:	learn -0.0469499901	total: 28.2s	remaining: 11.2s
    716:	learn -0.04689642987	total: 28.2s	remaining: 11.1s
    717:	learn -0.0468495465	total: 28.3s	remaining: 11.1s
    718:	learn -0.04681093651	total: 28.3s	remaining: 11.1s
    719:	learn -0.04675635864	total: 28.3s	remaining: 11s
    720:	learn -0.04668672083	total: 28.4s	remaining: 11s
    721:	learn -0.04663040508	total: 28.4s	remaining: 10.9s
    722:	learn -0.04656168534	total: 28.5s	remaining: 10.9s
    723:	learn -0.0464225171	total: 28.5s	remaining: 10.9s
    724:	learn -0.04632180819	total: 28.5s	remaining: 10.8s
    725:	learn -0.04624473083	total: 28.6s	remaining: 10.8s
    726:	learn -0.04618350926	total: 28.6s	remaining: 10.7s
    727:	learn -0.04609668393	total: 28.6s	remaining: 10.7s
    728:	learn -0.04600422257	total: 28.7s	remaining: 10.7s
    729:	learn -0.04594471399	total: 28.7s	remaining: 10.6s
    730:	learn -0.04587766344	total: 28.8s	remaining: 10.6s
    731:	learn -0.04581089215	total: 28.8s	remaining: 10.5s
    732:	learn -0.04573850882	total: 28.8s	remaining: 10.5s
    733:	learn -0.04566668593	total: 28.9s	remaining: 10.5s
    734:	learn -0.04560293734	total: 28.9s	remaining: 10.4s
    735:	learn -0.04550948383	total: 29s	remaining: 10.4s
    736:	learn -0.04543606656	total: 29s	remaining: 10.3s
    737:	learn -0.04536932086	total: 29s	remaining: 10.3s
    738:	learn -0.04531561867	total: 29.1s	remaining: 10.3s
    739:	learn -0.04523687093	total: 29.1s	remaining: 10.2s
    740:	learn -0.04516533229	total: 29.1s	remaining: 10.2s
    741:	learn -0.04510133512	total: 29.2s	remaining: 10.1s
    742:	learn -0.04498526146	total: 29.2s	remaining: 10.1s
    743:	learn -0.04492508079	total: 29.3s	remaining: 10.1s
    744:	learn -0.04488035148	total: 29.3s	remaining: 10s
    745:	learn -0.04481920663	total: 29.3s	remaining: 9.99s
    746:	learn -0.04477853884	total: 29.4s	remaining: 9.95s
    747:	learn -0.04470371451	total: 29.4s	remaining: 9.91s
    748:	learn -0.04463406674	total: 29.5s	remaining: 9.87s
    749:	learn -0.04454246037	total: 29.5s	remaining: 9.83s
    750:	learn -0.04448187854	total: 29.5s	remaining: 9.79s
    751:	learn -0.04440196368	total: 29.6s	remaining: 9.75s
    752:	learn -0.04434271311	total: 29.6s	remaining: 9.71s
    753:	learn -0.04427214609	total: 29.6s	remaining: 9.67s
    754:	learn -0.04411195631	total: 29.7s	remaining: 9.63s
    755:	learn -0.04407198702	total: 29.7s	remaining: 9.59s
    756:	learn -0.04402339564	total: 29.8s	remaining: 9.55s
    757:	learn -0.04388682337	total: 29.8s	remaining: 9.51s
    758:	learn -0.04381381868	total: 29.8s	remaining: 9.47s
    759:	learn -0.04374249576	total: 29.9s	remaining: 9.43s
    760:	learn -0.04367527473	total: 29.9s	remaining: 9.39s
    761:	learn -0.04362754713	total: 29.9s	remaining: 9.35s
    762:	learn -0.04358421424	total: 30s	remaining: 9.31s
    763:	learn -0.04353561855	total: 30s	remaining: 9.27s
    764:	learn -0.04345629096	total: 30s	remaining: 9.23s
    765:	learn -0.04334540477	total: 30.1s	remaining: 9.19s
    766:	learn -0.04330899771	total: 30.1s	remaining: 9.15s
    767:	learn -0.04321971629	total: 30.2s	remaining: 9.11s
    768:	learn -0.04316930122	total: 30.2s	remaining: 9.07s
    769:	learn -0.04312539594	total: 30.2s	remaining: 9.03s
    770:	learn -0.04300406215	total: 30.3s	remaining: 8.99s
    771:	learn -0.04294255848	total: 30.3s	remaining: 8.95s
    772:	learn -0.04288683824	total: 30.3s	remaining: 8.91s
    773:	learn -0.04283321973	total: 30.4s	remaining: 8.87s
    774:	learn -0.04276162519	total: 30.4s	remaining: 8.83s
    775:	learn -0.04268241529	total: 30.4s	remaining: 8.79s
    776:	learn -0.04264705662	total: 30.5s	remaining: 8.75s
    777:	learn -0.04261359064	total: 30.5s	remaining: 8.7s
    778:	learn -0.04257219224	total: 30.5s	remaining: 8.66s
    779:	learn -0.04247757602	total: 30.6s	remaining: 8.62s
    780:	learn -0.04242166621	total: 30.6s	remaining: 8.58s
    781:	learn -0.04233771501	total: 30.6s	remaining: 8.54s
    782:	learn -0.04223951977	total: 30.7s	remaining: 8.5s
    783:	learn -0.04217501078	total: 30.7s	remaining: 8.46s
    784:	learn -0.04213015916	total: 30.8s	remaining: 8.42s
    785:	learn -0.04209671153	total: 30.8s	remaining: 8.38s
    786:	learn -0.04201567733	total: 30.8s	remaining: 8.34s
    787:	learn -0.04195216632	total: 30.8s	remaining: 8.3s
    788:	learn -0.0419021494	total: 30.9s	remaining: 8.26s
    789:	learn -0.04187345567	total: 30.9s	remaining: 8.22s
    790:	learn -0.04182711101	total: 31s	remaining: 8.18s
    791:	learn -0.04176235314	total: 31s	remaining: 8.14s
    792:	learn -0.04172175016	total: 31s	remaining: 8.1s
    793:	learn -0.04167800494	total: 31.1s	remaining: 8.06s
    794:	learn -0.0416362486	total: 31.1s	remaining: 8.02s
    795:	learn -0.04154311932	total: 31.1s	remaining: 7.98s
    796:	learn -0.041496208	total: 31.2s	remaining: 7.94s
    797:	learn -0.04142994741	total: 31.2s	remaining: 7.9s
    798:	learn -0.04136285604	total: 31.2s	remaining: 7.86s
    799:	learn -0.04132670165	total: 31.3s	remaining: 7.82s
    800:	learn -0.04129742668	total: 31.3s	remaining: 7.78s
    801:	learn -0.04124943098	total: 31.4s	remaining: 7.74s
    802:	learn -0.04115703027	total: 31.4s	remaining: 7.7s
    803:	learn -0.04105730649	total: 31.4s	remaining: 7.67s
    804:	learn -0.04099126739	total: 31.5s	remaining: 7.63s
    805:	learn -0.04094263249	total: 31.5s	remaining: 7.59s
    806:	learn -0.04089947836	total: 31.6s	remaining: 7.55s
    807:	learn -0.04086597952	total: 31.6s	remaining: 7.51s
    808:	learn -0.04083459842	total: 31.7s	remaining: 7.47s
    809:	learn -0.04078723613	total: 31.7s	remaining: 7.43s
    810:	learn -0.04076009536	total: 31.7s	remaining: 7.4s
    811:	learn -0.0407109858	total: 31.8s	remaining: 7.36s
    812:	learn -0.04061227417	total: 31.8s	remaining: 7.32s
    813:	learn -0.04054038831	total: 31.9s	remaining: 7.28s
    814:	learn -0.04050561512	total: 31.9s	remaining: 7.24s
    815:	learn -0.04047035111	total: 31.9s	remaining: 7.2s
    816:	learn -0.04044300322	total: 32s	remaining: 7.16s
    817:	learn -0.04036755597	total: 32s	remaining: 7.12s
    818:	learn -0.04031005031	total: 32s	remaining: 7.08s
    819:	learn -0.04027427639	total: 32.1s	remaining: 7.04s
    820:	learn -0.0402252278	total: 32.1s	remaining: 7s
    821:	learn -0.04018762802	total: 32.1s	remaining: 6.96s
    822:	learn -0.04016119895	total: 32.2s	remaining: 6.92s
    823:	learn -0.04012267615	total: 32.2s	remaining: 6.88s
    824:	learn -0.04008288192	total: 32.3s	remaining: 6.84s
    825:	learn -0.04000183042	total: 32.3s	remaining: 6.8s
    826:	learn -0.03996436296	total: 32.3s	remaining: 6.76s
    827:	learn -0.03992118084	total: 32.4s	remaining: 6.72s
    828:	learn -0.03987283879	total: 32.4s	remaining: 6.68s
    829:	learn -0.03983520068	total: 32.4s	remaining: 6.64s
    830:	learn -0.03980314178	total: 32.5s	remaining: 6.6s
    831:	learn -0.03974263123	total: 32.5s	remaining: 6.56s
    832:	learn -0.0396839157	total: 32.5s	remaining: 6.52s
    833:	learn -0.03962004218	total: 32.6s	remaining: 6.48s
    834:	learn -0.03951579626	total: 32.6s	remaining: 6.44s
    835:	learn -0.03947233345	total: 32.6s	remaining: 6.4s
    836:	learn -0.03941233502	total: 32.7s	remaining: 6.36s
    837:	learn -0.03936226063	total: 32.7s	remaining: 6.32s
    838:	learn -0.03932573882	total: 32.7s	remaining: 6.28s
    839:	learn -0.0392726898	total: 32.8s	remaining: 6.24s
    840:	learn -0.03922174426	total: 32.8s	remaining: 6.2s
    841:	learn -0.03916722428	total: 32.8s	remaining: 6.16s
    842:	learn -0.03912799275	total: 32.9s	remaining: 6.12s
    843:	learn -0.03909820415	total: 32.9s	remaining: 6.08s
    844:	learn -0.03906218027	total: 32.9s	remaining: 6.04s
    845:	learn -0.03899814274	total: 33s	remaining: 6s
    846:	learn -0.03894932127	total: 33s	remaining: 5.96s
    847:	learn -0.03882554969	total: 33s	remaining: 5.92s
    848:	learn -0.0387784185	total: 33.1s	remaining: 5.88s
    849:	learn -0.03874365097	total: 33.1s	remaining: 5.84s
    850:	learn -0.03870606052	total: 33.1s	remaining: 5.8s
    851:	learn -0.03864492531	total: 33.2s	remaining: 5.76s
    852:	learn -0.03858450124	total: 33.2s	remaining: 5.72s
    853:	learn -0.03854811809	total: 33.3s	remaining: 5.68s
    854:	learn -0.03849551923	total: 33.3s	remaining: 5.65s
    855:	learn -0.03844958992	total: 33.3s	remaining: 5.61s
    856:	learn -0.03839809987	total: 33.4s	remaining: 5.57s
    857:	learn -0.03835994136	total: 33.4s	remaining: 5.53s
    858:	learn -0.03832115752	total: 33.4s	remaining: 5.49s
    859:	learn -0.03828011027	total: 33.5s	remaining: 5.45s
    860:	learn -0.03825783577	total: 33.5s	remaining: 5.41s
    861:	learn -0.03822991576	total: 33.5s	remaining: 5.37s
    862:	learn -0.03818010566	total: 33.6s	remaining: 5.33s
    863:	learn -0.03812463963	total: 33.6s	remaining: 5.29s
    864:	learn -0.03808398863	total: 33.6s	remaining: 5.25s
    865:	learn -0.03805006314	total: 33.7s	remaining: 5.21s
    866:	learn -0.03800235836	total: 33.7s	remaining: 5.17s
    867:	learn -0.03793253407	total: 33.8s	remaining: 5.13s
    868:	learn -0.03788545067	total: 33.8s	remaining: 5.09s
    869:	learn -0.03785259625	total: 33.8s	remaining: 5.05s
    870:	learn -0.03780588362	total: 33.9s	remaining: 5.01s
    871:	learn -0.03773502072	total: 33.9s	remaining: 4.97s
    872:	learn -0.03768960173	total: 33.9s	remaining: 4.93s
    873:	learn -0.0376064202	total: 34s	remaining: 4.9s
    874:	learn -0.03754813134	total: 34s	remaining: 4.86s
    875:	learn -0.03743973462	total: 34s	remaining: 4.82s
    876:	learn -0.03737468103	total: 34.1s	remaining: 4.78s
    877:	learn -0.03732358235	total: 34.1s	remaining: 4.74s
    878:	learn -0.03725864085	total: 34.2s	remaining: 4.7s
    879:	learn -0.03720957748	total: 34.2s	remaining: 4.66s
    880:	learn -0.03716359887	total: 34.2s	remaining: 4.62s
    881:	learn -0.03710597778	total: 34.3s	remaining: 4.58s
    882:	learn -0.03704661531	total: 34.3s	remaining: 4.55s
    883:	learn -0.03700907892	total: 34.4s	remaining: 4.51s
    884:	learn -0.03695639679	total: 34.4s	remaining: 4.47s
    885:	learn -0.03692096645	total: 34.4s	remaining: 4.43s
    886:	learn -0.03689318604	total: 34.5s	remaining: 4.39s
    887:	learn -0.03686228202	total: 34.5s	remaining: 4.35s
    888:	learn -0.03682084545	total: 34.5s	remaining: 4.31s
    889:	learn -0.03676922082	total: 34.6s	remaining: 4.27s
    890:	learn -0.0367105672	total: 34.6s	remaining: 4.23s
    891:	learn -0.03665295125	total: 34.6s	remaining: 4.19s
    892:	learn -0.03661126177	total: 34.7s	remaining: 4.15s
    893:	learn -0.0365667389	total: 34.7s	remaining: 4.12s
    894:	learn -0.03646602441	total: 34.7s	remaining: 4.08s
    895:	learn -0.03640530297	total: 34.8s	remaining: 4.04s
    896:	learn -0.03633237679	total: 34.8s	remaining: 4s
    897:	learn -0.03628794772	total: 34.9s	remaining: 3.96s
    898:	learn -0.03622215365	total: 34.9s	remaining: 3.92s
    899:	learn -0.03619060001	total: 34.9s	remaining: 3.88s
    900:	learn -0.0361592728	total: 35s	remaining: 3.84s
    901:	learn -0.03613185269	total: 35s	remaining: 3.8s
    902:	learn -0.03609449613	total: 35s	remaining: 3.77s
    903:	learn -0.0360472602	total: 35.1s	remaining: 3.73s
    904:	learn -0.03600973588	total: 35.1s	remaining: 3.69s
    905:	learn -0.03597706866	total: 35.2s	remaining: 3.65s
    906:	learn -0.03594183749	total: 35.2s	remaining: 3.61s
    907:	learn -0.03589644115	total: 35.3s	remaining: 3.57s
    908:	learn -0.03586366617	total: 35.3s	remaining: 3.53s
    909:	learn -0.03576492456	total: 35.3s	remaining: 3.5s
    910:	learn -0.03570888425	total: 35.4s	remaining: 3.46s
    911:	learn -0.03565494923	total: 35.4s	remaining: 3.42s
    912:	learn -0.03562654243	total: 35.5s	remaining: 3.38s
    913:	learn -0.03559816177	total: 35.5s	remaining: 3.34s
    914:	learn -0.03553960082	total: 35.6s	remaining: 3.3s
    915:	learn -0.03547297757	total: 35.6s	remaining: 3.27s
    916:	learn -0.03545633312	total: 35.7s	remaining: 3.23s
    917:	learn -0.03542446051	total: 35.7s	remaining: 3.19s
    918:	learn -0.03537686616	total: 35.7s	remaining: 3.15s
    919:	learn -0.03534709935	total: 35.8s	remaining: 3.11s
    920:	learn -0.03529966606	total: 35.8s	remaining: 3.07s
    921:	learn -0.03525214764	total: 35.8s	remaining: 3.03s
    922:	learn -0.03520054457	total: 35.9s	remaining: 2.99s
    923:	learn -0.03516845966	total: 35.9s	remaining: 2.95s
    924:	learn -0.03511636452	total: 35.9s	remaining: 2.91s
    925:	learn -0.03503071045	total: 36s	remaining: 2.88s
    926:	learn -0.0349593681	total: 36s	remaining: 2.84s
    927:	learn -0.03491986084	total: 36.1s	remaining: 2.8s
    928:	learn -0.03488467748	total: 36.1s	remaining: 2.76s
    929:	learn -0.03485361765	total: 36.1s	remaining: 2.72s
    930:	learn -0.03481295718	total: 36.2s	remaining: 2.68s
    931:	learn -0.03478024098	total: 36.2s	remaining: 2.64s
    932:	learn -0.03475290794	total: 36.2s	remaining: 2.6s
    933:	learn -0.03468428922	total: 36.3s	remaining: 2.56s
    934:	learn -0.03464728786	total: 36.3s	remaining: 2.52s
    935:	learn -0.03461838077	total: 36.4s	remaining: 2.49s
    936:	learn -0.03457583206	total: 36.4s	remaining: 2.45s
    937:	learn -0.03449601106	total: 36.5s	remaining: 2.41s
    938:	learn -0.03447102005	total: 36.5s	remaining: 2.37s
    939:	learn -0.03440048124	total: 36.6s	remaining: 2.33s
    940:	learn -0.03436391948	total: 36.6s	remaining: 2.29s
    941:	learn -0.03433117781	total: 36.6s	remaining: 2.25s
    942:	learn -0.03430358406	total: 36.7s	remaining: 2.22s
    943:	learn -0.03425898271	total: 36.7s	remaining: 2.18s
    944:	learn -0.03423373014	total: 36.8s	remaining: 2.14s
    945:	learn -0.03420713624	total: 36.8s	remaining: 2.1s
    946:	learn -0.03416579217	total: 36.8s	remaining: 2.06s
    947:	learn -0.03411423973	total: 36.9s	remaining: 2.02s
    948:	learn -0.03408230199	total: 36.9s	remaining: 1.98s
    949:	learn -0.03405711718	total: 37s	remaining: 1.95s
    950:	learn -0.03403118306	total: 37s	remaining: 1.91s
    951:	learn -0.03400049199	total: 37s	remaining: 1.87s
    952:	learn -0.03397805114	total: 37.1s	remaining: 1.83s
    953:	learn -0.03395147277	total: 37.1s	remaining: 1.79s
    954:	learn -0.03391937354	total: 37.2s	remaining: 1.75s
    955:	learn -0.0339047639	total: 37.2s	remaining: 1.71s
    956:	learn -0.03388817626	total: 37.2s	remaining: 1.67s
    957:	learn -0.03382338011	total: 37.3s	remaining: 1.63s
    958:	learn -0.03379434223	total: 37.3s	remaining: 1.59s
    959:	learn -0.03375464521	total: 37.3s	remaining: 1.55s
    960:	learn -0.03370634409	total: 37.4s	remaining: 1.52s
    961:	learn -0.03368048288	total: 37.4s	remaining: 1.48s
    962:	learn -0.03364889162	total: 37.4s	remaining: 1.44s
    963:	learn -0.03361492804	total: 37.5s	remaining: 1.4s
    964:	learn -0.03355551837	total: 37.5s	remaining: 1.36s
    965:	learn -0.03349817769	total: 37.6s	remaining: 1.32s
    966:	learn -0.03345779784	total: 37.6s	remaining: 1.28s
    967:	learn -0.03342535018	total: 37.6s	remaining: 1.24s
    968:	learn -0.03339862507	total: 37.7s	remaining: 1.2s
    969:	learn -0.03336090235	total: 37.7s	remaining: 1.17s
    970:	learn -0.03331805229	total: 37.7s	remaining: 1.13s
    971:	learn -0.03327882376	total: 37.8s	remaining: 1.09s
    972:	learn -0.03321909053	total: 37.8s	remaining: 1.05s
    973:	learn -0.03319298484	total: 37.8s	remaining: 1.01s
    974:	learn -0.03316640042	total: 37.9s	remaining: 971ms
    975:	learn -0.0331035021	total: 37.9s	remaining: 932ms
    976:	learn -0.03306294382	total: 37.9s	remaining: 893ms
    977:	learn -0.03302504063	total: 38s	remaining: 854ms
    978:	learn -0.03298066019	total: 38s	remaining: 815ms
    979:	learn -0.03295778156	total: 38s	remaining: 776ms
    980:	learn -0.03290598485	total: 38.1s	remaining: 737ms
    981:	learn -0.03286851891	total: 38.1s	remaining: 699ms
    982:	learn -0.0328414851	total: 38.2s	remaining: 660ms
    983:	learn -0.0328098182	total: 38.2s	remaining: 621ms
    984:	learn -0.03275620153	total: 38.2s	remaining: 582ms
    985:	learn -0.03273358573	total: 38.3s	remaining: 543ms
    986:	learn -0.03271687264	total: 38.3s	remaining: 504ms
    987:	learn -0.03265378137	total: 38.3s	remaining: 466ms
    988:	learn -0.03261620268	total: 38.4s	remaining: 427ms
    989:	learn -0.03256895319	total: 38.4s	remaining: 388ms
    990:	learn -0.03253403834	total: 38.4s	remaining: 349ms
    991:	learn -0.0324830947	total: 38.5s	remaining: 310ms
    992:	learn -0.03245540484	total: 38.5s	remaining: 272ms
    993:	learn -0.0324276877	total: 38.5s	remaining: 233ms
    994:	learn -0.0323955457	total: 38.6s	remaining: 194ms
    995:	learn -0.03236817871	total: 38.6s	remaining: 155ms
    996:	learn -0.03233099157	total: 38.6s	remaining: 116ms
    997:	learn -0.03229663746	total: 38.7s	remaining: 77.5ms
    998:	learn -0.03226163078	total: 38.7s	remaining: 38.8ms
    999:	learn -0.03222369767	total: 38.8s	remaining: 0us





    <catboost.core._CatBoostBase at 0x7f86ea28f310>




```python
feature_score = pd.DataFrame(list(zip(one_hot.dtypes.index, model.get_feature_importance(Pool(one_hot, label=y, cat_features=categorical_features_indices)))),
                columns=['Feature','Score'])
```


```python
feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
```


```python
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
```


![png](output_64_0.png)



```python
cm = pd.DataFrame()
cm['Satisfaction'] = y_test
cm['Predict'] = model.predict(X_test)
```


```python
mappingSatisfaction = {0:'Unsatisfied', 1: 'Neutral', 2: 'Satisfied'}
mappingPredict = {0.0:'Unsatisfied', 1.0: 'Neutral', 2.0: 'Satisfied'}
cm = cm.replace({'Satisfaction': mappingSatisfaction, 'Predict': mappingPredict})
```


```python
pd.crosstab(cm['Satisfaction'], cm['Predict'], margins=True)
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predict</th>
      <th>Neutral</th>
      <th>Satisfied</th>
      <th>Unsatisfied</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Satisfaction</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Neutral</th>
      <td>144</td>
      <td>13</td>
      <td>3</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Satisfied</th>
      <td>6</td>
      <td>139</td>
      <td>0</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Unsatisfied</th>
      <td>6</td>
      <td>0</td>
      <td>130</td>
      <td>136</td>
    </tr>
    <tr>
      <th>All</th>
      <td>156</td>
      <td>152</td>
      <td>133</td>
      <td>441</td>
    </tr>
  </tbody>
</table>
</div>




```python
model.score(X_test, y_test)
```




    0.93650793650793651


