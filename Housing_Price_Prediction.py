#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# #  Real Estate - Price Predictor

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


housing = pd.read_csv('data.csv')


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


# Plot Histogram
# housing.hist(bins = 50, figsize = (20, 15))
# plt.show()


# # Train - Test Splitting

# In[9]:


# For learning purpose
# import numpy as np


# In[10]:


# def split_train_test(data, test_ratio) :
#    np.random.seed(42)
#    shuffled = np.random.permutation(len(data))
#    test_set_size = int(len(data)*test_ratio)
#    test_indices = shuffled[:test_set_size]
#    train_indices = shuffled[test_set_size:]
#    return data.iloc[train_indices], data.iloc[test_indices]


# In[11]:


# train_set, test_set = split_train_test(housing, 0.2)


# In[12]:


# print("Train :", len(train_set))
# print("Test :", len(test_set))


# In[13]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# In[14]:


print("Train :", len(train_set))
print("Test :", len(test_set))


# # Stratified Shuffle for 'CHAS' feature

# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[16]:


strat_train_set['CHAS'].value_counts()


# In[17]:


strat_test_set['CHAS'].value_counts()


# In[18]:


# 376/28, 95/7


# In[19]:


housing = strat_train_set.copy()


# # Corelation matrix lies between -1 to +1

# In[20]:


corr_matrix = housing.corr()
corr_matrix ['MEDV'].sort_values(ascending = False)


# In[ ]:





# In[21]:


from pandas.plotting import scatter_matrix
attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(housing[attributes], figsize = (12, 8))
plt.show()


# In[22]:


housing.plot(kind = 'scatter', x = 'RM', y = 'MEDV', alpha = 0.9)
plt.show()


# # Trying out new attribute combination

# In[23]:


housing['TAXRM'] = housing['TAX']/housing['RM']
housing.head()


# In[ ]:





# In[24]:


corr_matrix = housing.corr()
corr_matrix ['MEDV'].sort_values(ascending = False)


# In[25]:


housing.plot(kind = 'scatter', x = 'TAXRM', y = 'MEDV', alpha = 0.9)
plt.show()


# In[26]:


housing = strat_train_set.drop('MEDV', axis = 1)
housing_labels = strat_train_set['MEDV'].copy()   # Make the features and labels separate


# # Missing Attribute

# In[27]:


# To take care of missing attribute
#   1. Get rid of missing points
#   2. Get rid of whole attribute
#   3. Set the value to some value(0, mean or median)


# In[28]:


a = housing.dropna(subset = ['RM']) # option 1
a.shape
# Housing dataframe remains unchanged


# In[29]:


housing.drop('RM', axis = 1).shape # option 2
# There is no RM column and original housing dataframe remains unchanged


# In[30]:


median = housing['RM'].median() # compute median for option 3


# In[31]:


housing['RM'].fillna(median) # option 3
# Housing dataframe remains unchanged


# In[32]:


housing.describe() # Before filling the null value


# In[33]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
imputer.fit(housing)
# Calculates median value of all the features and fits it in the missing values


# In[34]:


x = imputer.transform(housing)


# In[35]:


housing_tr = pd.DataFrame(x, columns = housing.columns)


# In[36]:


housing_tr.describe()


# # Scikit-learn Design

# 1.Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method.
#   Fit method - Fits the dataset and calculates internal parameters
# 2.Transformers - transform method takes input and returns output based on the learnings from fit(). It also has a convenience     function called fit_transform() which fits and then transforms.    
# 3.Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions.
#   It also gives score() function which will evaluate the predictions.
#   
#     

# # Feature Scaling

# Min-max scaling (Normalization) (value - min)/(max - min) Sklearn provides a class called MinMaxScaler for this
# 
# Standardization (value - mean)/std Sklearn provides a class called StandardScaler for this

# In[37]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    #    ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler())
])


# In[38]:


housing_num_tr = my_pipeline.fit_transform(housing)


# # Selecting desired model

# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor() 
model.fit(housing_num_tr, housing_labels)


# In[41]:


some_data = housing.iloc[:5]


# In[42]:


some_labels = housing_labels.iloc[:5]


# In[43]:


prepared_data = my_pipeline.transform(some_data)


# In[44]:


model.predict(prepared_data)


# In[45]:


list(some_labels)


# # Evaluating the model

# In[46]:


from sklearn.metrics import mean_squared_error
housing_prediction = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_prediction)
rmse = np.sqrt(mse)


# In[47]:


mse


# # Using better evaluation technique - Cross Validation

# In[48]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)
rmse_scores = np.sqrt(-scores)


# In[49]:


rmse_scores


# In[50]:


def print_scores(scores):
    print('Scores :', scores)
    print('Mean :', scores.mean())
    print('Steandard deviation :', scores.std())


# In[51]:


print_scores(rmse_scores)


# # Saving the model

# In[53]:


from joblib import dump, load
dump(model, 'Nachi.joblib')


# # Testing the model on test data

# In[54]:


X_test = strat_test_set.drop('MEDV', axis = 1)
Y_test = strat_test_set['MEDV'].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_prediction = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_prediction)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[55]:


prepared_data[0]


# In[56]:


from joblib import dump, load
import numpy as np
model = load('Nachi.joblib')


# In[57]:


features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.23979304, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  11.41164221, -0.86091034]])
model.predict(features)


# In[ ]:




