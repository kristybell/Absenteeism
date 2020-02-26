#!/usr/bin/env python
# coding: utf-8

# # Creating a Logistic Regression to Predict Absenteeism

# ## Import the Relevant Libraries

# In[1]:


import pandas as pd
import numpy as np


# ## Load the Data

# In[2]:


data_preprocessed = pd.read_csv('Absenteeism_preprocessed.csv')


# In[3]:


data_preprocessed.head()


# The model itself will give us a fair indication of which variables are important for the analysis.
# 
# Logistic Regression is a type of classification.

# ## Create the Targets

# Create two classes:
#    1. Moderately Absent
#    2. Excessively Absent
#   
# Will take the median value of the Absenteeism Time in Hours and use it as a cut-off line.

# In[4]:


data_preprocessed['Absenteeism Time in Hours'].median()


# 1. Less than or Equal to 3.0 --> Moderately Absent 
# 2. Greater than 3.0 --> Excessively Absent
# 
# In ML, 0s and 1s are TARGETS
# 
# Using the median to classify essentially balances the data, thus half of the data fits into classification 1 and the other half into classification 2. Therefore, prevents our model from learning to output only 0s or only 1s.

# In[5]:


# np.where(condition, value if True, value is False) --> checks if a condition has been satisfied and assigns a value accordingly 
targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 
                   data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)
targets


# In[6]:


# create new column in dataframe for Excessive Absenteeism
data_preprocessed['Excessive Absenteeism', 'Daily Work Load Average', 'Distance to Work'] = targets
data_preprocessed.head()


# ## A Comment on the Targets

# To prove our model is not learning to output only 0s and 1s...

# In[7]:


targets.sum() / targets.shape[0]


# Around 46% of the targets are 1s. A 60-40 split will usually work for a logistic regression, but not true for other algorithms such as neural networks. 
# 
# 45-55 is almost always sufficient.

# In[8]:


# Drop Absenteeism Time in Hours
data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours'], axis=1)


# In[9]:


# check that at this point, there is a checkpoint of the data
# Using 'is' --> true = the 2 variables refer to the same object
#                false = the 2 variable refer to different objects
data_with_targets is data_preprocessed


# In[10]:


data_with_targets.head()


# # Select the Inputs for the Regression

# In[11]:


data_with_targets.shape


# In[12]:


# 'DataFrame.iloc[row indices, column indices]' --> selects (slices) data by position when given rows and columns wanted
data_with_targets.iloc[:,0:14]  # all rows and columns 0 through 13; '.loc' is inclusive of the range whereas '.iloc' is exclusive 


# In[13]:


data_with_targets.iloc[:,:-1]    #will give the same results without have to count the number of columns; just want all columns but the last


# In[14]:


unscaled_inputs = data_with_targets.iloc[:,:-1] 


# ## Standardize the Data

# In[15]:


# THE FOLLOWING CODE (WHICH HAS BEEN COMMENTED OUT) IS BAD PRACTICE BECAUSE IT ALSO STANDARDIZES THE DUMMY VARIABLES
#from sklearn.preprocessing import StandardScaler

# declare standard scaler object
#absenteeism_scaler = StandardScaler()       #EMPTY scaler object; no information in it yet
                                           # will be use to subtract the mean and divide by the standard deviation variablewise (featurewise)


# In[16]:


# Use this instead
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator,TransformerMixin):
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# In[17]:


unscaled_inputs.columns.values


# In[18]:


# scale columns except for the (4) dummy variables
#columns_to_scale = ['Month Value',
#       'Day of the Week', 'Transportation Expense', 'Distance to Work',
#       'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
#       'Children', 'Pets']

columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Education']


# In[19]:


# List Comprehension -- a syntactic construct which allos us to create a list from existing lists based on loops, conditionals, etc.
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]


# In[20]:


absenteeism_scaler = CustomScaler(columns_to_scale)


# In[21]:


absenteeism_scaler.fit(unscaled_inputs)


# In[22]:


# scale the inputs
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
scaled_inputs

# new_data_raw = pd.read_csv('new_data.csv')
# new_data_scaled = absenteeism_scaler.transform(new_data_raw)


# In[23]:


scaled_inputs.shape   # print out size


# ## Split the Data into Train & Test, then Shuffle

# ### Import the Relevant Module

# In[24]:


from sklearn.model_selection import train_test_split

#splits arrays or matrices into random train and test subsets


# ### Split

# In[25]:


train_test_split(scaled_inputs, targets)


# In[26]:


# default split shows training size = 75% and test size = 25%
# add parameters 'train _size = 0.8' to change training size = 80%
# 'sklearn.mode_selection.train_test_split(inputs, targets, train_size, shuffle=True, random_state)'
# splits arrays or matrices into random train and test subsets
# rerunning our code, we get a different split every time due to shuffle by default 

x_train, x_test, y_train, y_test = train_test_split(scaled_inputs,targets, train_size = 0.8, random_state = 20)


# In[27]:


print(x_train.shape, y_train.shape)


# In[28]:


print(x_test.shape, y_test.shape)


# # Logistic Regression with Sklearn

# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# ### Training the Model

# In[30]:


# declare Logistic Regression object
reg = LogisticRegression(solver='liblinear')


# In[31]:


reg.fit(x_train, y_train)


# In[32]:


reg.score(x_train, y_train)


# In[33]:


# can conclude the model has an accuracy of about 80%
# also stated as 80% of the model outputs match the targets


# ### Manually Check the Accuracy

# In[34]:


model_outputs = reg.predict(x_train)
model_outputs


# In[35]:


y_train


# In[36]:


# hard to compare with the naked eye
# use code to compare the outputs to the targets
model_outputs == y_train

# True = matches
# False = does not match


# In[37]:


# Boolean: True = 1 and False = 0

np.sum((model_outputs==y_train))


# In[38]:


model_outputs.shape[0]


# In[39]:


np.sum((model_outputs==y_train)) / model_outputs.shape[0]


# ## Finding the Intercept and Coefficients of Linear Regression

# In[40]:


reg.intercept_


# In[41]:


reg.coef_


# In[42]:


# we wnat to know what variables these coeffecients refer to


# In[43]:


unscaled_inputs.columns.values
# 'scaled_inputs.columns.values' will receive an error due to employing sklearn, the results are arrays and not dataframes
# thus must use unscaled_inputs and then create a dataframe


# In[44]:


feature_name = unscaled_inputs.columns.values


# In[45]:


summary_table = pd.DataFrame(columns=['Feature Name'], data = feature_name)

# must transpose the array, because by default, np.arrays are rows and we want columns
summary_table['Coefficient'] = np.transpose(reg.coef_)

summary_table     # prints summary tables of the variables and correlating coefficients


# In[46]:


# add one to all indices of dataframe 'summary_table'
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table


# Standardized Coefficients are the coefficients of a regression where all variables have been standardized

# ### Interpreting the Coefficients

# log(odds) = intercept + (b_1 * x_1) + (b_2 * x_2) +...+ (b_n * x_n)

# In[47]:


summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)
summary_table.head()


# In[48]:


# 'DataFrame.sort_values(Series)' --> sorts the values in a data frame with respect to a given column (series)
summary_table.sort_values('Odds_ratio', ascending=False)


# A Feature is not particulary important if..
#  - coefficient is close to '0':
#          - implies that no matter the feature value, we will multiply is by 0 (in the modela)
#          
#  - coefficient is close to '1':
#          - for a unit change in the standardized fature, the odds increase by a multiple equal to the odds ratio (1 = no change)
#          - i.e.   odds * odds ratio = new odds
#                   5:1  *      2     = 10:1
#                   5:1  *     0.2    = 1:1
#                   5:1  *      1     = 5:1 

# Interpreting the Summary Table:
#  - the variables 'Daily Work Load Average', 'Distance to Work', and 'Day of the Week' may be dropped due to their coefficients being close to 0
#  - When employees give Reasons 1 through 4, there seems to be a likelihood of future absenteeism

# BACKWARD ELIMINATION
#  - The idea is that we can simplify our model by removing all features which have close to no contribution to the model
#  - When we ahve the p-values, we get rid of all coeff with p-values > 0.05
#  - if the weight is small enough, it won't make a difference; if these variables are removed, the rest of the model should not really change in terms of coefficient values

# # Testing the Model

# In[49]:


reg.score(x_test, y_test)   #reg.score(train, test)


# Based on data the model has NEVER seen before, in 73.6% of the cases, the model will predict (correctly) if a person is going to be excessively absent
# 
# Often the test accuracy is 10-20% lower than the train accuract (due to overfitting)

# In[51]:


# 'sklearn.linear_model.LogisticRegression.predict_proba(x)' -- returns the probability estimates for all possible outputs (classes)
predicted_proba = reg.predict_proba(x_test)
predicted_proba  # first column: probability of being 0, second column: prob of being 1


# In[52]:


predicted_proba.shape


# In[53]:


predicted_proba[:,1]

# if the probability is below 0.5, it places a 0 and vice versa, a 1


# # Save the Model

# 'pickle[module]' -- is a Python module used to convert a Python object into a character stream

# In[54]:


import pickle


# - file name: model
# - write bytes: wb
# - dump = save
# - reg = object to be dumped

# In[55]:


with open('model', 'wb') as file:
    pickle.dump(reg, file)


# # A Note on Pickling
#  
# 
# There are several popular ways to save (and finalize) a model. To name some, you can use Joblib (a part of the SciPy ecosystem), and JSON. Certainly, each of those choices has its pros and cons. Pickle is probably the most intuitive and definitely our preferred choice.
# 
# Once again, ‘pickle’ is the standard Python tool for serialization and deserialization. In simple words, pickling means: converting a Python object (no matter what) into a string of characters. Logically, unpickling is about converting a string of characters (that has been pickled) into a Python object.
# 
# 
# 
# There are some potential issues you should be aware of, though!
# 
# Pickle and Python version.
# 
# Pickling is strictly related to Python version. It is not recommended to (de)serialize objects across different Python versions. Logically, if you’re working on your own this will never be an issue (unless you upgrade/downgrade your Python version). 
# 
# 
# 
# Pickle is slow.
# 
# Well, you will barely notice that but for complex structures it may take loads of time to pickle and unpickle.
# 
# 
# 
# Pickle is not secure.
# 
# This is evident from the documentation of pickle, quote: “Never unpickle data received from an untrusted or unauthenticated source.” The reason is that just about anything can be pickled, so you can easily unpickle malicious code.
# 
# 
# 
# Now, if you are unpickling your own code, you are more or less safe.
# 
# 
# 
# If, however, you receive pickled objects from someone you don’t fully trust, you should be very cautious. That’s how viruses affect your operating system.
# 
# 
# 
# Finally, even your own file may be changed by an attacker. Thus, the next time you unpickle, you can unpickle just about anything (that this unethical person put there)

# In[ ]:




