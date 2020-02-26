#!/usr/bin/env python
# coding: utf-8

# A good business analyist must:
# - be able to manage information
# - have a substantial amount of mathematical and statistical tools
# - present results in the most intuitive way
# - be able to tackle the problem from a business prospective
# 
# This lesson:
#  1. Focus on the business problem
#  2. Look into the dataset.
#  3. Apply logistic regression
#  4. Create a dataviz
#  - apply business logic, intuition and interprestation
#  
# ABSENTEEISM: 
# - Absence from work during normal working hours, resulting in temporary incapacity to execute regular working activity
# 
# Base on what information should we preduct whether an employee is expected to be absent or not?
# 
# How would we measure absenteeism?
# 
# Purpose of the business exercise:
# - Explore whether a person presenting certain characteristics is expected to be away from work at some point in time or not
# 
# - We want to know for how many working hours an employee could be away from work
#  a. How far away they live from work?
#  b. How many children and pets they have?
#  c. Do they have higher education?

# In[1]:


import pandas as pd

# '.read_csv()' assigns the information from the initial *.csv file to this variable
raw_csv_data = pd.read_csv("Absenteeism_data.csv")
raw_csv_data


# Data can be defined in 2 ways:
#  1. Primary
#      - data YOU have created (i.e. survey, questionnaire, sales operations, changing inventoru, accounting ledger data/business intelligence etc.) 
#  2. Secondary (aka "raw data")
#      - an already existing dataset that somebody else has create, NOT YOU (i.e. free data from website, paid data from an organization, etc.)
#  
#  - Primary and secondary sources of data are relative terms
#  
# DATA PREPROCESSING:
# A group of operations that will convert your raw data (the data you have been given) into a format that is easier to understand and, hence, useful for further processing and analysis
#     - fix the problems that can inevitable occur with data gathering
#     - organize your information in a suitable and practical way before doing analysis and making predictions
#     
# Pandas = PANel DAta; allows us to work with panel data   
#   - possesses various tools for handling data in a tabular format (a DataFrame)
# 
# You must ALWAYS MAKE A COPY of your initial dataset
#   Why is that necessary?
#     - When you start manipulating the DataFrame, the changes you make will be applied to your original dataset
#     - Using a copy is playing on the safe side, making sure the initial DataFrame won't be modified
#   

# In[2]:


# 'df' -> the naming convention for Data Frames in Python
df = raw_csv_data.copy()


# In[3]:


df # check to make sure the data is a copy of the raw dataframe
# can make edits to this dataset and if make a mistake, can use the raw data again to copy to df


# In[4]:


pd.options.display.max_columns = None  # no limit to maximum value of columns to display
pd.options.display.max_rows = None


# In[5]:


display(df)


# In[6]:


# 'df.info()' -> prints a concise summary of the dataframe
df.info()  # display the number of columns and name of columns in a dataframe


# MATHEMATICS
# 
# - Variable: a symbol or a letter, that stands for a number we don't know yet (x,y)
# 
# - Vector = [3, 13] 
# - Matrix = [1 15, 4 20]
# 
# 
# DATA ANALYTICS (& ECONOMICS)
# 
# - Variable : a characteristic, or a quantit, that may change its value over time under different circumstances
# Variable = Feature = Attribute = Input
# 
# - Data Table
# 
# 
# PROGRAMMING
# 
# - Variable: acts like a storage location which contains a certain amount of information
# 
# - Array (Python): array{[3,13]); array([[1, 15], [4, 20]])
# 
# - DataFrame

# Individual Identification: 
#  - indicates precisely who has been away during work hours
#  
# Label Variable:
#  - a number that is there to distinguish the individuals from one another, not to carry any numeric information (nominal data) 

# ## Drop 'ID':

# In[7]:


# '.drop()' removes specified rows or columns; temporary output
# axis = 0 (default)
df.drop(['ID'], axis = 1)  # axis = 1 is the horizontal axis


# In[8]:


# check that the column is still permanently in the dataframe
df 

# to permanently change dataframe, assign to variable df
df = df.drop(['ID'], axis = 1)
df


# # 'Extract for Absence':

# In[9]:


# extract 'Reason for Absence' from dataframe
df['Reason for Absence']

# indexes: designate the order in which elements appear; starting from 0 naturally


# In[10]:


df_no_age = df.drop(['Age'], axis = 1)
df_no_age


# In[11]:


# '.min()' --> returns the lowest value
df['Reason for Absence'].min()


# In[12]:


# '.max()' --> returns the highest value
df['Reason for Absence'].max()


# In[13]:


# extract a list containing distinct values only
pd.unique(df['Reason for Absence'])


# In[14]:


# 'unique()' --> extracts distinct values only
df['Reason for Absence'].unique()


# In[15]:


# dtype = in64 --> 64-bit integers


# In[16]:


# 'len()' --> returns the number of elements in an object
len(df['Reason for Absence'].unique())


# Recall, the min value was 0 and max value was 28, thus a total of 29 values. A number is missing between 0 and 28.

# In[17]:


# 'sorted()' --> returns a new, sorted list from the items in its argument
sorted(df['Reason for Absence'].unique())


# Reviewing the list, you can spot that the number 20 is missing.

# Attention to detail induces you to perform checks that will only solidify the inferences made throughout your analysis later.

# Database Theory: 
#  - using less characters will shrink the volume of our dataset
#  
# Quantitative Analysis:
#  - add numeric meaning to our categorical nominal values
#  
# Dummy Variables:
#  - an explanatory binary variable that equals...
#      1 --> if a certain categorical effect is present, and that equals
#      0 --> if that same effect is absent     
#  - '.get_dummies()' --> converts categorical variable into dummy variables

# ## .get_dummies()

# In[18]:


reason_columns = pd.get_dummies(df['Reason for Absence'])


# In[19]:


reason_columns


# In[20]:


# check that rows are not missing values
# 0 --> missing value
# 1 --> single value
# 2, 3, 4... --> value is in more than one row
reason_columns['check'] = reason_columns.sum(axis=1)
reason_columns


# In[21]:


reason_columns['check'].sum(axis = 0)


# In[22]:


reason_columns['check'].unique()


# In[23]:


# remove 'check' column
reason_columns = reason_columns.drop(['check'], axis = 1)
reason_columns


# In[24]:


age_dummies =  pd.get_dummies(df['Age'])
age_dummies


# Drop reason 0 to avoid potential multicollinearity issues.

# In[25]:


# 'drop_first = True' --> remove first columns
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
reason_columns


# ## Group the Reasons for Absence:

# In[26]:


df.columns.values


# In[27]:


reason_columns.columns.values


# Group these variables; re-organizing a certain type of variables into groups in a regression analysis
# 
# Group = Class
# 
# - Reasons 1 through 14 --> Group 1 (Related to Various Diseases)
# - Reasons 15 through 17 --> Group 2 (Related to Pregnancy/Giving Birth)
# - Reason 18 through 21 --> Group 3 (Related to Poisoning)
# - Reason 22 through 28 --> Group 4 (Related to Medical Visit)
# 
# After splitting this object into smaller pieces, each piece itself will be a DataFrame object as well.

# In[28]:


# remove 'Reason for Absence' column to avoid multicollinearity
df = df.drop(['Reason for Absence'], axis = 1)
df


# In[29]:


# '.loc[]' --> retrieves rows and columns avaliable
reason_columns.loc[:, 1:14]

# retrieves all rows and all columns from 1 through 14 inclusive


# In[30]:


# create variables for the 4 groups
reason_type_1 = reason_columns.loc[:,1:14].max(axis=1)
reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)


# In[31]:


reason_type_1


# ## Cocatenate Column Values

# In[32]:


df


# In[33]:


# 'pd.concat()' --> concatenate
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)
df


# We would opt to assign column names in a more meaningful way than "0, 1, 2, 3".

# In[34]:


df.columns.values   # list the column titles


# In[35]:


column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education','Children', 'Pets', 'Absenteeism Time in Hours', 'Reason 1', 'Reason 2', 'Reason 3', 'Reason 4']


# In[36]:


df.columns = column_names


# In[37]:


# check df
# '.head()' --> displays the top five rows of our data table, together with the relevant column names
df.head()


# In[38]:


# concatenate the 'df_no_age' and 'age_dummies' objects you previously obtained. Store the result in a new object called 'df_concatenated'
df_concatenated = pd.concat([df_no_age, age_dummies], axis = 1)
df_concatenated


# In[39]:


df.columns.values   # list the column titles


# ## Reorder Columns

# In[40]:


column_names_reordered = ['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']


# In[41]:


# add the reordered column names to the dataframe
df = df[column_names_reordered]

# show the top 5 rows only
df.head() 


# In[42]:


df_concatenated.columns.values   # list the column titles


# In[43]:


# reorder the columns from 'df_concatenated' in such a way that 'Absenteeism Time in Hours' column appears at the far right of the data set
column_names = ['Reason for Absence', 'Date', 'Transportation Expense',
       'Distance to Work', 'Daily Work Load Average', 'Body Mass Index',
       'Education', 'Children', 'Pets', 'Absenteeism Time in Hours', 27,
       28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 43, 46, 47, 48,
       49, 50, 58]
df_concatenated = df_concatenated[column_names]
df_concatenated


# ## Create a Checkpoint

# Checkpoint --> an interim save of your work
#  - create a copy of the current state of the df DataFrame

# In[44]:


# Create a temporary save of your work so that you reduce the risk of losing important data at a later stage
df_reason_mod = df.copy()


# In[45]:


df_reason_mod


# In programming in general, and in Jupyter in particular, creating checkpoints refers to storing the current version of your code, not really the content of a variable

# In[46]:


# create a checkpoint of your work on the exercises, storing the current output in a variable called 'df_checkpoint'
df_checkpoint = df_concatenated.copy()
df_checkpoint


# ## 'Date':

# In[47]:


type(df_reason_mod['Date'][0])   # 'type' --> to determine if column consists of strings or values of another kind
# add '[0]' behind '['Date']' to change it to a string
# string - data is saved as text
# in 1 column, or in 1 series, we can have values of a single data type only!


# In[48]:


# 'timestamp' --> a classical data type found in many programming languages out there, used for values representing data and time
# 'pd.to_datetime()' --> converts values into timestamp
df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'])


# In[49]:


# when doing this conversion, must always specify the proper formate of the date values you will be working on
# the following syntax will cause mistakes in the displaye dates
df_reason_mod['Date']


# %d = day
# %m = month
# %Y = year
# %H = hour
# %M = minute
# %S = second

# In[50]:


# ', format' --> 'string' allows you to take control over how Python will read the current dates, so that it can accurately understand which numbers refer to days, months, years, hours, minutes, or seconds
# 'string' will NOT designate the format of the timestamps you are about to create
df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format = '%d/%m/%Y')    # display as date-month-year
df_reason_mod['Date']
#rerun the code at the checkpoint to use the dataframe copy


# In[51]:


type(df_reason_mod['Date'][0])


# In[52]:


df_reason_mod.info()


# ## Extract the Month Value:

# In[53]:


df_reason_mod['Date'][0]  # display timestamp at row 0


# In[54]:


df_reason_mod['Date'][0].month  #extract month only at row 0


# In[55]:


# assign an empty list
list_months = []
list_months


# In[56]:


# extract the month value for each row through a 'for-loop'
# '.append()' --> attaches the new value obtained from each iteration to the existing content of the designated list
for i in range(df_reason_mod.shape[0]):
    list_months.append(df_reason_mod['Date'][i].month)


# In[57]:


list_months


# In[58]:


len(list_months)


# In[59]:


df_reason_mod['Month Value'] = list_months


# In[60]:


df_reason_mod.head(20)


# ## Extract the Day of the Week:

# Monday: 0 
# Tuesday: 1
# Wednesday: 2
# Thursday: 3
# Friday: 4
# Saturday: 5
# Sunday: 6
# 
# Drawback of using modules: 'the rules of the game' are set by the person who created them

# In[61]:


# 'weekday()' --> returns an integer corresponding to the day of the week
df_reason_mod['Date'][699].weekday()  # display what day of the week was at this row


# In[62]:


df_reason_mod['Date'][699]


# To apply a certain type of modification iterativel on each value from a Series or a column in a DataFrame, it is a great idea to create a function that can execute this operation for one element, and then implement it to all values from the column of interest.

# In[63]:


# define a function to directly return to weekday for a row
def date_to_weekday(date_value):
    return date_value.weekday()


# In[64]:


# create a new column for Day of the Week calling the function to apply
df_reason_mod['Day of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)


# In[65]:


df_reason_mod.head()    # view the first 5 rows of the modified dataframe


# In[66]:


# drop the 'Date' column from the 'df_reason_mod' dataframe
df_reason_mod = df_reason_mod.drop(['Date'], axis = 1)
df_reason_mod


# In[67]:


# re-order the columns in 'df_reason_mod' so that "Month Value" and "Day of the Week" appear exactly where "Date" used to be. That is between "Reason_4" and "Transportation Expense"
df_reason_mod.columns.values


# In[68]:


df_reason_mod_reordered = ['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Month Value',
       'Day of the Week', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']
df_reason_mod = df_reason_mod[df_reason_mod_reordered]
df_reason_mod


# ## 'Education' , 'Children', 'Pets'

# 1 = High School  2 = Graduate  3 = Post-Graduate   4 = Master or Doctor in Scientific Field

# In[69]:


# '.unique()' --> extracts distinct values only
df_reason_mod['Education'].unique()


# In[70]:


df_reason_mod['Education'].value_counts()


# In[71]:


# using Boolean, note that 0=no advanced degress 1 = advanced degree
df_reason_mod['Education'] = df_reason_mod['Education'].map({1:0, 2:1, 3:1, 4:1})


# In[73]:


df_reason_mod['Education'].unique()


# In[74]:


# count how many entries have advanced degrees and how many do not
df_reason_mod['Education'].value_counts()


# # Final Checkpoint

# In[76]:


df_preprocessed = df_reason_mod.copy()
df_preprocessed.head(10)


# ## Export Data as a *.csv File

# In[78]:


df_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)


# In[ ]:




