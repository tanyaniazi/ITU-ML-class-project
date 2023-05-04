import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import Telco dataset and store it in a dataframe
df = pd.read_csv("/Users/raha/Downloads/Telco(2).csv")


#dropping the unnecessary columns 
df = df.drop('customerID', axis = 1)

#coding some columns that have a range of only a couple of values
df = df.replace({'gender': {'Female': 1, 'Male': 0}})
df = df.replace({'Partner': {'Yes': 1, 'No': 0}})
df = df.replace({'Dependents': {'Yes': 1, 'No': 0}})
df = df.replace({'PhoneService': {'Yes': 1, 'No': 0}})
df = df.replace({'PaperlessBilling': {'Yes': 1, 'No': 0}})
df = df.replace({'Churn': {'Yes': 1, 'No': 0}})

#Then, for non-binary categorical columns, we use the `get_dummies` function to sort them in new columns based on their category.
df = pd.get_dummies(df, columns=['MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'], drop_first=False)



'''
**Data Cleaning**
After converting the categorical data to numeric, we found out that there are problems in the data. <br/>
In the `TotalCharges` column, there are some rows that have `' '` value (space), which is not detectable via `isNa` or `isNaN` method.</br>
So we had to manually find those and deal with them. 

There are 3 methods to deal with the empty values.

<ol type='I' >
1. Deleting the Empty Values
df.drop(value for value in list_null_values)


2. Fill the MANUALLY with mean of other data
non_null_df = df.drop(value for value in list_null_values)
mean = non_null_df['TotalCharges'].mean()

for i in list_null_values:
  df['TotalCharges'].iloc[i] = mean

3. Fill them AUTOMATICALLY with the mean of data from their respective calss.

Before dealing with NaN data, the data type of the `TotalCharges` column needs to be changed to `float`, since they are `str`.
In the following cell, first a list is extracted, where data with no values exits. 
Then, if there is any values, they are converted to `float`, since they are stored in `str` format. Otherwise, I add them to our list, where data with no values are being stored.
Other methods to achieve the same result:
data = data.replace('^\s*$', np.nan, regex = True)
data.dropna(axis = 0 ,inplace = True)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])

P.S. The reason I didn't use the upper code, is because I needed to keep track of the nan data, instead of just simply dropping them. Therefore I had to use my own method

'''
list_null_values = []
for i in range(df.shape[0]):
  if df['TotalCharges'].iloc[i] == ' ':
    list_null_values.append(i)
  else:
    df['TotalCharges'].iloc[i] = float(df['TotalCharges'].iloc[i])

#use the third method to automatically fill the data with no values, with the mean of the same column of their respective class.
non_null_df = df.drop(value for value in list_null_values)
classes = list(df.Churn.unique())
classes_mean = {}

#This for loop is used for getting the mean of each class
for i in classes:
  sum = non_null_df['TotalCharges'].where(non_null_df['Churn'] == classes[i]).dropna()
  classes_mean[i] = sum.mean()

#This for loop is used for filling each missing data with the mean of it's class (label)
for i in list_null_values:
  label = df['Churn'].iloc[i]
  df['TotalCharges'].iloc[i] = classes_mean.get(label)

#save the cleaned dataframe as a csv file
df.to_csv('/Users/Tanya/cleaned_Telco.csv')
