import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from statsmodels.formula.api import ols

df= pd.read_csv('data/brain_size.csv', sep=';', na_values=".")
df.shape

# Fill in the missing values for Height for plotting
df['Height'].fillna(method='pad', inplace=True)

#Box plots for each column for each gender
groupby_gender = df.groupby('Gender')
groupby_gender.boxplot(column=['FSIQ', 'VIQ', 'PIQ'])

#scatter matrices
scatter_matrix(df[['Weight', 'Height', 'MRI_Count']])
scatter_matrix(df[['PIQ', 'VIQ', 'FSIQ']])
scatter_matrix(df[['VIQ', 'MRI_Count', 'Height']], c=(df['Gender'] == 'Female'), marker='o', alpha=1, cmap='winter')
fig = plt.gcf()
fig.suptitle("blue: male, green: female", size=13)

df.plot()
plt.show()

#1-sample t-test
stats.ttest_1samp(df['VIQ'], 0)

#2-sample t-test
female_viq = df[df['Gender'] == 'Female']['VIQ']
male_viq = df[df['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)   

#Paired Tests
stats.ttest_ind(df['FSIQ'], df['PIQ']) 
stats.ttest_rel(df['FSIQ'], df['PIQ'])
stats.ttest_1samp(df['FSIQ'] - df['PIQ'], 0) 

#statsmodels
model = ols('VIQ ~ Gender + MRI_Count + Height', df).fit()
print(model.summary())

