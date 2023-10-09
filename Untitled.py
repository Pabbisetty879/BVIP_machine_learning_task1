#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector,make_column_transformer
from sklearn.pipeline import make_pipeline
from scipy import stats
from scipy.stats import f_oneway
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import warnings
import locale


# In[9]:


df = pd.read_csv('data.csv')

df.head()


# In[10]:


df.info()


# In[11]:


print("Total number of house sales records in the dataset:", len(df))


# In[12]:


null_counts = df.isnull().sum()

print("Number of null values in each column:")
print(null_counts)


# In[13]:


unique_counts = df.nunique()

print("\nNumber of unique values in each column:")
print(unique_counts)


# In[14]:


df.describe()


# In[ ]:





# In[17]:


sns.boxplot(x='city', y='bedrooms', data=df)
plt.title('Boxplot of Bedrooms by City')
plt.xticks(rotation=90)
plt.show()


# In[18]:


sns.boxplot(x='condition', y='bathrooms', data=df)
plt.title('Boxplot of Bathrooms by Condition')
plt.xticks(rotation=90)
plt.show()


# In[19]:


anova_results = pd.DataFrame(columns=['Feature', 'p-value', 'Significant'])

numeric_features = df.select_dtypes(include=['number']).columns.tolist()

numeric_features.remove('price')

for feature in numeric_features:
    groups = [df[df[feature] == value]['price'] for value in df[feature].unique()]
    
    f_statistic, p_value = f_oneway(*groups)
    
    significant = "Yes" if p_value < 0.05 else "No"
    
    result = {'Feature': feature, 'p-value': p_value, 'Significant': significant}
    
    anova_results = pd.concat([anova_results, pd.DataFrame([result])], ignore_index=True)
    
    anova_results['p-value'] = anova_results['p-value'].astype(float)
    anova_results['p-value'] = anova_results['p-value'].apply(lambda x: "{:.2e}".format(x))

print(anova_results)


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns

significant_features_df = anova_results[anova_results['Significant'] == 'Yes']

significant_features = significant_features_df['Feature']

num_features = len(significant_features)
num_cols = 3 
num_rows = (num_features + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

if num_rows == 1:
    axes = axes.reshape(1, -1)

fig.text(0.04, 0.5, 'Price', va='center', rotation='vertical')

for i, feature in enumerate(significant_features):
    row_idx = i // num_cols
    col_idx = i % num_cols
    ax = axes[row_idx, col_idx]
    sns.scatterplot(x=df[feature], y=df['price'], ax=ax)
    ax.set_title(f'{feature} vs. Price')
    ax.set_xlabel(feature)

for i in range(num_features, num_rows * num_cols):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout(pad=2.0)

plt.show()


# In[21]:


df["date"] = pd.to_datetime(df["date"])

grouped_dates = df.groupby(df["date"].dt.month)
grouped_condition = df.groupby(df["condition"])

dates_per_group = grouped_dates.size()
condition_per_group = grouped_condition.size()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].pie(dates_per_group, labels=dates_per_group.index, autopct="%1.1f%%")
axes[0].legend(loc="upper left")
axes[0].set_title("Distribution of House Sales by Month")

axes[1].pie(condition_per_group, labels=condition_per_group.index, autopct="%1.1f%%")
axes[1].legend(loc="upper left")
axes[1].set_title("Distribution of Houses based on Condition")
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




