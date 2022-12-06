#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import probplot


# In[4]:


df = pd.read_excel('DS - Assignment Part 1 data set.xlsx')
df.head()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


df.rename({'House size (sqft)': 'Area (sqft)', 'House price of unit area': 'Price', 'House Age': 'Age'},axis=1,inplace=True)


# In[9]:


df.describe()


# In[10]:


#EDA
for col in df.columns:
    if col != 'Transaction date':
        print(f"Skewness of {col}:",df[col].skew())
        print(f"Kurtosis of {col}:",df[col].kurtosis())
        sns.distplot(df[col])
        plt.title(f'Distribution Plot of {col}')
        plt.show()
        sns.boxplot(df[col])
        plt.title(f'Box Plot of {col}')
        plt.show()
        probplot(df[col],plot=plt,rvalue=True)
        plt.title(f'Probability Plot of {col}')
        plt.show();


# In[11]:


df.corr()['Price'].sort_values(ascending=False)[1:]


# In[12]:


sns.pairplot(df,palette='viridis')


# In[13]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True,cmap='Set2',vmin=-1,vmax=1)
plt.tight_layout();


# In[14]:


sns.clustermap(df)


# In[15]:


#creating a copy
df_copy = df.copy()
df_copy.head()


# In[16]:


scaler = StandardScaler()
features = df_copy.columns
df_copy = scaler.fit_transform(df_copy)
df_copy = pd.DataFrame(df_copy,columns=features)
df_copy.head()


# In[17]:


#Splitting the dataset
X = df.drop('Price',axis=1)
y = df.Price


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101,shuffle=True)


# In[19]:


models = []
scores = []


# In[20]:


def fit_and_test_model(model):
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    print("RMSE:",np.sqrt(mean_squared_error(y_test,pred)))
    print("R2 Score:",r2_score(y_test,pred))
    models.append(str(model).split('(')[0])
    scores.append(r2_score(y_test,pred))


# In[21]:


#Linear regression
fit_and_test_model(LinearRegression())


# In[22]:


#KNeighborsRegressor
fit_and_test_model(KNeighborsRegressor())


# In[23]:


#SVR
fit_and_test_model(SVR())


# In[24]:


#GradientBoostingRegressor
fit_and_test_model(GradientBoostingRegressor())


# In[25]:


#BaggingRegressor
fit_and_test_model(BaggingRegressor())


# In[26]:


#HistGradientBoostingRegressor
fit_and_test_model(HistGradientBoostingRegressor())


# In[27]:


#ExtraTreesRegeressor
fit_and_test_model(ExtraTreesRegressor())


# In[28]:


#DecisionTreeRegressor
fit_and_test_model(DecisionTreeRegressor())


# In[29]:


#RandomForestRegressor
fit_and_test_model(RandomForestRegressor())


# In[30]:


#XGBRegressor
fit_and_test_model(XGBRegressor())


# In[32]:


#MlPRegessor
fit_and_test_model(MLPRegressor())


# In[34]:


#Comparison
model_performances = pd.DataFrame([models,scores]).T
model_performances.columns = ['Model','R2 Score']
model_performances.set_index('Model',inplace=True)
model_performances = model_performances.sort_values('R2 Score',ascending=False)
model_performances


# In[ ]:


#the model which exhibits the best performance is Extra Trees Regressor as its accuracy score is 79%

