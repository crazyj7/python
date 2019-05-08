#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import datasets
from sklearn import metrics
import types
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="darkgrid", palette="bright", font_scale=1.5)


# In[2]:


df = pd.read_csv("./world-happiness-report/2017.csv")
df.head(60)


# In[3]:


sns.distplot(df['Happiness.Score'])


# In[4]:


corrmat = df.corr()
sns.color_palette("Paired")
sns.heatmap(corrmat, vmax=.8, square=True, cmap="PiYG", center=0)


# In[5]:


data = dict(type = 'choropleth', locations=df['Country'], locationmode='country names', z=df['Happiness.Rank'], 
            text=df['Country'],colorbar={'title':'Happiness'})
layout = dict(title = 'Global Happiness 2017', geo=dict(showframe = False))
choromap3 = go.Figure(data=[data], layout=layout)
iplot(choromap3)


# In[6]:


y = df['Happiness.Score']
X = df.drop(['Happiness.Score', 'Happiness.Rank', 'Country', 'Whisker.high', 'Whisker.low'], axis=1)


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print('Coefficients: \n', lm.coef_)


# In[8]:


predictions = lm.predict(X_test)


# In[9]:


plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[10]:


df_2015 = pd.read_csv("./world-happiness-report/2015.csv")
df_2017 = pd.read_csv("./world-happiness-report/2017.csv").drop(['Whisker.high', 'Whisker.low', 'Happiness.Rank'], axis=1)

df_2015 = df_2015[['Country', 'Region']]
df = pd.merge(df_2015, df_2017, on='Country')
df.head(5)


# In[11]:


corrmat = df.corr()
mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(10, 10))
sns.heatmap(corrmat, annot=True, vmax=.8, square=True, cmap="PiYG", center=0, mask=mask)


# In[12]:


corrmat = df[df['Region'].str.contains('Asia')].corr()
plt.figure(figsize=(10, 10))
ax = plt.axes()
ax.set_title('Asia')
sns.heatmap(corrmat, annot=True, vmax=.8, square=True, cmap="PiYG", center=0, mask=mask)


# In[13]:


corrmat = df[df['Region'].str.contains('Europe')].corr()
plt.figure(figsize=(10, 10))
ax = plt.axes()
ax.set_title('Europe')
sns.heatmap(corrmat, annot=True, vmax=.8, square=True, cmap="PiYG", center=0, mask=mask)


# In[14]:


corrmat = df[df['Region'].str.contains('America')].corr()
plt.figure(figsize=(10, 10))
ax = plt.axes()
ax.set_title('America')
sns.heatmap(corrmat, annot=True, vmax=.8, square=True, cmap="PiYG", center=0, mask=mask)


# In[15]:


corrmat = df[df['Region'].str.contains('Africa')].corr()
plt.figure(figsize=(10, 10))
ax = plt.axes()
ax.set_title('Africa')
sns.heatmap(corrmat, annot=True, vmax=.8, square=True, cmap="PiYG", center=0, mask=mask)


# In[16]:


kcj_df = df[df['Country'].str.contains('Korea|China|Japan')]
kcj_df


# In[17]:


sns.scatterplot(x='Country', y='Happiness.Score', data=kcj_df)


# In[ ]:





# In[ ]:




