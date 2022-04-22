#!/usr/bin/env python
# coding: utf-8

# # Motor Fraud Insurance Detection

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.float_format', lambda x: '%.2f' %x)
sns.set_style("darkgrid")


# In[4]:


df=pd.read_csv("Fraud_Data.csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df["FraudFound_P"].value_counts()


# In[10]:


plt.figure(figsize=(8,4))
df["FraudFound_P"].value_counts().plot(kind="bar",color=['salmon','lightblue'])


# In[11]:


df["Sex"].value_counts()


# In[12]:


plt.figure(figsize=(10,6))
df["Fault"].value_counts().plot(kind='bar')
plt.xticks(rotation=0)


# In[13]:


df.AccidentArea.value_counts()


# In[14]:


plt.figure(figsize=(12,6))
df.AccidentArea.value_counts().plot(kind='bar',color=["salmon","darkred"])
plt.xticks(rotation=0)


# In[15]:


plt.figure(figsize=(12,4))
df.VehicleCategory.value_counts().plot(kind="bar",color=["green","pink","orange"])


# In[16]:


df.AgeOfVehicle.value_counts()


# In[17]:


plt.figure(figsize=(12,6))
df.AgeOfVehicle.value_counts().plot(kind="bar")
plt.xticks(rotation=0);


# In[18]:


df.WitnessPresent.value_counts()


# In[19]:


plt.figure(figsize=(12,7))
df.WitnessPresent.value_counts().plot(kind="bar",color=["pink","green"])


# In[20]:


df.PoliceReportFiled.value_counts()


# In[21]:


plt.figure(figsize=(12,8))
df.PoliceReportFiled.value_counts().plot(kind="bar",color=["Red","yellow"])


# In[22]:


df.DriverRating.value_counts()


# In[23]:


plt.figure(figsize=(12,8))
df.DriverRating.value_counts().plot(kind="bar",color=["violet","pink"])


# In[24]:


df.VehiclePrice.value_counts()


# In[25]:


df.corr()


# In[26]:


plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='plasma_r')


# In[27]:


pd.crosstab(df.FraudFound_P, df.Sex)


# In[28]:


pd.crosstab(df.FraudFound_P, df.Fault).plot(kind="bar",
                                            color = ["salmon","lightblue"],
                                            figsize=(12,7))
plt.xticks(rotation = 0);


# In[29]:


gpd_by_val=df.groupby('Age').agg({'FraudFound_P':'sum'}).reset_index()

fig, (ax1) = plt.subplots(1,1,figsize=(22, 6))
grph =sns.barplot(x='Age', y='FraudFound_P', data = gpd_by_val, ax=ax1)

grph.set_xticklabels(grph.get_xticklabels(),
                    rotation=0,
                    horizontalalignment='right'
                    );


# In[30]:


df.info()


# In[31]:


print(df['Age'].unique()==0)
len(df[df['Age']==0])


# In[32]:


df_temp = df.copy()
# Finding columns which contains strings
for labels, content in df_temp.items():
    if pd.api.types.is_string_dtype(content):
        print(labels)
        


# In[33]:


df_temp.info()


# In[34]:


df_temp.head()


# In[35]:


df_temp.describe()


# In[36]:


x=df_temp.drop("FraudFound_P",axis=1)
y=df_temp["FraudFound_P"]


# In[37]:


x


# In[38]:


y


# In[39]:


# Models from scikit-learn 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2


# Model Evaluation libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, accuracy_score
from sklearn.metrics import plot_roc_curve


# In[40]:


np.random.seed(42)

x_train, x_test, y_train, y_test = train_test_split(x,y,)
       


# In[41]:


from collections import Counter
print(f"Training target statistics: {Counter(y_train)}")
print(f"Testing target statistics: {Counter(y_test)}")
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Target Feature is totally unbalanced so we would apply scikit-learn function to balance the weight of classes

# In[42]:


from sklearn.utils import class_weight
class_weights = dict(zip(np.unique(y_train), class_weight.compute_class_weight(class_weight='balanced',classes= np.unique(y_train), y = y_train)))
class_weights


# In[43]:


# Let's put our models into dictionary 
models = {"Logistic Regression": LogisticRegression(class_weight=class_weights,solver = 'liblinear'),
          "KNN": KNeighborsClassifier(),
          "Random Forest Classifier": RandomForestClassifier(class_weight=class_weights),
          }

# Let's create a function to fit and later score our models
def fit_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates the given machine learning models
    """
    # random seed for reproduction
    np.random.seed(42)
    
    # Let's create a empty dictionary to keep model score
    model_score = {}
    
    # Let's loop through the models dictionary
    for name, model in models.items():
        # Fit the model
        model.fit(X_train, y_train)
        # Evaluate the score and append it
        model_score[name] = model.score(X_test,y_test)
    return model_score
    






