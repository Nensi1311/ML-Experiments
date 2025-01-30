#!/usr/bin/env python
# coding: utf-8

# # Data Loading

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("ObesityDataSet.csv")
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.columns


# In[8]:


df.describe()


# In[9]:


df.describe(include="object")


# In[10]:


df.isnull().sum()


# In[11]:


df["Gender"].unique()


# In[12]:


df["family_history_with_overweight"].unique()


# In[13]:


df["FAVC"].unique()


# In[14]:


df["CAEC"].unique()


# In[15]:


df["SMOKE"].unique()


# In[16]:


df["SCC"].unique()


# In[17]:


df["CALC"].unique()


# In[18]:


df["MTRANS"].unique()


# In[19]:


df["NObeyesdad"].unique()


# # Label Encoding

# In[20]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[21]:


df["Gender"] = encoder.fit_transform(df["Gender"])
df["family_history_with_overweight"] = encoder.fit_transform(df["family_history_with_overweight"])
df["FAVC"] = encoder.fit_transform(df["FAVC"])
df["CAEC"] = encoder.fit_transform(df["CAEC"])
df["SMOKE"] = encoder.fit_transform(df["SMOKE"])
df["SCC"] = encoder.fit_transform(df["SCC"])
df["CALC"] = encoder.fit_transform(df["CALC"])
df["MTRANS"] = encoder.fit_transform(df["MTRANS"])
df["NObeyesdad"] = encoder.fit_transform(df["NObeyesdad"])


# In[22]:


df.head()


# In[23]:


df.info()


# # Standard Scaler

# In[24]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()


# In[25]:


scaler.fit_transform(df)


# # Data Split - train, test, valid

# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X = df.drop(columns=["NObeyesdad"])
Y = df["NObeyesdad"]


# In[28]:


X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=0.7)


# In[29]:


X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5)


# # Model Training

# ## 1. Random Forest

# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[31]:


RF = RandomForestClassifier(n_estimators=100, random_state=42) 


# In[32]:


RF.fit(X_train, Y_train)


# In[33]:


Y_pred_rf = RF.predict(X_valid)


# In[34]:


print(classification_report(Y_valid, Y_pred_rf))


# ## 2. SVM

# In[70]:


from sklearn.svm import SVC 


# In[71]:


SVM = SVC(kernel="linear")


# In[55]:


SVM.fit(X_train, Y_train)


# In[56]:


Y_pred_svm = SVM.predict(X_valid)


# In[57]:


print(classification_report(Y_valid, Y_pred_svm))


# ### Grid Search

# In[68]:


from sklearn.model_selection import GridSearchCV 


# In[72]:


param_grid = {'C': [0.1, 1, 10, 100, 1000], 
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
            'kernel': ['linear']} 

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 


# In[76]:


grid.fit(X_train, Y_train) 


# In[77]:


print(grid.best_params_) 


# In[78]:


print(grid.best_estimator_) 


# In[79]:


Y_pred_grid = grid.predict(X_valid)


# In[80]:


print(classification_report(Y_valid, Y_pred_grid))


# # 3. Extra Tree Classifier

# In[58]:


from sklearn.ensemble import ExtraTreesClassifier


# In[59]:


ETC = ExtraTreesClassifier(n_estimators=100)


# In[60]:


ETC.fit(X_train, Y_train)


# In[61]:


Y_pred_etc = ETC.predict(X_valid)


# In[62]:


print(classification_report(Y_valid, Y_pred_etc))


# # Ensemble Model

# In[63]:


from sklearn.ensemble import VotingClassifier


# In[64]:


ensemble_model = VotingClassifier(estimators=[
    ('RF', RF),
    ('ETC', ETC),
    ('SVM', SVM)
], voting='hard')


# In[65]:


ensemble_model.fit(X_train, Y_train)


# In[66]:


Y_pred_ensemble = ensemble_model.predict(X_valid)


# In[67]:


print(classification_report(Y_valid, Y_pred_ensemble))


# In[81]:


ensemble_model2 = VotingClassifier(estimators=[
    ('RF', RF),
    ('ETC', ETC),
    ('grid', grid)
], voting='hard')


# In[82]:


ensemble_model2.fit(X_train, Y_train)


# In[83]:


Y_pred_ensemble2 = ensemble_model2.predict(X_valid)


# In[84]:


print(classification_report(Y_valid, Y_pred_ensemble2))


# In[ ]:




