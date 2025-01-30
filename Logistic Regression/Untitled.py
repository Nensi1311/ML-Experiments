#!/usr/bin/env python
# coding: utf-8

# # Data Loading

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("Breast_Cancer.csv")
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.describe(include="object")


# In[8]:


df.shape


# In[9]:


df.columns


# # EDA

# In[10]:


df.isnull().sum()


# In[11]:


df["Age"].unique()


# In[12]:


df["Race"].unique()


# In[13]:


df["Marital Status"].unique()


# In[14]:


df["T Stage"].unique()


# In[15]:


df["N Stage"].unique()


# In[16]:


df["6th Stage"].unique()


# In[17]:


df["differentiate"].unique()


# In[18]:


df["Grade"].unique()


# In[19]:


df["A Stage"].unique()


# In[20]:


df["Tumor Size"].unique()


# In[21]:


df["Estrogen Status"].unique()


# In[22]:


df["Progesterone Status"].unique()


# In[23]:


df["Regional Node Examined"].unique()


# In[24]:


df["Reginol Node Positive"].unique()


# In[25]:


df["Survival Months"].unique()


# In[26]:


df["Status"].unique()


# In[27]:


sns.boxplot(x="Age", data=df)


# In[28]:


sns.boxplot(x="Tumor Size", data=df)


# In[29]:


sns.boxplot(x="Regional Node Examined", data=df)


# In[30]:


sns.boxplot(x="Reginol Node Positive", data=df)


# In[31]:


sns.boxplot(x="Survival Months", data=df)


# In[32]:


df.drop(columns=["Survival Months", "Marital Status"], inplace=True, axis=1)


# In[33]:


df.head()


# In[34]:


# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()


# In[35]:


# df['T Stage'] = encoder.fit_transform(df['T Stage'])
# df['N Stage'] = encoder.fit_transform(df['N Stage'])
# df['6th Stage'] = encoder.fit_transform(df['6th Stage'])
# df['differentiate'] = encoder.fit_transform(df['differentiate'])
# df['A Stage'] = encoder.fit_transform(df['A Stage'])
# df['Estrogen Status'] = encoder.fit_transform(df['Estrogen Status'])
# df['Progesterone Status'] = encoder.fit_transform(df['Progesterone Status'])
# df['Status'] = encoder.fit_transform(df['Status'])
# df['Race'] = encoder.fit_transform(df['Race'])
# df['Grade'] = encoder.fit_transform(df['Grade'])
# df.head()


# In[36]:


from sklearn.preprocessing import OneHotEncoder


# In[37]:


categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded = pd.concat([df, one_hot_df], axis=1)
df_encoded = df_encoded.drop(categorical_columns, axis=1)
df_encoded


# In[38]:


X = df_encoded.drop(['Status_Alive','Status_Dead'], axis=1)
Y = df['Status']


# # Decision Tree

# In[39]:


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# In[40]:


model = DecisionTreeClassifier(random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# In[41]:


model.fit(X_train, Y_train)


# In[42]:


Y_pred = model.predict(X_test)


# In[43]:


print(classification_report(Y_test, Y_pred))


# In[44]:


from sklearn.model_selection import cross_val_score,KFold,LeaveOneOut
k_folds = KFold(n_splits = 10)
loo = LeaveOneOut()


# In[45]:


scores = cross_val_score(model, X, Y, cv = loo)


# In[46]:


print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))


# ### Observation: By using descision tree and test size of 30%, we get 77% accuracy 

# # Grid Search Tuning

# In[48]:


from sklearn.model_selection import GridSearchCV


# In[49]:


param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state = 42), param_grid, cv = k_folds, scoring='accuracy')
grid_search.fit(X_train, Y_train)


# In[51]:


print(grid_search.best_params_)
print(grid_search.best_score_)


# # Random Forest

# In[52]:


from sklearn.ensemble import RandomForestClassifier


# In[53]:


model_rf = RandomForestClassifier(n_estimators=100, random_state=42) 


# In[54]:


model_rf.fit(X_train, Y_train)
Y_pred_rf = model_rf.predict(X_test)


# In[55]:


print(classification_report(Y_test, Y_pred_rf))


# In[56]:


scores = cross_val_score(model_rf, X, Y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))


# ### Observation: By using Random Forest we get 84% accuracy

# # Logistic Regression

# In[57]:


from sklearn.linear_model import LogisticRegression


# In[58]:


LR = LogisticRegression()


# In[59]:


LR.fit(X_train, Y_train)
Y_pred_LR = LR.predict(X_test)


# In[60]:


print(classification_report(Y_test, Y_pred_LR))


# In[61]:


scores = cross_val_score(LR, X, Y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))


# ### Observation: By using Logistic Regression we get 85% accuracy

# In[ ]:




