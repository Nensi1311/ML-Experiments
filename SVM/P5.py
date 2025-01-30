#!/usr/bin/env python
# coding: utf-8

# # 5. To fit the data into different Machine Learning Algorithms given  below: 
# - i) Linear Regression 
# - ii) Linear Discriminant Analysis (LDA) 
# - iii) Gaussian Naïve Bayes 
# - iv) Decision Tree 
# - v) Random Forest 
# - vi) Support Vector Machine 
# - vii) K-Nearest Neighbour Model

# ## Load Dataset

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("heart.csv")
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df['target'].value_counts()


# In[6]:


df.nunique()


# In[7]:


df.describe()


# In[8]:


df.corr()


# In[9]:


plt.subplots(figsize=(20,10))
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')


# In[10]:


df.target.value_counts(normalize=True)


# In[11]:


plt.figure(figsize=[15,7])
plt.title('Count Plot for Output')
sns.countplot(data=df,x='target')
plt.show()


# In[12]:


plt.subplots(figsize=(20,10))
plt.boxplot(df)
plt.show()


# In[13]:


plt.figure(figsize=(14,6))
sns.lineplot(data=df['target'])
sns.lineplot(data=df['age'])
plt.show()


# In[14]:


x=df.drop('target',axis=1)
y=df['target']


# In[15]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)


# ## i) Linear Regression 

# In[16]:


LRegression = LinearRegression()


# In[17]:


LRegression.fit(x_train,y_train)


# In[18]:


# Make predictions on training and testing data
y_train_pred = LRegression.predict(x_train)
y_test_pred = LRegression.predict(x_test)


# In[19]:


# Calculate performance metrics for training data
train_mae = mean_absolute_error(y_train, y_train_pred)
print("Train MAE", train_mae)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE", train_mse)
train_r2 = r2_score(y_train, y_train_pred)
print("Train R2 Score", train_r2)
train_accuracy = int(train_r2 * 100)
print("Training Accuracy: ", train_accuracy)


# In[20]:


# Calculate performance metrics for testing data
test_mae = mean_absolute_error(y_test, y_test_pred)
print("Test MAE", test_mae)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE", test_mse)
test_r2 = r2_score(y_test, y_test_pred)
print("Test R2 Score", test_r2)
test_accuracy = int(test_r2 * 100)
print("Testing Accuracy: ", test_accuracy)


# ## ii) Linear Discriminant Analysis (LDA) 

# In[21]:


clf=LinearDiscriminantAnalysis()


# In[22]:


clf.fit(x_train,y_train)


# In[23]:


clf.score(x_train,y_train)


# In[24]:


pred_clf=clf.predict(x_test)


# In[25]:


print(classification_report(y_test,pred_clf))


# In[26]:


cm_clf=metrics.confusion_matrix(y_test,pred_clf)
cm_clf


# In[27]:


plt.subplots(figsize=(6,3))
sns.heatmap(cm_clf,annot=True)


# In[28]:


Linear_Discriminant_Accuracy=accuracy_score(y_test,pred_clf)
Linear_Discriminant_Accuracy


# ## iii) Gaussian Naïve Bayes 

# In[29]:


gnb = GaussianNB()


# In[30]:


gnb.fit(x_train,y_train)


# In[31]:


gnb.score(x_train,y_train)


# In[32]:


pred_gnb=gnb.predict(x_test)


# In[33]:


Bayes_Theorem_Accuracy=accuracy_score(y_test,pred_gnb)
Bayes_Theorem_Accuracy


# In[34]:


cm_gnb=metrics.confusion_matrix(y_test,pred_gnb)
cm_gnb


# In[35]:


print(classification_report(y_test,pred_gnb))


# In[36]:


plt.subplots(figsize=(6,3))
sns.heatmap(cm_gnb,annot=True)


# ## iv) Decision Tree 

# In[37]:


DT = DecisionTreeClassifier(random_state=42)


# In[38]:


DT.fit(x_train, y_train)


# In[39]:


predictions = clf.predict(x_test) 
accuracy = accuracy_score(y_test, predictions) 
print(f"Accuracy: {accuracy * 100:.2f}") 


# In[40]:


print(classification_report(y_test,predictions))


# ## v) Random Forest 

# In[41]:


RF = RandomForestClassifier(n_estimators = 100)  


# In[42]:


RF.fit(x_train, y_train)


# In[43]:


RFC = RF.predict(x_test) 
accuracy = accuracy_score(y_test, RFC) 
print(f"Accuracy: {accuracy * 100:.2f}") 


# In[44]:


print(classification_report(y_test, RFC))


# ## vi) Support Vector Machine 

# In[45]:


SVM = svm.SVC()


# In[46]:


SVM.fit(x_train, y_train)


# In[47]:


SVMC = SVM.predict(x_test) 
accuracy = accuracy_score(y_test, SVMC) 
print(f"Accuracy: {accuracy * 100:.2f}") 


# In[48]:


print(classification_report(y_test, SVMC))


# ## vii) K-Nearest Neighbour 

# In[49]:


neigh = KNeighborsClassifier(n_neighbors=3)


# In[50]:


neigh.fit(x_train, y_train)


# In[51]:


KNNC = neigh.predict(x_test) 
accuracy = accuracy_score(y_test, KNNC) 
print(f"Accuracy: {accuracy * 100:.2f}") 


# In[52]:


print(classification_report(y_test, KNNC))


# In[ ]:




