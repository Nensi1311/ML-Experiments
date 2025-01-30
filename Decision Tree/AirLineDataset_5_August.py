#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform


# In[2]:


df = pd.read_csv("https://users.stat.ufl.edu/~winner/data/airq402.dat", sep="\s+", engine="python", on_bad_lines="skip", header=None)
df


# In[3]:


df.head()


# In[4]:


df.columns=["City1", "City2", "Average Flight Fare", "Distance", "Average Weekly Passengers", "Market Leading Airline (MLA)", "Market Share MLA","Average MLA Fare", "Low Price Airline (LPA)", "Market Share LPA", "Average LPA Fare"]


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# # Data Cleaning

# In[10]:


df.dtypes


# In[11]:


df.describe(include="object")


# In[12]:


df.select_dtypes(exclude="number").head()


# In[13]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1) # rows, columns, index
plt.title("Average Flight Fare")
sns.distplot(df["Average Flight Fare"], color="green")

plt.figure(figsize=(20,6))
plt.subplot(1,3,2) # rows, columns, index
plt.title("Distance")
sns.distplot(df["Distance"], color="blue")

plt.figure(figsize=(20,6))
plt.subplot(1,3,3) # rows, columns, index
plt.title("Average Weekly Passengers")
sns.distplot(df["Average Weekly Passengers"], color="red")


# In[14]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1) # rows, columns, index
plt.title("Average Flight Fare")
sns.boxplot(df["Average Flight Fare"], orient="vertical", color="green")

plt.figure(figsize=(20,6))
plt.subplot(1,3,2) # rows, columns, index
plt.title("Distance")
sns.boxplot(df["Distance"], orient="vertical", color="blue")


plt.figure(figsize=(20,6))
plt.subplot(1,3,3) # rows, columns, index
plt.title("Average Weekly Passengers")
sns.boxplot(df["Average Weekly Passengers"], orient="vertical", color="red")


# # Bivirate Analysis

# In[15]:


rs=np.random.RandomState(0)
df1=pd.DataFrame(rs.rand(10,10))
corr=df.corr()
corr.style.background_gradient(cmap="coolwarm")


# In[16]:


plt.figure(figsize=(20,12))
plt.subplot(1,2,1)
sns.boxplot(x="Market Leading Airline (MLA)", y="Average Flight Fare", data=df)

plt.subplot(1,2,2)
sns.boxplot(x="Low Price Airline (LPA)", y="Average Flight Fare", data=df)


# In[17]:


sns.pairplot(df, y_vars="Average Flight Fare", x_vars=["Distance", "Average Weekly Passengers", "Market Share MLA", "Average MLA Fare", "Market Share LPA", "Average LPA Fare"])
plt.show()


# In[18]:


D1 = np.log(df["Distance"])
D2 = np.log(df["Average Weekly Passengers"])
D3 = np.log(df["Market Share MLA"])
D4 = np.log(df["Average MLA Fare"])
D5 = np.log(df["Market Share LPA"])
D6 = np.log(df["Average LPA Fare"])


# In[19]:


df.drop(columns=["City1", "City2", "Market Leading Airline (MLA)", "Low Price Airline (LPA)"], axis=1, inplace=True)


# In[20]:


numeric_columns = ["Average Flight Fare", "Distance", "Average Weekly Passengers", "Market Share MLA", "Average MLA Fare", "Market Share LPA", "Average LPA Fare"]


# In[21]:


df.head()


# # Model Building

# In[22]:


scaler=StandardScaler()
data_scaled=df.copy()
data_scaled[numeric_columns]=scaler.fit_transform(df[numeric_columns])


# In[23]:


# split dataset
train_data, test_data=train_test_split(data_scaled, test_size=0.2, random_state=42)


# In[24]:


# separate features and target 
x_train=train_data.drop("Average Flight Fare", axis=1)
y_train=train_data["Average Flight Fare"]
x_test=test_data.drop("Average Flight Fare", axis=1)
y_test=test_data["Average Flight Fare"]


# In[25]:


# drop non numeric columns for model training
x_train=x_train[numeric_columns[1:]] # exclude the target column 
x_test=x_test[numeric_columns[1:]]


# In[26]:


# define the parameter grid
rf_param_grid={
    "n_estimators": randint(50,200),
    "max_features": ["sqrt", "log2"],
    "max_depth": randint(10,50),
    "min_samples_split": randint(2,10),
    "min_samples_leaf": randint(1,10),
    "bootstrap": [True, False]
}


# In[27]:


# intialize the model
rf = RandomForestRegressor(random_state=42)


# In[28]:


# Intialize RandomsearchCV
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=1, error_score="raise")


# In[29]:


# fit the model
rf_random.fit(x_train, y_train)


# In[30]:


# evaluation on tset set
rf_best=rf_random.best_estimator_
test_predictions=rf_best.predict(x_test)
test_mse=mean_squared_error(y_test, test_predictions)
test_r2=r2_score(y_test, test_predictions)


# In[31]:


print("Best rf estimator:" ,rf_best)
print("Test Prediction:" ,test_predictions)
print("Test mean squared error:" ,test_mse)
print("Test R2 Score" ,test_r2)


# In[32]:


# split into train test and valid
train_data, temp_data = train_test_split(data_scaled, test_size=0.4, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


# In[33]:


# separate features and target variable
X_train = train_data.drop("Average Flight Fare", axis=1)
y_train = train_data["Average Flight Fare"]
X_val = val_data.drop("Average Flight Fare", axis=1)
y_val = val_data["Average Flight Fare"]
X_test = test_data.drop("Average Flight Fare", axis=1)
y_test = test_data["Average Flight Fare"]


# In[34]:


# drop non numeric columns for model training
X_train = X_train[numeric_columns[1:]]
X_val = X_val[numeric_columns[1:]]
X_test = X_test[numeric_columns[1:]]


# In[35]:


# define the parameter grid
rf_param_grid={
    "n_estimators": randint(50,200),
    "max_features": ["sqrt", "log2"],
    "max_depth": randint(10,50),
    "min_samples_split": randint(2,10),
    "min_samples_leaf": randint(1,10),
    "bootstrap": [True, False]
}


# In[36]:


# intialize the model
rf1 = RandomForestRegressor(random_state=42)


# In[37]:


# Intialize RandomsearchCV
rf_random = RandomizedSearchCV(estimator=rf1, param_distributions=rf_param_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=1, error_score="raise")


# In[39]:


# fit the model
rf_random.fit(X_train, y_train)


# In[40]:


# evaluation on tset set
rf_best=rf_random.best_estimator_
test_predictions=rf_best.predict(X_test)
test_mse=mean_squared_error(y_test, test_predictions)
test_r2=r2_score(y_test, test_predictions)


# In[41]:


print("Best rf estimator:" ,rf_best)
print("Test Prediction:" ,test_predictions)
print("Test mean squared error:" ,test_mse)
print("Test R2 Score" ,test_r2)


# In[47]:


from sklearn.tree import DecisionTreeRegressor  


# In[48]:


# Define the parameter distribution to sample from
param_dist = {
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20)
}


# In[49]:


dtree_reg = DecisionTreeRegressor(random_state=42)
random_search = RandomizedSearchCV(dtree_reg, param_distributions=param_dist, n_iter=100, cv=5, random_state=42)


# In[50]:


random_search.fit(X_train, y_train)
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_


# In[51]:


print(f"Best Parameters (Random Search): {best_params_random}")
print(f"Best Score (Random Search): {best_score_random}")


# In[ ]:


plt.plot(history.history['loss'],label='Train_loss')
plt.plot(history.history['val_loss'],label='Val_loss')
plt.legend()
plt.xlabel('No of Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()


plt.plot(history.history['accuracy'],label = 'Train_acc')
plt.plot(history.history['val_accuracy'],label = 'Val_acc')
plt.legend()
plt.xlabel('No. of Epochs')
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.show()

