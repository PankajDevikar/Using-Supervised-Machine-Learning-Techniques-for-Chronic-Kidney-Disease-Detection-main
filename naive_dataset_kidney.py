#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[4]:


data=pd.read_csv("kidney_disease.csv")


# In[5]:


data


# In[6]:


type(data)


# In[7]:


data.info()


# In[8]:


data.describe()


# In[9]:


from sklearn import preprocessing 


# In[10]:


encode=data.apply(preprocessing.LabelEncoder().fit_transform)
encode


# In[11]:


x=encode.iloc[:,:-1]
y=encode.iloc[:,-1] #last colm contains the target varible


# In[ ]:





# In[12]:


x.head()


# In[13]:


y.head()


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2020)
print('shape of data =', x_train.shape)
print('shape of data =', y_train.shape)
print('shape of data =', x_test.shape)
print('shape of data =', y_test.shape)
#20% data kept as test data


# In[15]:


cols = data.columns[data.dtypes.eq('object')]
cols


# In[16]:


data.isnull().sum()


# In[17]:


##rbc_mean=encode["rbc"].mean()
#rbc_mean


# In[18]:


##encode['rbc'].fillna(rbc_mean,inplace= True)


# In[19]:


encode.isnull().sum()


# In[20]:


from sklearn.naive_bayes import GaussianNB


# In[21]:


encode.info()


# In[22]:


classifier = GaussianNB()
classifier.fit(x_train,y_train)


# In[23]:


classifier.score(x_test,y_test) ##accuracy


# In[24]:


classifier.predict(x)


# In[ ]:





# In[25]:


encode['classification']


# In[ ]:





# In[26]:


encode.isnull().any().sum()


# In[27]:


data['classification'].value_counts()


# In[28]:


encode['classification'].value_counts()


# In[29]:


data['classification'] = data['classification'].replace(to_replace={'ckd\t':'ckd'})


# In[30]:


data


# In[ ]:





# In[31]:


ecode=data.apply(preprocessing.LabelEncoder().fit_transform)
ecode


# In[32]:


print(x_train.shape)
print(x_test.shape)


# In[ ]:





# In[ ]:





# In[33]:


x=ecode.iloc[:,:-1]
y=ecode.iloc[:,-1] #last colm contains the target varible


# In[34]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2020)
print('shape of data =', x_train.shape)
print('shape of data =', y_train.shape)
print('shape of data =', x_test.shape)
print('shape of data =', y_test.shape)
#20% data kept as test data


# In[35]:


from sklearn.naive_bayes import GaussianNB


# In[36]:


classifier = GaussianNB()
classifier.fit(x_train,y_train)


# In[37]:


classifier.score(x_test,y_test) ##accuracy


# In[38]:


classifier.predict(x_train)


# In[39]:


classifier.predict(x_test)


# In[ ]:





# In[40]:


# Importing Performance Metrics:
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[41]:


sns.barplot(x=encode["classification"].value_counts().index,
            y=encode["classification"].value_counts().values, color = "darkblue", orient = "h")


# In[42]:


# Making predictions
predictions = classifier.predict(x_test)


# In[43]:


predictions


# In[44]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[45]:


#Start the training
# you tell the model based on your X data what is the corresponding y or target
classifier.fit(x_train,y_train)


# In[46]:


y_test.values


# In[47]:


y_train.values


# In[48]:


# Checking the accruacy of the model
from sklearn.metrics import accuracy_score
print("The accuracy of the  model is :", accuracy_score(predictions,y_test)*100)


# In[ ]:




