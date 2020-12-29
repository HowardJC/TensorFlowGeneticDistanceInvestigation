#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


abalone_train = pd.read_csv(
    "24_Taxon_Distances.csv")
abalone_train


# In[3]:


abalone_train.fillna(0,inplace=True)
abalone_train=abalone_train.set_index("Unnamed: 0")


# In[4]:


abalone_train


# In[5]:


abalone_train=abalone_train+abalone_train.T - np.diag(np.diag(abalone_train))

abalone_train


# In[6]:


abalone_train.reset_index(inplace=True)
abalone_train.drop("Unnamed: 0",axis=1,inplace=True)


# In[7]:


abalone_train


# In[8]:


abalone_train.to_csv('24_CompletedMatrix.csv',index=False)


# In[12]:


CompletetedMatrixedMatrix= pd.read_csv('24_CompletedMatrix.csv',index_col=0)


# In[13]:


for OIterations in range(1000):
    EditedCSV=pd.read_csv('24_CompletedMatrix.csv')
    for IIterations in range(np.random.randint(250,size=1)[0]):
        Cell=np.random.randint(24,size=2)
        EditedCSV.iloc[Cell[0],Cell[1]]=0
        EditedCSV.iloc[Cell[1],Cell[0]]=0
    EditedCSV.to_csv('24TaxonData/24_CompletedMatrix'+str(OIterations)+'.csv',index=False)


# In[14]:


for i in range(1000):
    CompletetedMatrixedMatrix.to_csv('24TaxonFull/24TaxonCorrect'+str(i)+'.csv')


# In[ ]:





# In[ ]:





# In[ ]:




