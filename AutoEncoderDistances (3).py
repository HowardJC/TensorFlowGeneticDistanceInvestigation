#!/usr/bin/env python
# coding: utf-8

# In[23]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os  
import glob
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow import keras
from pathlib import Path
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from keras.layers import Lambda


# In[ ]:





# In[ ]:





# In[45]:


latent_dim = 64 

class Encoder(Layer):
  def __init__(self):
    super(Encoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
    
      layers.Input(shape=(24, 24, 1)), 
      layers.Conv2D(16, (6,6), activation='relu', padding='valid', strides=1),
      layers.Conv2D(8,(2,2), activation='relu', padding='valid', strides=1)])

  def call(self, x):
    encoded = self.encoder(x)
    return encoded


# In[ ]:





# In[ ]:





# In[ ]:





# In[46]:


class Decoder(Layer):
  def __init__(self):
    super(Decoder, self).__init__()
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=(2,2), strides=1, activation='relu',padding='valid'),
      layers.Conv2DTranspose(16, kernel_size=(6,6), strides=1, activation='relu',padding='valid'),
      layers.Conv2D(1, kernel_size=(1,1), activation='sigmoid', padding='valid')])

  def call(self, x):
    decoded = self.decoder(x)
    return decoded


# In[ ]:





# In[47]:


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.Autoencoder=tf.keras.Sequential([
            Encoder(),
            Decoder()
        ])
    def call(self,x):
        return self.Autoencoder(x)


# In[ ]:





# In[48]:


autoencoder = Autoencoder()


# In[49]:


autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())


# In[ ]:





# In[50]:


Full=tf.io.gfile.listdir('24TaxonFull')


# In[51]:


Missing=tf.io.gfile.listdir('24TaxonData')


# In[ ]:





# In[52]:


FullDataset=[]
for i in Full:
    name, extension = os.path.splitext('24TaxonFull/'+str(i))
    if extension=='.csv':
        ReadPanda=pd.read_csv('24TaxonFull/'+str(i))
        FullDataset.append(ReadPanda)


# In[53]:


MissingDataset=[]
for i in Missing:

    name, extension = os.path.splitext('24TaxonData/'+str(i))
    if extension=='.csv':
        ReadPanda=pd.read_csv('24TaxonData/'+str(i))
        MissingDataset.append(ReadPanda)


# In[54]:


np.shape(np.array(FullDataset))
FullDataset=np.array(FullDataset)
np.save('FullDataset',FullDataset)
np.shape(np.array(MissingDataset))
MissingDataset=np.array(MissingDataset)
np.save('MissingDataset',MissingDataset)


# In[ ]:





# In[55]:


np.shape(MissingDataset)


# In[56]:


FullDataSet=FullDataset[...,tf.newaxis]
np.shape(FullDataSet)


# In[57]:


MissingDataSet=MissingDataset[...,tf.newaxis]
np.shape(MissingDataSet)


# In[ ]:





# In[58]:


if os.path.isdir("DistanceEncoderWeights"):
    pass
else:
    autoencoder.fit(MissingDataSet,FullDataSet,
                    epochs=1000,
                    shuffle=True,
                   batch_size=5)
    autoencoder.save("DistanceEncoderWeights")


# In[ ]:





# In[66]:


autoencoder.evaluate(MissingDataSet,FullDataSet,20)


# In[67]:


Test=MissingDataset[tf.newaxis,0,...,tf.newaxis]
np.shape(Test)


# In[68]:



Export=autoencoder.predict(Test)


# In[ ]:





# In[62]:


Export=pd.DataFrame(np.reshape(Export,(24,24)))


# In[63]:


Export=Export-np.diag(np.diag(Export))


# In[64]:


Export.to_csv("Example_Output.csv",index=False)


# In[65]:


Export


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




