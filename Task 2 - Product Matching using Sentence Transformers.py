#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import numpy as np 
import pandas as pd 


# In[2]:


#Loading the amazon data
amazon_data = pd.read_csv('amz_com-ecommerce_sample.csv',encoding='latin-1')
amazon = amazon_data.copy()
amazon_data.head()


# In[3]:


#Loading the flipkart Data
flipkart_data = pd.read_csv('flipkart_com-ecommerce_sample.csv',encoding='latin-1')
flipkart = flipkart_data.copy()
flipkart_data.head()


# In[4]:


amazon_data.shape, flipkart_data.shape


# In[5]:


amazon_data = amazon_data[['product_name','retail_price','discounted_price']]
flipkart_data = flipkart_data[['product_name','retail_price','discounted_price']]


# In[6]:


amazon_data.head()


# In[7]:


#Sentence transformers is a Python framework for cutting-edge sentence vector representations. 

get_ipython().system('pip install sentence_transformers')


# In[8]:


from sentence_transformers import SentenceTransformer, util
import torch


# In[9]:


# Instantiate a model of the SentenceTransformer class
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embedding for both Amazon and Flipkart product names
amazon_prod_embs = model.encode(amazon_data.product_name,convert_to_tensor=True)
flipkart_prod_embs = model.encode(flipkart_data.product_name,convert_to_tensor=True)

# Compute cosine similarities
cosine_sim_scores = util.cos_sim(amazon_prod_embs,flipkart_prod_embs)


# In[10]:


# Storing the index of a similar Flipkart product corresponding to each Amazon product 
similar_prod_indexes = []

for i in range(20000):
    max_score_idxs = torch.topk(cosine_sim_scores[i],k=5,largest=True,sorted=True).indices
    for idx in max_score_idxs:
        if idx != i:
            similar_prod_indexes.append(idx)
            break


# In[12]:


similar_prod_indexes = [x.item() for x in similar_prod_indexes]


# In[13]:


amazon_data.product_name[2936], flipkart_data.product_name[similar_prod_indexes[2936]]


# In[14]:


# Extracting the details such as retail price and discounted price for both Flipkart and Amazon similar products
flipkart_prod_data = []
amazon_prod_data = []

for idx, prod_name in enumerate(amazon_data.product_name):
    flipkart_prod_data.append(flipkart_data.iloc[similar_prod_indexes[idx]])
    amazon_prod_data.append(amazon_data.iloc[idx])


# In[15]:


flipkart_prod_data = pd.DataFrame(flipkart_prod_data)
flipkart_prod_data.columns = ['Product name in Flipkart','Retail Price in Flipkart','Discounted Price in Flipkart']
amazon_prod_data = pd.DataFrame(amazon_prod_data)
amazon_prod_data.columns = ['Product name in Amazon','Retail Price in Amazon','Discounted Price in Amazon']


# In[16]:


flipkart_prod_data.head()


# In[17]:


amazon_prod_data['Retail Price in Amazon'] = amazon_prod_data['Retail Price in Amazon'].astype(np.float64)
amazon_prod_data['Discounted Price in Amazon'] = amazon_prod_data['Discounted Price in Amazon'].astype(np.float64)


# In[18]:


amazon_prod_data.head()


# In[19]:


flipkart_prod_final_data = flipkart_prod_data.copy()
flipkart_prod_final_data = flipkart_prod_final_data.reset_index(drop=True)
flipkart_prod_final_data.head()


# In[20]:



amazon_prod_data.shape, flipkart_prod_final_data.shape


# In[21]:


final_prod_data = pd.concat([flipkart_prod_final_data,amazon_prod_data],axis=1)
final_prod_data.to_csv('final_result.csv')


# In[22]:


final_prod_data.head()


# In[23]:


df = pd.read_csv('final_result.csv')
df.head()


# In[ ]:




