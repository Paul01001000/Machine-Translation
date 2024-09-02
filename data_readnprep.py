#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd

# Data set source: https://www.manythings.org/anki/ German - English deu-eng.zip (271774)

# Data is read and loaded into a pandas dataframe.

# In[ ]:


data = pd.read_csv("deu.txt", sep="\t", header=None)
data.drop(2, axis=1, inplace=True)
data.columns = ["Eng", "Deu"]
print(data.shape)
data.head()


# In[ ]:


data.drop_duplicates(subset="Eng",inplace=True)
print(data.shape)
data.head()


# After removing duplicates 1000 rows are selected.

# In[ ]:


whole_data = data
data = whole_data.iloc[:1000]
data


# Start token '\t' and and end token '\n' are added to the Strings

# In[ ]:


data.index = [x for x in range(data.shape[0])]
data["Start"] = pd.Series(["\t " for _ in range(data.shape[0])])
data["End"] = pd.Series([" \n" for _ in range(data.shape[0])])
data["Eng"] = data["Start"] + data["Eng"] + data["End"]
data["Deu"] = data["Start"] + data["Deu"] + data["End"]

data.head()


# In[ ]:


data.drop(["Start", "End"], axis=1, inplace=True)
print(data.iloc[0,0])
data.head()


# Shuffle the data to avoid any dependencies between rows n and n+1.

# In[ ]:


data = data.sample(frac=1).reset_index(drop=True)
data.to_csv("deu_prep.txt", sep=";", index=False)
#data.head()