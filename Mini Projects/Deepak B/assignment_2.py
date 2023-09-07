#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
identity_matrix = np.eye(4)
print(identity_matrix)


# In[2]:


import numpy as np
array = np.array([[0, 1, 0, 0],
                  [2, 0, 0, 3],
                  [0, 4, 0, 5]])
nonzero_indices = np.nonzero(array)
print(nonzero_indices)


# In[3]:


import numpy as np

data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

mean_values = np.mean(data, axis=0)
print("Mean values along the axis:", mean_values)
median_values = np.median(data, axis=0)
print("Median values along the axis:", median_values)
std_deviation_values = np.std(data, axis=0)
print("Standard deviation values along the axis:", std_deviation_values)


# In[4]:


import numpy as np
shape = (3, 4)
random_array = np.random.rand(*shape)
print(random_array)


# In[5]:


import numpy as np
array1 = np.array([[1, 2, 3],
                   [4, 5, 6]])
array2 = np.array([[7, 8, 9],
                   [10, 11, 12]])
result = np.concatenate((array1, array2), axis=0) 
print(result)


# In[6]:


import numpy as np
array = np.array([1, 2, 3, 4, 5])
def my_function(x):
    return x ** 2
result = my_function(array)
print(result)


# In[7]:


import numpy as np
array = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
unique_elements, element_counts = np.unique(array, return_counts=True)
print("Unique elements:", unique_elements)
print("Element counts:", element_counts)


# In[8]:


import numpy as np
data_list = [1, 2.5, "hello", True]
array = np.array(data_list)
print(array)


# In[9]:


import numpy as np
array = np.array([[1, 2, 3],
                  [4, 5, 6]])
np.savetxt('array.txt', array)
loaded_array = np.loadtxt('array.txt')
print(loaded_array)


# In[10]:


import numpy as np
array1 = np.array([1, 2, 3])
array2 = np.array([2, 2, 3])
comparison_result = array1 == array2
print(comparison_result)


# In[ ]:




