
# coding: utf-8

# In[2]:


get_ipython().system('pip install scikit-learn')


# In[3]:


import numpy as np


# In[4]:


from sklearn.datasets import load_digits


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


digits = load_digits()


# In[7]:


X = digits.images.reshape((1797,64))


# In[22]:


Y = digits.target


# In[9]:


# activation functions
def relu(x):
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                pass  # do nothing since it would be effectively replacing x with x
            else:
                x[i][k] = 0
    return x

def relu_der(x):
    for i in range(0, len(x)):
        for k in range(len(x[i])):
            if x[i][k] > 0:
                x[i][k] = 1
            else:
                x[i][k] = 0
    return x


# In[16]:


def ex_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# In[17]:


np.random.seed(1)


# In[23]:


y = np.zeros((len(Y),10))


# In[26]:


for i in range(len(Y)):
        temp = np.zeros(10)
        for j in range(10):
            if(Y[i] == j):
                temp[j] = 1
            else:
                temp[j] = 0
        y[i] = temp 


# In[112]:


#  It is autoencoder which compresses image down and again reconstruct it
epochs =  20
dense_node = 2

syn0 = 2*np.random.random((64,dense_node)) - 1

syn1 = 2*np.random.random((dense_node,64)) - 1

d = []

for j in range(epochs):
    l0 = X

    l1 = relu(np.dot(l0,syn0))
    
#     l1 = softmax(l_transit)

    l2 = relu(np.dot(l1,syn1))

    l2_error = X - l2

    d.append(np.mean(np.abs(l2_error)))
    
    print("Epoch number "+str(j))
    print("Error : "+ str(np.mean(np.abs(l2_error)))+"\n\n")
    
    l2_delta = l2_error*relu_der(l2)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error*relu_der(l1)

    syn1 += l1.T.dot(l2_delta)
    
    syn0 += l0.T.dot(l1_delta)

plt.plot(d)
plt.show()


# In[113]:


l0 = X[1:2]

l_transit = relu(np.dot(l0,syn0))
    
l1 = softmax(l_transit)

l2 = relu(np.dot(l1,syn1))

plt.imshow(l2[0].reshape((8,8)))

plt.show()


# In[116]:


plt.imshow(X[0].reshape((8,8)))
plt.show()

