import numpy as np

def sig(x):
    return 1/(1 + np.exp(-x))

def sig_der(x):
    return x*(1-x)

X = np.array([0,0,1,0,1,1,1,0,1,1,1,1])

X  = X.reshape(4,3)

y = np.array([0,1,1,0])

y = y.reshape(4,1)

# y

# X,y

np.random.seed(1)

syn0 = 2*np.random.random((3,4)) - 1

syn1 = 2*np.random.random((4,1)) -  1

for j in range(10000):
    l0 = X
    l1 = sig(np.dot(l0,syn0))
    l2 = sig(np.dot(l1,syn1))
#     print("L0      \n"+str(l0))
#     print("syn0      \n"+str(syn0))
#     print("L1      \n"+str(l1))
#     print("syn1      \n"+str(syn1))
#     print("L2      \n"+str(l2))
    
    l2_error = y - l2
    print("Error"+ str(np.mean(np.abs(l2_error))))
    
    l2_delta = l2_error*sig_der(l2)
    
    l1_error = l2_delta.dot(syn1.T)
    
    l1_delta = l1_error*sig_der(l1)
    
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
    print("Output after training -> l2 = \n"+ str(l2)+"\n\n\n\n")

# Test cases for vector 1,0,1 it returns 0.99 ~ 1
k = np.array([1,0,1])

print(k)
l0 = k
l1 = sig(np.dot(l0,syn0))
print(l1)
l2 = sig(np.dot(l1,syn1))

print(l2)

