# About ::
# This n-net detects 8 X 8 png images of numeric characters 0 to 9 taken from sklearn datasets
# 2 hidden-layer -> 1st hidden layer 5 node and 2nd hidden layer 5 node
# Relu as activation function and gradient decent alorithm for finding minimum for 10000 epochs
import numpy as np

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

def ex_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# X1 is input 898 x 64
# l0 is input 898 x 64
# syn0 is 64 x 5
# l1 is input 5 x 1
# syn1 is 5 x 5
# l2 is input 5 x 1
# syn2 is 5 x 10
# l3 is input 10 x 1

# y = np.array([0,1,0,1,1,1])

# y = y.reshape(6,1)

# y

# X,y
s = np.zeros((5,5))
first_hidden_layer_node = 64
# second_hidden_layer_node = 5
d = []

np.random.seed(1)

syn0 = 2*np.random.random((64,10)) - 1

# syn0 = np.ones((64,first_hidden_layer_node))

# syn1 = 2*np.random.random((first_hidden_layer_node,10)) - 1

# syn1 = np.ones((first_hidden_layer_node,10))

# syn2 = 2*np.random.random((second_hidden_layer_node,10)) -  1

for j in range(1000):



#             print(j)
    l0 = X1
    l_transit = relu(np.dot(l0,syn0))
    l1 = softmax(l1)
#     l2 = relu(np.dot(l1,syn1))
#     l3 = relu(np.dot(l2,syn2))
#     print(l1)
    l1_error = y - l1

# if (j == 9999):
#     print(first_hidden_layer_node)
#     print(second_hidden_layer_node)
#     print(j)
#     print(syn0)
#     print(syn1)
#     print(syn2)
    print("Error"+ str(np.mean(np.abs(l2_error))))
    d.append(np.mean(np.abs(l2_error)))
#     s[first_hidden_layer_node,second_hidden_layer_node] = np.mean(np.abs(l3_error))

#     l3_delta = l3_error*relu_der(l3)
#     l2_error = l3_delta.dot(syn2.T)
#     l2_delta = l2_error*relu_der(l2)
#     l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*relu_der(l1)

#     syn2 += l2.T.dot(l3_delta)
#     syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


plt.plot(d)
plt.show()

# test set validation 
total = len(ytest)
correct = 0
for i in range(len(ytest)):
    l0 = xtest[i:i+1]
    l_transit = relu(np.dot(l0,syn0))
    l1 = softmax(l_transit)

    if(ytest[i] == np.argmax(l1)):
        correct += 1
percentage = (correct/total)
print("Correcness of model is: "+str(percentage)+"  percent")
# l0 = xtest
# l_transit = relu(np.dot(l0,syn0))
# l1 = softmax(l_transit)
# l1
    