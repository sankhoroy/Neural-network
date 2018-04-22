# About ::
# This n-net detects 59 X 63 png images of character hand-written images of a,b,c 
# 2 hidden-layer -> 1st hidden layer 2 node and 2nd hidden layer 1 node
atest =  atest.reshape(1,3717)
import numpy as np

def sig(x):
    return 1/(1 + np.exp(-x))

def sig_der(x):
    return x*(1-x)

# X1 is input 9 x 3717
# l0 is input 9 x 3717
# syn0 is 3717 x 4
# l1 is input 4 x 1
# syn1 is 4 x 3
# l1 is input 3 x 1

# y = np.array([0,1,0,1,1,1])

# y = y.reshape(6,1)

# y

# X,y
s = np.zeros((5,5))
first_hidden_layer_node = 2
second_hidden_layer_node = 1

for  first_hidden_layer_node in range(2,15):
    for second_hidden_layer_node in range(2,15):
        np.random.seed(1)

        syn0 = 2*np.random.random((3717,first_hidden_layer_node)) - 1

        syn1 = 2*np.random.random((first_hidden_layer_node,second_hidden_layer_node)) - 1

        syn2 = 2*np.random.random((second_hidden_layer_node,3)) -  1

        for j in range(10000):



#             print(j)
            l0 = X1
            l1 = sig(np.dot(l0,syn0))
            l2 = sig(np.dot(l1,syn1))
            l3 = sig(np.dot(l2,syn2))


            l3_error = y - l3

            if (j == 9999):
                print(first_hidden_layer_node)
                print(second_hidden_layer_node)
                print(j)
                print("Error"+ str(np.mean(np.abs(l3_error))))
                s[first_hidden_layer_node,second_hidden_layer_node] = np.mean(np.abs(l3_error))

            l3_delta = l3_error*sig_der(l3)

            l2_error = l3_delta.dot(syn2.T)

            l2_delta = l2_error*sig_der(l2)

            l1_error = l2_delta.dot(syn1.T)

            l1_delta = l1_error*sig_der(l1)

            syn2 += l2.T.dot(l3_delta)
            syn1 += l1.T.dot(l2_delta)
            syn0 += l0.T.dot(l1_delta)

            if (j == 9999):
                print("Output after training -> l2 = \n"+ str(l3)+"\n\n\n\n")



            # k = np.array(X1[0,:])

        print(atest)
        l0 = atest
        l1 = sig(np.dot(l0,syn0))
        print(l1)

        l2 = sig(np.dot(l1,syn1))
        print(l2)

        l3 = sig(np.dot(l2,syn2))
        print(l3)

# for i in range(5):
#     for j in range(5):
#         s[i,j] = i + j
#         xx.append(i)
#         yy.append(j)
#         zz.append(i+j)
plt.imshow(s)
