# About ::
# This n-net detects 59 X 63 png images of character hand-written images of a,b,c 
# Very very accurate with mid_layer_node = 25
# atest result [[4.94836621e-04 4.27252829e-03 9.66332135e-01]]  ~ [1 0 0] = 'a'
# btest result [[1.33517175e-01 9.99901978e-01 2.88430202e-07]] ~ [0 1 0] = 'b'
# ctest result [[2.49131158e-02 9.99347773e-01 1.80139512e-04]] ~ [0 1 0] = 'b'
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
mid_layer_node = 25

np.random.seed(1)

syn0 = 2*np.random.random((3717,mid_layer_node)) - 1

syn1 = 2*np.random.random((mid_layer_node,3)) -  1


for j in range(10000):
        
    print(j)
    l0 = X1
    l1 = sig(np.dot(l0,syn0))
    l2 = sig(np.dot(l1,syn1))
    #     print("L0      \n"+str(l0))
    #     print("syn0      \n"+str(syn0))
    #     print("L1      \n"+str(l1))
    #     print("syn1      \n"+str(syn1))
    #     print("L2      \n"+str(l2))
    l2_error = y - l2
    if (j == 9999):
        print(mid_layer_node)
        print(j)
        print("Error"+ str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error*sig_der(l2)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error*sig_der(l1)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

    if (j == 9999):
        print("Output after training -> l2 = \n"+ str(l2)+"\n\n\n\n")



    # k = np.array(X1[0,:])

print(atest)
l0 = atest
l1 = sig(np.dot(l0,syn0))
print(l1)
l2 = sig(np.dot(l1,syn1))

print(l2)
