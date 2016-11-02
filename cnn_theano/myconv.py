# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 00:25:48 2015

@author: becky
"""

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import time
import matplotlib.pyplot as plt

seed = np.random.randint(1,10000,size = 1)[0]

def model(X,w,w1,w2,dorpc,dorph):
    l1a = relu(conv2d(X,w,border_mode='full'))
    l1b = max_pool_2d(l1a,(2,2))
    l1 = dropout(l1b,dropc);
 
    l2a = relu(conv2d(l1,w1,border_mode='full'))
    l2b = max_pool_2d(l2a,(2,2))
    l2 = dropout(l2b,dropc);
    
    l3a = T.flatten(l2,outdim=2)
    pyx = T.nnet.softmax(T.dot(l3a,w2))
    return pyx

def myweights(shape):
    return theano.shared(np.asarray((np.random.randn(*shape) * 0.01), dtype=theano.config.floatX))
def relu(X):
    return T.maximum(X,0.)
def dropout(X,pr):
    rn = RandomStreams(seed)
    X *= rn.binomial( X.shape,p = pr,dtype = theano.config.floatX)
    return X
    
def RMSprop(cost, params, lr, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

method = 'RMSPROP'
'''
def bp (cost,params,lr):
    momentum = 0.001
    updates = []
    for para  in params:
        param = theano.shared(para.get_value()*0.)
        updates.append((para,para - lr * param ))
        updates.append((param, momentum *param+ (1. - momentum)*T.grad(cost,para)))
    return updates
  '''  
trX ,teX,trY,teY = mnist(onehot = True)
trX = trX.reshape(-1,1,28,28)
teX = teX.reshape(-1,1,28,28)

    
X = T.tensor4()
Y = T.matrix()
w = myweights( (8, 1, 3, 3) )  
w1 = myweights( (4,8,3,3) )
w2 = myweights( (324,10) )
dropc = 0.85
droph = 0.5
pyx = model(X,w,w1,w2,dropc,droph);
y = T.argmax(pyx,axis = 1)

cost = T.mean(T.nnet.categorical_crossentropy(pyx,Y))
params = [w,w1,w2]

lr = 0.05
grads = T.grad(cost=cost,wrt=params)
#update = [(param,param - lr*grad) for param,grad in zip(params,grads)]
update= RMSprop(cost, params,lr)

#update = bp(cost,params,lr)
train = theano.function(inputs = [X,Y],outputs =cost,updates = update,allow_input_downcast = True)
predict = theano.function(inputs = [X],outputs= y,allow_input_downcast = True)

print time.strftime('%Y-%m-%d %H:%M:%S')


st =[]
st.append(str(lr))
st.append('meothod')
st.append(method)


inn =30
tright = 0
costt = []
tright = []
axis = range(inn)
for k in range(inn):
    for i,j in zip(range(0,len(trX)-1000,1000),range(1000,len(trX),1000)):
        ct = train(trX[i:j],trY[i:j])
    tt =np.mean(predict(teX) == np.argmax(teY, axis=1) )
    print tt
    print ct
    costt.append(ct)
    tright.append(tt)
plt.plot(axis,costt,'b')
plt.plot(axis,tright,'r')
st.append(str(tright[-1]))
st.append('--droprate:--')
st.append(str(dropc))
st.append('.jpg')
name = '+'.join(st)
print name
plt.savefig(name)
print time.strftime('%Y-%m-%d %H:%M:%S')