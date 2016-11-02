import theano
from theano import tensor as T
import numpy as np
from load import mnist
import matplotlib.pyplot as plt
import time

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

def sgd(cost, params, lr):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g ,l in zip(params, grads,lr):
        updates.append([p, p - g * l])
    return updates

def model(X, w_h,w_h2,w_o):
    h = T.switch(T.dot(X,w_h) > 0, T.dot(X,w_h), 0)
    h2 = T.switch(T.dot(h,w_h2) > 0, T.dot(h,w_h2), 0)
    pyx = T.nnet.softmax(T.dot(h2, w_o))
    return pyx

trX, teX, trY, teY = mnist(onehot=True)

X = T.fmatrix()
Y = T.fmatrix()

w_h = weights((784, 256))
w_h2 =weights((256,64))
w_o = weights((64, 10))

py_x = model(X, w_h, w_h2,w_o)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w_h,w_h2,w_o]
lr = [0.05,0.1,0.2]
updates = sgd(cost, params,lr)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

costt = []
rr = []
st = []
for x in lr:
    st.append('%s'% x)
st.append('.jpg')
name = '+'.join(st)


axis= range(20)
print time.strftime('%Y-%m-%d %H:%M:%S')
for i in range(20):
    for start, end in zip(range(0, len(trX), 100), range(100, len(trX), 100)):
        costti = train(trX[start:end], trY[start:end])
    r = np.mean(np.argmax(teY, axis=1) == predict(teX))
    print r
    rr.append(r)
    costt.append(costti)
    
plt.plot(axis,costt,'b')
plt.plot(axis,rr,'r')
plt.savefig(name)
print time.strftime('%Y-%m-%d %H:%M:%S')
