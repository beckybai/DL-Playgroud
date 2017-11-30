import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim


def data_generator():
	mean = [0,0]
	rou = 0.5
	std1 = 1
	std2 = 1
	sample_points = mb_size
	cov = [[std1*std1,rou*(std1)*(std2)],[rou*std1*std2, std2*std2]]
	x,y = np.random.multivariate_normal(mean,cov,sample_points).T
	y_margin = np.random.normal(mean[1], std2, sample_points).T
	return x,y,y_margin


mb_size = 1000
Z_dim = 2
X_dim = 1 # mutual information
# y_dim = 1
h_dim = 10
c = 0
lr = 1e-5


def xavier_init(size):
	in_dim = size[0]
	xavier_stddev = 1. / np.sqrt(in_dim / 2.)
	return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


""" ==================== Mutual Information calculator ======================== """

Wzh = xavier_init(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)


def T(z):
	h = F.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
	X = F.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
	return X

T_params = [Wzh, bzh, Whx, bhx]

def reset_grad():
	for p in T_params:
		if p.grad is not None:
			data = p.grad.data
			p.grad = Variable(data.new().resize_as_(data).zero_())


T_solver = optim.Adam(T_params, lr)

iteration = 100000
v_history = []


for it in range(iteration):
	x,y,y_hat = data_generator()
	xy = Variable(torch.from_numpy(np.array([x.astype('float32'),y.astype('float32')]).T))
	xy_hat = Variable(torch.from_numpy(np.array([x.astype('float32'),y_hat.astype('float32')]).T))

	t1 = T(xy)
	t2 = T(xy_hat)
	v = torch.mean(t1) - torch.log(torch.mean(torch.exp(t2)))
	v = -v

	v_history.append(-v)
	v.backward()
	T_solver.step()

	if it % 1000 == 0:
		print('Iter: {}: mutual_information: {}, '.format(it, v))



