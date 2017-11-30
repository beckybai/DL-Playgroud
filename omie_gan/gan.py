import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_swiss_roll
from sklearn.cluster import AgglomerativeClustering
import mpl_toolkits.mplot3d.axes3d as p3
import time as time

def data_generator():
	n_samples = 1500
	noise = 0.1
	X, _ = make_swiss_roll(n_samples, noise,random_state=4)
	# Make it thinner
	X[:, 0] /= 10
	X[:, 2] /= 10
	x1 = np.array([X[:,0],X[:,2]])
	x2 = np.array([-X[:,0],-X[:,2]])
	return np.concatenate([x1,x2],1)
	



# data = data_generator()
# a = 1
# plt.scatter(data[0,:],data[1,:])
