import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib as mpl
from datetime import datetime

mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import shutil, sys
import mutil
import toy_model as model
import data_prepare
import matplotlib.gridspec as gridspec

out_dir = './out/vanila_gan_{}'.format(datetime.now())
out_dir = out_dir.replace(" ", "_")
print(out_dir)

if not os.path.exists(out_dir):
	os.makedirs(out_dir)
	shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')

sys.stdout = mutil.Logger(out_dir)
gpu = 0
torch.cuda.set_device(gpu)
mb_size = 100  # mini-batch_size
mode_num = 2

roll_noise = 0.1
data = data_prepare.SwissRoll(mb_size, roll_noise)
X_origin = np.array([[0,0]])
for i in range(10):
	X_origin = np.concatenate([X_origin,data.batch_next()],axis=0)
	# X_origin.conjugate(data.batch_next())

Z_dim = 2
X_dim = 2
h_dim = 128
cnt = 0

num = '0'

# else:
#     print("you have already creat one.")
#     exit(1)

G = model.G_Net(Z_dim, X_dim, h_dim).cuda()
D = model.D_Net(X_dim, 1, h_dim).cuda()
T = model.T_Net(X_dim + Z_dim, h_dim).cuda()

G.apply(model.weights_init)
D.apply(model.weights_init)
T.apply(model.weights_init)

""" ===================== TRAINING ======================== """

G_solver = optim.Adam(G.parameters(), lr=1e-4)
D_solver = optim.Adam(D.parameters(), lr=1e-4)
T_solver = optim.Adam(T.parameters(), lr=1e-4)

ones_label = Variable(torch.ones([mb_size, 1])).cuda()
zeros_label = Variable(torch.zeros([mb_size, 1])).cuda()

criterion = nn.BCELoss()

"""Things related to the gradient"""
grads = {}
skip = (slice(None, 5, 5), slice(None, 5, 5))


def save_grad(name):
	def hook(grad):
		grads[name] = grad.data.cpu().numpy()
	
	return hook


x_limit = 2.5
y_limit = 2.5
grid_num = 50
unit = x_limit / (float(grid_num)) * 2

y_fixed, x_fixed = np.mgrid[-x_limit:x_limit:unit, -y_limit:y_limit:unit]
x_fixed, y_fixed = x_fixed.reshape(grid_num * grid_num, 1), y_fixed.reshape(grid_num * grid_num, 1)
mesh_fixed_cpu = np.concatenate([x_fixed, y_fixed], 1)
mesh_fixed = Variable(torch.from_numpy(mesh_fixed_cpu.astype("float32")).cuda())

# mesh_fixed.register_hook(save_grad('Mesh'))
z_fixed = Variable(torch.randn(mb_size*20, Z_dim)).cuda()


# def get_grad(input, label, name):
# 	sample = G(input)
# 	sample.register_hook(save_grad(name))
# 	d_result = D(sample)
# 	ones_label_tmp = Variable(torch.ones([d_result.data.size()[0], 1])).cuda()
# 	loss_real = criterion(d_result, ones_label_tmp * label)
# 	loss_real.backward()
# 	return d_result

def get_grad(input, label, name, c=None, is_z=True, need_sample=False, loss=False):
	D.zero_grad()
	if (is_z):
		if c:
			sample = G(torch.cat([input, c], 1))
		else:
			sample = G(input)
	else:
		input.requires_grad = True
		sample = input
	sample.register_hook(save_grad(name))
	if c is not None:
		d_result = D(torch.cat([sample, c], 1))
	else:
		d_result = D(sample)
	
	ones_label_tmp = Variable(torch.ones([d_result.data.size()[0], 1]) * label).cuda()
	loss_real = criterion(d_result, ones_label_tmp)
	loss_real.backward()
	D.zero_grad()
	G.zero_grad()
	if (need_sample):
		return d_result, sample
	elif (loss):
		return d_result, loss_real
	else:
		return d_result


def combine(z, x):
	return torch.cat([z, x], dim=1)


for it in range(200000):
	# Sample data
	z = Variable(torch.randn(mb_size, Z_dim)).cuda()
	X = data.batch_next()
	X = Variable(torch.from_numpy(X.astype('float32'))).cuda()
	
	### Dicriminator forward-loss-backward-update
	D_solver.zero_grad()
	G_sample = G(z)
	D_real = D(X)
	D_fake = D(G_sample)
	
	D_loss_real = criterion(D_real, ones_label)
	D_loss_fake = criterion(D_fake, zeros_label)
	D_loss = D_loss_real + D_loss_fake
	
	D_loss.backward()
	D_solver.step()
	
	# Housekeeping - reset gradient
	D.zero_grad()
	G.zero_grad()
	
	### Generator forward-loss-backward-update
	z = Variable(torch.randn(mb_size, Z_dim).cuda(), requires_grad=True)
	G_sample = G(z)
	# G_sample.register_hook(save_grad('G'))
	# G_sample.requires_grad= True
	D_fake = D(G_sample)
	GD_loss = criterion(D_fake, ones_label)
	# T part
	T_mix = T(combine(z, G_sample))
	T_ind = T(combine(z, X))
	# GT_loss = torch.mean(T_mix) - torch.log(torch.mean(torch.exp(T_ind)))
	# G_loss = GD_loss + GT_loss
	G_loss = GD_loss
	G_loss.backward()
	G_solver.step()
	# print(grads['G'])
	
	# Housekeeping - reset gradient
	D.zero_grad()
	G.zero_grad()
	
	# Print and plot every now and then
	if it % 500 == 0:
		print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {}'.format(it, D_loss_real.data.tolist(),
																	D_loss_fake.data.tolist(), G_loss.data.tolist()))
		
		# 1. Draw the sampling points
		fig, ax = plt.subplots()
		plt.xlim([-x_limit, x_limit])
		plt.ylim([-y_limit, y_limit])
		# X = X.cpu().data.numpy()
		X_cpu = X_origin
		d_g_sample_cpu, G_sample = get_grad(z_fixed, 1, 'G', c=None, is_z=True, need_sample=True)
		G_sample_cpu = G_sample.cpu().data.numpy()
		plt.scatter(X_cpu[:, 0], X_cpu[:, 1], s=1, edgecolors='blue', color='blue')
		plt.scatter(G_sample_cpu[:, 0], G_sample_cpu[:, 1], s=1, color='red', edgecolors='red')
		plt.show()
		ax.set(adjustable='box-forced', aspect='equal')
		ax.set(title='cgan_{}'.format(it))
		# if (it % 5000 == 0):
		# 	plt.savefig('{}/hehe_{}.pdf'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
		plt.savefig('{}/hehe_{}.png'.format(out_dir, str(it).zfill(3)), bbox_inches='tight')
		
		# 2. Draw the gradient of the sampling point
		gd_cpu = -grads['G']
		ax.quiver(G_sample_cpu[:, 0], G_sample_cpu[:, 1], gd_cpu[:, 0], gd_cpu[:, 1], d_g_sample_cpu.cpu().data.numpy(),
				  units='xy')
		ax.set(title='3 mode CGAN_{}'.format(it))
		# if (it % 5000 == 0):
		# 	plt.savefig('{}/haha_{}.pdf'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
		plt.savefig('{}/haha_{}.png'.format(out_dir, str(it).zfill(3)), bbox_inches='tight')
		plt.close()
		
		# 3. Draw the gradient of the grid points
		d_mesh, loss_mesh = (get_grad(mesh_fixed.detach(), 1, 'mesh', c=None, is_z=False, loss=True))
		d_mesh = d_mesh.cpu().data.numpy()
		loss_mesh = loss_mesh.cpu().data.numpy()
		gd_mesh_cpu = -grads['mesh']
		gd_mesh_cpu_x, gd_mesh_cpu_y = np.expand_dims(gd_mesh_cpu[:, 0], 1).reshape(grid_num,
																					grid_num), np.expand_dims(
				gd_mesh_cpu[:, 1], 1).reshape(grid_num, grid_num)
		d_mesh = d_mesh.reshape(grid_num, grid_num)
		x_fixed, y_fixed = x_fixed.reshape(grid_num, grid_num), y_fixed.reshape(grid_num, grid_num)
		ax.quiver(x_fixed[::3, ::3], y_fixed[::3, ::3], gd_mesh_cpu_x[::3, ::3], gd_mesh_cpu_y[::3, ::3], d_mesh,
				  units='xy')
		
		plt.scatter(X_cpu[:, 0], X_cpu[:, 1], s=1, edgecolors='blue', color='blue')
		plt.scatter(G_sample_cpu[:, 0], G_sample_cpu[:, 1], s=1, color='red', edgecolors='red')
		plt.ylim((-y_limit, y_limit))
		plt.xlim((-x_limit, x_limit))
		plt.savefig('{}/huhu_{}.png'.format(out_dir, str(it).zfill(3)), bbox_inches='tight')
		plt.close()
		## old one
		
		# test_command = os.system("convert -quality 100 -delay 20 {}/*.png {}/video.mp4".format(out_dir, out_dir))
		
		torch.save(G.state_dict(), "{}/G.model".format(out_dir))
		torch.save(D.state_dict(), "{}/D.model".format(out_dir))