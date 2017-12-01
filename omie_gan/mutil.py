import os
import shutil
import numpy as np
import torch
import sys

class Logger(object):
    def __init__(self,path):
        self.terminal = sys.stdout
        self.log = open("{}/logfile.log".format(path), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def label_num2vec(num_vec, max_label = 2):
    batch_size = num_vec.shape[0]
    if(np.max(num_vec)==0) or np.max(num_vec)==np.min(num_vec):
        pass
    else:
        max_label = np.max(num_vec)
#    print(num_vec)
    label_mat = np.zeros([batch_size,max_label+1])
    for i in range(0,batch_size):
#        print(num_vec[i])
    	label_mat[i,num_vec[i]]=1
    return label_mat

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])
# something strange
# def calc_gradient_penalty(z):
#     G_sample = G(z).detach()
#
#     T_mix = T(combine(z, G_sample))
#     T_ind = T(combine(z, X))
#     GT_loss = -(torch.mean(T_mix) - torch.log(torch.mean(torch.exp(T_ind))))
#
#     D_fake = D(G_sample)
#     GD_loss = criterion(D_fake, ones_label)
#     GD_loss.backward(create_graph=True)
#     GT_loss.backward(create_graph=True)
#     gradients_g = \
#     autograd.grad(outputs=D_fake, inputs=z, grad_outputs=torch.ones(D_fake.size()).cuda() if use_cuda else torch.ones(
#             D_fake.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
#
#     gradients_t = autograd.grad(outputs=GT_loss, inputs=z, create_graph=True, retain_graph=True, only_inputs=True)[0]
#
#     return gradients_g.norm(2, dim=1), gradients_t.norm(2, dim=1)