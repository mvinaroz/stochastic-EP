import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
import torch
import argparse
import torch.optim as optim
# from sklearn.metrics import roc_auc_score
import math


class NN_Model(nn.Module):

    def __init__(self, len_m, len_v, num_samps, init_var_params, device, lamb):
        # model = NN_Model(len_m, len_v, n_MC_samps, ms_vs, noise_sams, device, lamb)
        # len_m, len_m_pri, len_v, num_samps, ms_vs, device, gam, lamb, d
        super(NN_Model, self).__init__()
        self.parameter = Parameter(torch.Tensor(init_var_params), requires_grad=True)
        self.num_MC_samps = num_samps
        self.device = device
        self.len_m = len_m
        self.len_v = len_v
        self.lamb = lamb
        self.data_dim = len_m-1
        self.relu = F.relu
        # self.random_ness = random_ness

    def forward(self, x):  # x is mini_batch_size by input_dim

        # unpack ms_vs
        ms_vs = self.parameter
        m_pri = ms_vs[0:self.len_m]
        # because these variances have to be non-negative
        v_pri = F.softplus(ms_vs[self.len_m:])
        sqrt_v_pri = torch.sqrt(v_pri + 1e-6*torch.ones(v_pri.size())) # to avoid numerical issues.

        # Generate MC samples to approx w0
        normal_samps=torch.zeros((self.num_MC_samps, self.len_m))

        samps = torch.zeros((self.num_MC_samps, self.len_m))
        out = torch.zeros((x.shape[0], self.num_MC_samps))
        for i in range(0, self.num_MC_samps):
            # samps[i,:] = m_pri + self.random_ness*sqrt_v_pri
            aux=torch.randn(m_pri.shape)
            normal_samps[i, :]=aux
            #samps[i, :] = m_pri + torch.randn(m_pri.shape) * sqrt_v_pri
            samps[i, :] = m_pri + aux * sqrt_v_pri
            out[:,i] = torch.mm(x, samps[i,:].unsqueeze(1)).squeeze(1)

        #normal_samps=torch.randn((self.num_MC_samps, self.len_m))
        samps_std_adjusted=torch.einsum('kd, d -> kd', normal_samps, sqrt_v_pri)
        #print("This is  samps_std_adjusted: ",  samps_std_adjusted.shape)
        #print("This is m_pri: ", m_pri[None, :].repeat(self.num_MC_samps, 1).shape)
        samps_w=  m_pri[None, :].repeat(self.num_MC_samps, 1) + samps_std_adjusted

        #print("This is samps std: ", torch.std(samps))
        #print("This is samps mean: ", torch.mean(samps))

        #print("This is samps_w std: ", torch.std(samps_w))
        #print("This is samps_w mean: ", torch.mean(samps_w))

        w0_x=torch.einsum('bd, md -> bm', x, samps_w)

        #print("Are out and w0_x equal: ", torch.div(out, w0_x) -1)


        #Compute KL term
        
        KL_term=0.5*( self.lamb*torch.sum(sqrt_v_pri) - (self.data_dim + 1) + self.lamb*torch.sum(m_pri**2) - (self.data_dim + 1)*torch.log(torch.tensor(self.lamb)) - torch.log(torch.sum(sqrt_v_pri)) ) 

        return w0_x, m_pri, v_pri, KL_term # size(out) = n_datapoints by n_MC_samps


def loss_func(pred_samps, y, gam, m_pri, v_pri, KL_term):
    # print("This is pred_samps shape: ", pred_samps.shape)

    n, n_MC_samps = pred_samps.shape

    out1 = torch.zeros((n, n_MC_samps))
    out2 = torch.zeros((n, n_MC_samps))
    for i in range(0, n):
        for j in range(0, n_MC_samps):
            out1[i,j] = gam*0.5/n_MC_samps*(y[i]-pred_samps[i, j])**2

            out2[i,j]=gam*0.5/n_MC_samps*(y[i]**2-2*y[i]*pred_samps[i, j] + pred_samps[i, j]**2)
    
    #print("Check if out1 and out 2 are equal: ", torch.div(out1, out2) -1)
    #out = torch.sum(out1) + 0.5*n*torch.log(2*torch.pi/gam)

    out_expectation=torch.sum(out2) + 0.5*n*torch.log(2*torch.pi/gam)

    trm1_n=y**2
    trm1=trm1_n.repeat(1, n_MC_samps)
    #print("This is trm1: ", trm1)
    #print("This is y.shape", y.shape)
    #print("This is pred_samps.shape: ", pred_samps.shape)
    trm2=2*torch.einsum('nl,nm -> nm', y, pred_samps)

    trm3=torch.square(pred_samps)
    #print("This is torch.square(pred_samps) shape: ", torch.square(pred_samps).shape)


    out_vect=(gam*0.5/n_MC_samps)*(trm1 - trm2 + trm3)
    out_expectation_vect=torch.sum(out_vect) +  0.5*n*torch.log(2*torch.pi/gam)

    #print("Check if out2 and out_vect are equal: ", torch.div(out2, out_vect) - 1)


    out= out_expectation_vect + KL_term

    return out

def main():

    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    """ basic quantities """
    d = 2
    n = 2000
    lamb = 0.1 # prior precision for weights
    gam = 1 # noise precision for y = w*x + noise, where noise \sim N(0, 1/gam)
    epochs = 1000
    n_MC_samps = 10

    """ data generation """
    W = torch.randn(d+1,1) # input dim plus a bias term
    W = torch.sqrt(1/torch.tensor(lamb))*W
    print('ground truth W is', W)
    X = 0.1*torch.randn(n,d)
    X = torch.cat((torch.ones(n,1), X), 1) # n by (d+1)
    noise_samps = torch.sqrt(1/torch.tensor(gam))*torch.randn(n,1)
    y = torch.mm(X, W) + noise_samps

    # for sanity check, take a look at MAP estimate to see if it matches the ground truth W.
    W_MAP = torch.mm(torch.linalg.inv(torch.mm(X.T, X) + 1/torch.tensor(gam)*torch.tensor(lamb)*torch.eye(d+1)), torch.mm(X.T,y))
    print('W_MAP', W_MAP)

    len_m = d + 1  # length of mean parameters for W_0: (d+1)
    len_v = d + 1  # length of variance parameters for W_0

    init_ms = 10*torch.randn(len_m)  # initial values for all means
    init_vs = 10*torch.randn(len_v)  # initial values for all variances
    ms_vs = torch.cat((init_ms, init_vs), 0)

    # noise_samps = torch.randn(len_m)
    model = NN_Model(len_m, len_v, n_MC_samps, ms_vs, device, lamb)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    batch_size = 200
    how_many_iter = np.int(n / batch_size)

    for epoch in range(1, epochs + 1):
        model.train()

        for i in range(how_many_iter):
            # get the inputs
            inputs = X[i * batch_size:(i + 1) * batch_size, :]
            labels = y[i * batch_size:(i + 1) * batch_size]

            optimizer.zero_grad()

            pred_samps, m_pri, v_pri, KL_term = model(inputs)
            loss = loss_func(pred_samps, labels, torch.tensor(gam), m_pri, v_pri, KL_term)

            loss.backward()
            optimizer.step()

        print('Epoch {}: loss : {}'.format(epoch, loss.sum()))

        print('posterior mean of w', m_pri)
        print('ground truth W is', W.squeeze())


if __name__ == '__main__':
    main()