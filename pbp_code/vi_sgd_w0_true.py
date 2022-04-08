from email.header import decode_header
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

    def __init__(self,  len_m_pri, len_v_pri, num_samps, init_var_params, device, lamb, d_h, W0):
        # len_m, len_m_pri, len_v, num_samps, ms_vs, device, gam, lamb, d
        super(NN_Model, self).__init__()
        self.parameter = Parameter(torch.Tensor(init_var_params), requires_grad=True)
        self.num_MC_samps = num_samps
        self.device = device
        self.len_m_pri = len_m_pri
        self.len_v_pri = len_v_pri
        self.lamb = lamb
        self.relu = F.relu
        self.d_h = d_h
        self.W0=W0

        # self.random_ness = random_ness
   

    def forward(self, x):  # x is mini_batch_size by input_dim

        #data_dim = int(self.len_m / self.d_h - 1)
        #print("This is data_dim: ", data_dim)

        # unpack ms_vs
        z=torch.matmul(x, self.W0) 
        #print("This is z: ", z)
        z=self.relu(z) # N by d_h
        z=torch.cat((torch.ones(x.shape[0], 1), z), 1)

        #print("This is z: ", z)


        ms_vs = self.parameter
        #m_w0 = ms_vs[0:self.len_m]
        #m_pri = ms_vs[self.len_m:self.len_m + self.len_m_pri]
        m_pri1 = ms_vs[0:self.d_h + 1]
        v_pri1 = torch.abs(ms_vs[self.d_h + 1:])


        #print("This is ms_vs: ", ms_vs)
        #print("This is m_pri1: ", m_pri1)
        #print("This is v_pri1: ", v_pri1)

        # Generate MC samples to approx w1
        samps_w1 = torch.zeros((m_pri1.shape[0], self.num_MC_samps))


        for i in range(0, self.num_MC_samps):
            samps_w1[:, i]=   m_pri1 + torch.randn(m_pri1.shape)*torch.sqrt(v_pri1)


        predsamps = torch.matmul(z, samps_w1)



        return predsamps, m_pri1


        
def loss_func(pred_samps, y, gam):

    n, n_MC_samps = pred_samps.shape  

    out1=torch.zeros((n,  n_MC_samps))

    for i in range(0, n):
        for j in range(0, n_MC_samps):

           out1[i,j]=0.5*gam*(y[i] - pred_samps[i,j])**2


    out=torch.mean(out1) + 0.5*torch.log(2*torch.pi/gam)
    return out


def main():

    np.random.seed(0)
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    """ basic quantities """
    d = 2
    n = 2000
    lamb = 1 # prior precision for weights
    gam = 100 # noise precision for y = w*x + noise, where noise \sim N(0, 1/gam)
    epochs = 200
    n_MC_samps = 10
    d_h=2

    np.random.seed(0)

    """ data generation """

    #Generate W1 from N(0, 1./\lamb)
    W1 = torch.randn(d_h+1,1) # num hidden units plus a bias term
    W1 = torch.sqrt(1/torch.tensor(lamb))*W1 

    #Generate W0 from N(0, 1./\lamb)
    W0=torch.sqrt(1/torch.tensor(lamb))*torch.randn(d+1,d_h) # input dim plus a bias term to hidden units 
    #print("This is W0: ", W0)

    print('ground truth W0 is', W0.shape)
    print('ground truth W1 is', W1.shape)

    X = 1*torch.randn(n,d)
    X = torch.cat((torch.ones(n,1), X), 1) # n by (d+1)

    #print("This is X shape: ", X.shape)

    x_w0=torch.mm(X, W0)
    #x_w0_einsum=torch.einsum('nd, dh -> nh', X, W0)
    #print("Are x_w0 and x_w0_einsum equal: ", torch.div(x_w0, x_w0_einsum) - 1)

    sigma=F.relu(x_w0)
    #print("Negative elements after relu: ", sigma[sigma < 0])

    sigma_plus_bias=torch.cat((torch.ones(n,1), sigma), 1)
    #print("This is  sigma_plus_bias: ",  sigma_plus_bias.shape)

    sigma_w1=torch.mm(sigma_plus_bias, W1) 
    #print("This is sigma_w1: ", sigma_w1)

    y=torch.randn((n,1))*torch.sqrt(1/torch.tensor(gam)) + sigma_w1
    #print("This are the labels: ", y)

    """initialize model and it's parameters"""
    #len_m = d_h * (d + 1)  # length of mean parameters for W_0, where the size of W_0 is d_h by (d+1)
    #print("This is len_m: ", len_m)
    #len_v = d_h * (d + 1)  # length of variance parameters for W_0
    len_m_pri = d_h + 1 # length of mean parameters for w_1
    len_v_pri = d_h + 1 # length of variance parameters for w_1
    init_ms = 1*torch.randn(len_m_pri) # initial values for all means
    init_vs = 1*torch.randn(len_v_pri) # initial values for all variances
    #init_ms = 1*torch.randn(len_m + len_m_pri) # initial values for all means
    #init_vs = 1*torch.randn(len_v + len_v_pri) # initial values for all variances
    ms_vs = torch.cat((init_ms, init_vs), 0)

    print("This is ms_vs: ", ms_vs.shape)

    model = NN_Model(len_m_pri, len_v_pri, n_MC_samps, ms_vs, device, lamb, d_h, W0)
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

            predsamps, m_pri1 = model(inputs)
            loss = loss_func(predsamps, labels, torch.tensor(gam))

            loss.backward()
            optimizer.step()

        print('Epoch {}: loss : {}'.format(epoch, loss.sum()))

        print('posterior mean of w1', m_pri1)
        print('ground truth W1 is', W1.squeeze())


if __name__ == '__main__':
    main()