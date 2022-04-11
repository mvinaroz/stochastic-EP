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

import matplotlib
from matplotlib import pyplot
#matplotlib.pyplot.switch_backend('agg')

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class NN_Model(nn.Module):

    def __init__(self,  len_m, len_v, num_samps_w0, num_samps_w1, init_var_params, device, lamb, d_h):
        # len_m, len_m_pri, len_v, num_samps, ms_vs, device, gam, lamb, d
        super(NN_Model, self).__init__()
        self.parameter = Parameter(torch.Tensor(init_var_params), requires_grad=True)
        self.num_MC_samps_w0 = num_samps_w0
        self.num_MC_samps_w1 = num_samps_w1
        self.device = device
        self.len_m = len_m
        self.len_v = len_v
        self.m_pri= d_h + 1
        self.v_pri= d_h + 1
        self.lamb = lamb
        self.relu = F.relu
        self.d_h = d_h

        # self.random_ness = random_ness
   

    def forward(self, x):  # x is mini_batch_size by input_dim


        # unpack ms_vs
        ms_vs = self.parameter
        m_w0 = ms_vs[0:self.len_m]
        m_pri=ms_vs[self.len_m: self.len_m + self.m_pri]


        v_w0 = torch.abs(ms_vs[self.len_m + self.m_pri: self.len_m + self.m_pri + self.len_m])
        #v_w0=self.relu(ms_vs[self.len_m + self.m_pri: self.len_m + self.m_pri + self.len_m])
        v_w0=v_w0 + 1e-6*torch.ones(v_w0.size())
   

        v_pri=torch.abs(ms_vs[self.len_m + self.m_pri + self.len_m:])
        #v_pri=self.relu(ms_vs[self.len_m + self.m_pri + self.len_m:])
        v_pri=v_pri + 1e-6*torch.ones(v_pri.size())



        # Generate MC samples to approx w0
        samps_w0 = torch.zeros((m_w0.shape[0], self.num_MC_samps_w0))


        for i in range(0, self.num_MC_samps_w0):
            samps_w0[:, i]=   m_w0 + torch.randn(m_w0.shape)*torch.sqrt(v_w0)


        data_dim = int(self.len_m / self.d_h - 1)
        W0=torch.reshape(samps_w0, (data_dim + 1, self.d_h, self.num_MC_samps_w0))

        #x_w0=torch.zeros((x.shape[0], self.d_h, self.num_MC_samps_w0))


        #for i in range(0, self.d_h):
        #    for j in range(0, self.num_MC_samps_w0):
                # x has shape N by d+1
                # W0[:, i, j] has shape d+1 
        #        x_w0[:, i, j]=torch.matmul(x, W0[:, i, j]) 

        x_w0=torch.einsum('nd, dhm -> nhm', x, W0)
           
        z=self.relu(x_w0)
        z=torch.cat((torch.ones(x.shape[0], 1, self.num_MC_samps_w0), z), 1)

        # Generate MC samples to approx w1
        samps_w1 = torch.zeros((m_pri.shape[0], self.num_MC_samps_w1))

        for i in range(0, self.num_MC_samps_w1):
            samps_w1[:, i]= m_pri + torch.randn(m_pri.shape)*torch.sqrt(v_pri)
            
        #pred_samps=torch.zeros((x.shape[0], self.num_MC_samps_w0, self.num_MC_samps_w1))

        #for i in range(0, self.num_MC_samps_w0):
        #    pred_samps[:, i, :]=torch.matmul(z[:,:, i], samps_w1)

        pred_samps=torch.einsum('ndm, dc -> nmc', z, samps_w1)

        #print("Check if pred_samps and pred_samps_einsum are equal: ", torch.div( pred_samps, pred_samps_einsum) - 1)

        return pred_samps, m_w0,  v_w0, m_pri, v_pri


        
def loss_func(pred_samps, y, gam):

    n, n_MC_samps_w0, n_MC_samps_w1 = pred_samps.shape  

    out1=torch.zeros((n,  n_MC_samps_w0, n_MC_samps_w1))


    for j in range(0, n_MC_samps_w0):
        for k in range(0, n_MC_samps_w1):

            out1[:,j,k]=0.5*gam*(y.squeeze() - pred_samps[:,j,k])**2


    out=torch.mean(out1) + 0.5*torch.log(2*torch.pi/gam)

    
        
    return out

def main():

    np.random.seed(5)
    torch.manual_seed(5)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    """ basic quantities """
    d = 2
    n = 10000
    n_test=1000
    lamb = 1 # prior precision for weights
    gam = 100 # noise precision for y = w*x + noise, where noise \sim N(0, 1/gam)
    epochs = 200
    n_MC_samps_w0 = 20
    n_MC_samps_w1 = 20
    d_h=5

    """ data generation """

    #Generate W1 from N(0, 1./\lamb)
    W1 = torch.randn(d_h+1,1) # num hidden units plus a bias term
    W1 = torch.sqrt(1/torch.tensor(lamb))*W1 

    #Generate W0 from N(0, 1./\lamb)
    W0=torch.sqrt(1/torch.tensor(lamb))*torch.randn(d+1,d_h) # input dim plus a bias term to hidden units 
    #print("This is W0: ", W0)

    #print('ground truth W0 is', W0.shape)
    #print('ground truth W1 is', W1.shape)

    X = 1*torch.randn(n,d)
    X = torch.cat((torch.ones(n,1), X), 1) # n by (d+1)

    X_test = 1*torch.randn(n_test,d)
    X_test = torch.cat((torch.ones(n_test,1), X_test), 1) # n by (d+1)

    #print("This is X shape: ", X.shape)

    x_w0=torch.mm(X, W0)
    #x_w0_einsum=torch.einsum('nd, dh -> nh', X, W0)
    #print("Are x_w0 and x_w0_einsum equal: ", torch.div(x_w0, x_w0_einsum) - 1)

    x_test_w0=torch.mm(X_test, W0)

    sigma=F.relu(x_w0)
    sigma_test=F.relu(x_test_w0)
    #print("Negative elements after relu: ", sigma[sigma < 0])

    sigma_plus_bias=torch.cat((torch.ones(n,1), sigma), 1)
    sigma_test_plus_bias=torch.cat((torch.ones(n_test,1), sigma_test), 1)
    #print("This is  sigma_plus_bias: ",  sigma_plus_bias.shape)

    sigma_w1=torch.mm(sigma_plus_bias, W1) 
    sigma_test_w1=torch.mm(sigma_test_plus_bias, W1) 
    #print("This is sigma_w1: ", sigma_w1)

    y=torch.randn((n,1))*torch.sqrt(1/torch.tensor(gam)) + sigma_w1
    y_test=torch.randn((n_test, 1))*torch.sqrt(1/torch.tensor(gam)) + sigma_test_w1
    print("This is sigma_test_w1: ", sigma_test_w1)

    """initialize model and it's parameters"""
    len_m = d_h * (d + 1)  # length of mean parameters for W_0, where the size of W_0 is d_h by (d+1)
    len_v = d_h * (d + 1)  # length of variance parameters for W_0
    len_m_pri = d_h + 1 # length of mean parameters for w_1
    len_v_pri = d_h + 1 # length of variance parameters for w_1
    init_ms = 1*torch.randn(len_m + len_m_pri) # initial values for all means
    init_vs = 1*torch.randn(len_v + len_v_pri)# initial values for all variances
    #init_vs=1*torch.ones(len_m + len_m_pri)
    ms_vs = torch.cat((init_ms, init_vs), 0)

    print("This is ms_vs: ", ms_vs)

    model = NN_Model(len_m, len_v, n_MC_samps_w0, n_MC_samps_w1, ms_vs, device, lamb, d_h)
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

            pred_samps, m_w0, v_w0, m_w1, v_w1= model(inputs)
            loss = loss_func(pred_samps, labels, torch.tensor(gam))

            loss.backward()
            optimizer.step()

        print('Epoch {}: loss : {}'.format(epoch, loss.sum()))

        """Compute y_test predicted from model parameters to compare it with y_test"""

        pred_samps_test, m_w0_test, v_w0_test, m_w1_test, v_w1_test = model(X_test)

        W0_test=torch.randn((d+1)*d_h)*torch.sqrt(v_w0_test) +  m_w0_test
        W0_test=torch.reshape(W0_test, (d + 1, d_h))

        x_w0_test=torch.mm(X_test,  W0_test)

        z0_test=F.relu(x_w0_test)
        z0_test=torch.cat((torch.ones(n_test,1), z0_test), 1)

        w1_test=torch.randn(d_h + 1)*torch.sqrt(v_w1_test) + m_w1_test

        z1_test=torch.matmul(z0_test,  w1_test) 


        y_test_pred=torch.randn(n_test)*torch.sqrt(1/torch.tensor(gam)) + z1_test


        mse_test=torch.mean((y_test - y_test_pred)**2)

        print("This is test MSE: ", mse_test)
        #print("This is y_test: ", y_test.squeeze().numpy())


    matplotlib.pyplot.figure(figsize = [10, 5]) # larger figure size for subplots

    # example of somewhat too-large bin size
    matplotlib.pyplot.subplot(1, 2, 1) # 1 row, 2 cols, subplot 1

    matplotlib.pyplot.hist(y_test.squeeze().detach().numpy(), bins=20)
    matplotlib.pyplot.xlabel('y_test')

    # example of somewhat too-small bin size
    matplotlib.pyplot.subplot(1, 2, 2) # 1 row, 2 cols, subplot 2


    matplotlib.pyplot.hist(y_test_pred.squeeze().detach().numpy(),  bins=20)
    matplotlib.pyplot.xlabel('y_test_pred')
    matplotlib.pyplot.show()




if __name__ == '__main__':
    main()