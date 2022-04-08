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
        v_w0 = torch.abs(ms_vs[self.len_m: self.len_m + self.len_m])

        m_pri=ms_vs[self.len_m + self.len_m:self.len_m + self.len_m + self.m_pri]
        v_pri=torch.abs(ms_vs[self.len_m + self.len_m + self.m_pri:])


        #print("This is ms_vs: ", ms_vs)
        #print("This is m_pri1: ", m_pri)
        #print("This is v_pri1: ", v_pri)

        # Generate MC samples to approx w0
        samps_w0 = torch.zeros((m_w0.shape[0], self.num_MC_samps_w0))


        for i in range(0, self.num_MC_samps_w0):
            samps_w0[:, i]=   m_w0 + torch.randn(m_w0.shape)*torch.sqrt(v_w0)


        data_dim = int(self.len_m / self.d_h - 1)
        W0=torch.reshape(samps_w0, (data_dim + 1, self.d_h, self.num_MC_samps_w0))

        x_w0=torch.zeros((x.shape[0], self.d_h, self.num_MC_samps_w0))


        for i in range(0, self.d_h):
            #print("This is torch.matmul(x, W0[:, i, :] shape: ", torch.matmul(x, W0[:, i, :]).shape)
            x_w0[:, i, :]=torch.matmul(x, W0[:, i, :]) 
            
        z=self.relu(x_w0)
        z=torch.cat((torch.ones(x.shape[0], 1, self.num_MC_samps_w0), z), 1)

        # Generate MC samples to approx w1
        samps_w1 = torch.zeros((m_pri.shape[0], self.num_MC_samps_w1))

        for i in range(0, self.num_MC_samps_w1):
            samps_w1[:, i]=   m_pri + torch.randn(m_pri.shape)*torch.sqrt(v_pri)
            
        pred_samps=torch.zeros((x.shape[0], self.num_MC_samps_w0, self.num_MC_samps_w1))

        for i in range(0, self.num_MC_samps_w0):
            #z[:,:, mc_w0] has shape n by d_h + 1
            for j in range(0, self.num_MC_samps_w1):
                pred_samps[:, i, j ]=torch.matmul(z[:,:, i], samps_w1[:, j])


        return pred_samps, m_w0,  v_w0, m_pri, v_pri


        
def loss_func(pred_samps, y, gam):

    n, n_MC_samps_w0, n_MC_samps_w1 = pred_samps.shape  

    out1=torch.zeros((n,  n_MC_samps_w0, n_MC_samps_w1))

    for i in range(0, n):
        for j in range(0, n_MC_samps_w0):
            for k in range(0, n_MC_samps_w1):

                out1[i,j,k]=0.5*gam*(y[i] - pred_samps[i,j,k])**2


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
    epochs = 2000
    n_MC_samps_w0 = 10
    n_MC_samps_w1 = 10
    d_h=2

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
    len_m = d_h * (d + 1)  # length of mean parameters for W_0, where the size of W_0 is d_h by (d+1)
    len_v = d_h * (d + 1)  # length of variance parameters for W_0
    len_m_pri = d_h + 1 # length of mean parameters for w_1
    len_v_pri = d_h + 1 # length of variance parameters for w_1
    init_ms = 1*torch.randn(len_m + len_m_pri) # initial values for all means
    init_vs = 1*torch.randn(len_v + len_v_pri) # initial values for all variances
    ms_vs = torch.cat((init_ms, init_vs), 0)

    #print("This is ms_vs: ", ms_vs.shape)

    model = NN_Model(len_m, len_v, n_MC_samps_w0, n_MC_samps_w1, ms_vs, device, lamb, d_h)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    batch_size = 1000
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

        print('posterior mean of w1', m_w1)
        print('ground truth W1 is', W1.squeeze())
        print('posterior mean of w0', m_w0)
        print('ground truth W0 is', torch.reshape(W0, (-1,)))

        """Compute y predicted from model parameters to compare it with y"""

        W0_pred=torch.randn((d+1)*d_h)*torch.sqrt(v_w0) + m_w0
        W0_pred=torch.reshape(W0_pred, (d + 1, d_h))

        x_w0_pred=torch.mm(X,  W0_pred)

        z0_pred=F.relu(x_w0_pred)
        z0_pred=torch.cat((torch.ones(n,1), z0_pred), 1)

        w1_pred=torch.randn(d_h + 1)*torch.sqrt(v_w1) + m_w1

        z1_pred=torch.matmul(z0_pred,  w1_pred) 

        y_pred=torch.randn((n,1))*torch.sqrt(1/torch.tensor(gam)) + z1_pred

        mse=torch.mean((y - y_pred)**2)

        print("This is MSE: ", mse)

        print("This are the mean and std for y: ,", torch.mean(y), torch.std(y))
        print("This are the mean and std for y_pred: ,", torch.mean(y_pred), torch.std(y_pred))




if __name__ == '__main__':
    main()