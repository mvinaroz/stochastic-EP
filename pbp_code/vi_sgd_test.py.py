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

    def __init__(self, len_m, len_v, len_m_pri, len_v_pri, num_samps, init_var_params, device, lamb, d_h):
        # len_m, len_m_pri, len_v, num_samps, ms_vs, device, gam, lamb, d
        super(NN_Model, self).__init__()
        self.parameter = Parameter(torch.Tensor(init_var_params), requires_grad=True)
        self.num_MC_samps = num_samps
        self.device = device
        self.len_m = len_m
        self.len_v = len_v
        self.len_m_pri = len_m_pri
        self.len_v_pri = len_v_pri
        self.lamb = lamb
        self.relu = F.relu
        self.d_h = d_h

        # self.random_ness = random_ness
   

    def forward(self, x):  # x is mini_batch_size by input_dim

        data_dim = int(self.len_m / self.d_h - 1)
        #print("This is data_dim: ", data_dim)

        # unpack ms_vs

        ms_vs = self.parameter
        m_w0 = ms_vs[0:self.len_m]
        m_pri = ms_vs[self.len_m:self.len_m + self.len_m_pri]
        # because these variances have to be non-negative
        v_w0 = F.softplus(ms_vs[self.len_m + self.len_m_pri:self.len_m + self.len_m_pri + self.len_v])
        sqrt_v_w0 = torch.sqrt(v_w0  + 1e-6*torch.ones(v_w0.size())) # to avoid numerical issues.
        #print("This is sqrt_v shape: ", sqrt_v.shape)

        v_pri = F.softplus(ms_vs[self.len_m + self.len_m_pri + self.len_v:])
        sqrt_v_pri = torch.sqrt(v_pri + 1e-6*torch.ones(v_pri.size())) # to avoid numerical issues.
        #print("This is sqrt_v shape: ", sqrt_v_pri.shape)

        # Generate MC samples to approx w0
        samps_w0 = torch.zeros((self.len_m, self.num_MC_samps))
        #print("This is samps shape: ", samps_w0.shape)

        for i in range(0, self.num_MC_samps):
            samps_w0[:, i]=  m_w0 + torch.randn(self.len_m)*sqrt_v_w0

        #print("This is x shape: ", x.shape)
        W0=torch.reshape(samps_w0, (data_dim + 1, self.d_h, self.num_MC_samps))
        #print("This is W0: ", W0.shape)

        #print("This is data shape: ", x.shape)
        x_w0=torch.zeros((x.shape[0], self.d_h, self.num_MC_samps))

        for hidden_dim in range(0,  self.d_h):
            #print("W0 shape: ", W0[:, hidden_dim,:].shape)
            for mc in range(0,  self.num_MC_samps):
                #print("W0 shape: ", W0[:, hidden_dim,mc].shape)
                #print("W0.unsqueeze(1) shape: ", W0[:, hidden_dim,mc].unsqueeze(1).shape)
                #print("x_w0: ", torch.mm(x, W0[:, hidden_dim, mc].unsqueeze(1)).shape)
                x_w0[:, hidden_dim, mc]=torch.mm(x, W0[:, hidden_dim, mc].unsqueeze(1)).squeeze(1)


        #print("This is x_w0: ",  x_w0.shape )
        #x_w0_einsum=torch.einsum('nd, dhm -> nhm', x, W0)

        #print("Check if x_w0 and x_w0_einsum are equal: ", torch.div(x_w0, x_w0_einsum) -1)

        relu_x_w0=F.relu(x_w0)
        #print("This is relu_x_w0: ", relu_x_w0.shape)

        KL_w1=0.5*( self.lamb*torch.sum(sqrt_v_pri) - (self.d_h + 1) + self.lamb*torch.sum(m_pri**2) - (self.d_h + 1)*torch.log(torch.tensor(self.lamb)) - torch.log(torch.sum(sqrt_v_pri)) )

        #print("This is sqrt_v_w0 shape: ",sqrt_v_w0.shape)

        KL_w0=0.5*( self.lamb*torch.sum(sqrt_v_w0) - (data_dim + 1) + self.lamb*torch.sum(m_w0**2) - (data_dim + 1)*torch.log(torch.tensor(self.lamb)) - torch.log(torch.sum(sqrt_v_w0)) )

        KL_term= KL_w1 + KL_w0


        return m_pri, v_pri, relu_x_w0, KL_term


        
def loss_func(pred_samps, y, gam, m_pri, v_pri, KL_term):
    # print("This is pred_samps shape: ", pred_samps.shape)

    n, d_h, n_MC_samps = pred_samps.shape  

    pred_samps_bias=torch.cat((torch.ones(n, 1, n_MC_samps), pred_samps), 1) # n by (d_h+1) by MC_samps
    #print("This is pred_samps_bias shape: ", pred_samps_bias.shape)

    trm1_n=y**2
    trm1=trm1_n.repeat(1, n_MC_samps)

    #print("This is m_pri shape: ", m_pri.shape)

    m_pri_z0=torch.zeros((n,  n_MC_samps))

    for i in range(0, n):
        for j in range(0, n_MC_samps):
            #print("This is pred_samps_bias[i, :, j]: ", pred_samps_bias[i, :, j])
            #print("This is m_pri_z0: ", torch.matmul(m_pri,pred_samps_bias[i, :, j]))
            m_pri_z0[i, j]=torch.matmul(m_pri,pred_samps_bias[i, :, j])

    m_pri_z_einsum=torch.einsum('h, nhm -> nm', m_pri, pred_samps_bias)

    #print("Check if m_pri_z0 and m_pri_z_einsum are equal: ", torch.div(m_pri_z0, m_pri_z_einsum) -1)

    y_m_pri_z0=torch.zeros((n, n_MC_samps))

    for i in range(0, n):
        for j in range(0,  n_MC_samps):
            #print("This is m_pri_z0[:, m] shape: ", m
            # _pri_z0[:, m].shape)
            y_m_pri_z0[i, j]=2*y[i]*m_pri_z0[i,j]

    #trm2=torch.einsum('nl,nm -> nm', y, m_pri_z_einsum)

    #print("Check if  y_m_pri_z0 and the same with einsum are equal: ", torch.div(y_m_pri_z0, trm2) -1)
    

    trm3=torch.square(m_pri_z0)


    #print("This is  torch.square(pred_samps_bias) shape: ",  torch.square(pred_samps_bias))

    trace_trm=torch.zeros((n, n_MC_samps))

    z0_square=torch.square(pred_samps_bias)

    for i in range(0, n):
        for j in range(0,  n_MC_samps):
            trace_trm[i, j]=torch.matmul(z0_square[i, : , j ], v_pri)



    trm4=torch.einsum('nhm, h ->nm', torch.square(pred_samps_bias), v_pri)
    
    #print("Check if  trace_trm and the same with einsum are equal: ", torch.div(trace_trm, trm4) -1)



    out_vect=gam*(0.5/n_MC_samps)*(trm1 - y_m_pri_z0 + trm3 + trm4)
    out_expectation_vect=torch.sum(out_vect) +  0.5*n*torch.log(2*torch.pi/gam)

    out= out_expectation_vect + KL_term

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
    gam = 10 # noise precision for y = w*x + noise, where noise \sim N(0, 1/gam)
    epochs = 10000
    n_MC_samps = 20
    d_h=4

    np.random.seed(0)

    """ data generation """

    #Generate W1 from N(0, 1./\lamb)
    W1 = torch.randn(d_h+1,1) # num hidden units plus a bias term
    W1 = torch.sqrt(1/torch.tensor(lamb))*W1 

    #Generate W0 from N(0, 1./\lamb)
    W0=torch.sqrt(1/torch.tensor(lamb))*torch.randn(d+1,d_h) # input dim plus a bias term to hidden units 

    print('ground truth W0 is', W0.shape)
    print('ground truth W1 is', W1.shape)

    X = 0.1*torch.randn(n,d)
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
    #print("This is len_m: ", len_m)
    len_v = d_h * (d + 1)  # length of variance parameters for W_0
    len_m_pri = d_h + 1 # length of mean parameters for w_1
    len_v_pri = d_h + 1 # length of variance parameters for w_1
    init_ms = 0.1*torch.randn(len_m + len_m_pri) # initial values for all means
    init_vs = 0.1*torch.randn(len_v + len_v_pri) # initial values for all variances
    ms_vs = torch.cat((init_ms, init_vs), 0)

    model = NN_Model(len_m, len_v, len_m_pri, len_v_pri, n_MC_samps, ms_vs, device, lamb, d_h)
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    batch_size = 100
    how_many_iter = np.int(n / batch_size)

    for epoch in range(1, epochs + 1):
        model.train()

        for i in range(how_many_iter):
            # get the inputs
            inputs = X[i * batch_size:(i + 1) * batch_size, :]
            labels = y[i * batch_size:(i + 1) * batch_size]

            optimizer.zero_grad()

            m_pri, v_pri, relu_x_w0, KL_term = model(inputs)
            loss = loss_func(relu_x_w0, labels, torch.tensor(gam), m_pri, v_pri, KL_term)

            loss.backward()
            optimizer.step()

        print('Epoch {}: loss : {}'.format(epoch, loss.sum()))

        print('posterior mean of w1', m_pri)
        print('ground truth W1 is', W1.squeeze())







if __name__ == '__main__':
    main()