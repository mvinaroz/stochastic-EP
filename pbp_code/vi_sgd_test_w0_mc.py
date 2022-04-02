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

    def __init__(self, len_m, len_v, num_samps, init_var_params, device, lamb, data_dim):
                       #len_m, len_m_pri, len_v, num_samps, ms_vs, device, gam, lamb, d
        super(NN_Model, self).__init__()
        self.parameter = Parameter(torch.Tensor(init_var_params), requires_grad=True)
        self.num_MC_samps = num_samps
        self.device = device
        self.len_m = len_m
        self.len_v = len_v
        self.lamb = lamb
        self.data_dim = data_dim
        self.relu = F.relu


    def forward(self, x): # x is mini_batch_size by input_dim

        # unpack ms_vs
        ms_vs = self.parameter

        m_pri = ms_vs[0:self.len_m]
        #print("This is m_pri shape: ", m_pri)

        # because these variances have to be non-negative
        v_pri =F.softplus(ms_vs[self.len_m:])
        #print("This is v_pri shape: ", v_pri)


        #Generate MC samples to approx w0

        samps_from_standard_normal = torch.randn(self.len_m, self.num_MC_samps)
        samps_std_adjusted = torch.einsum('i,ik ->ik', torch.sqrt(v_pri), samps_from_standard_normal)
        samples_W_0 = m_pri[:,None].repeat(1, samps_std_adjusted.shape[1]) + samps_std_adjusted   #Has shape (d+1)x MC_samps

        W_0=samples_W_0[0:self.data_dim, :]
        bias = samples_W_0[-1,:]
        #print("This is W_0: ", W_0.shape)
        #print("This is bias: ", bias.shape)
        #print("This is x.shape: ", x.shape)

        x_W_0 = torch.einsum('nd,dk -> nk', x, W_0)
        #print("This is x_W_0.shape: ", x_W_0.shape)

        x_W_0_plus_bias = torch.einsum('nk,k -> nk', x_W_0, bias)
        #print("This is x_W_0_plus_bias.shape: ",x_W_0_plus_bias.shape)


        return x_W_0_plus_bias

def loss_func(pred_samps, y, gam):

    #print("This is pred_samps shape: ", pred_samps.shape)

    n=pred_samps.shape[0]

    trm1 = torch.sum(y**2)

    y_pred_samps= torch.einsum('n, nk -> nk', y, pred_samps) # N by MC_samps (y*W0_xn)
    trm2=torch.mean(torch.sum(2*y_pred_samps, 0))

    trm3 = torch.mean(torch.sum(pred_samps**2,0)) #W0_xn**2

    trm4 = 0.5*n*torch.log(2*torch.pi/gam)

    out=gam*0.5*(trm1 - trm2 + trm3) - trm4

    print("This is out: ", out)

    return out




def generate_data(n, data_dim, lamb, gam):

    #Generate data features from N(0,I)

    X=np.random.randn(n, data_dim)

    #Generate W0 from N(0, 1/lamb)
    w_0=np.random.randn(data_dim)*np.sqrt(1./lamb) 

    #Generate noise from N(0, 1/gam)
    noise=np.random.randn(n)*np.sqrt(1./gam) 

    y=np.einsum('ij,j -> i', X, w_0) + noise
    print("This is y: ", y.shape)

    return X, y


def main():

    d=4
    n=10000
    n_test=1000
    lamb=100 #weights precision
    gam=10 #noise precision
    d_h=50
    num_samps=500 #MC samples
    epochs=10
    normalize_data=False
    v_noise =  (1./gam) 

    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_train, y_train = generate_data(n, d, lamb, gam)
    X_test, y_test=generate_data(n_test, d, lamb, gam)

    len_m = d + 1  # length of mean parameters for W_0: (d+1)
    len_v = d + 1  # length of variance parameters for W_0

    init_ms = 0.01*torch.randn(len_m) # initial values for all means
    init_vs = 0.01*torch.randn(len_v) # initial values for all variances
    ms_vs = torch.cat((init_ms, init_vs), 0)

    model = NN_Model(len_m, len_v, num_samps, ms_vs, device, lamb, d)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred_samps= model(torch.Tensor(X_train))
        loss = loss_func(pred_samps, torch.Tensor(y_train), torch.tensor(gam))

        loss.backward()
        optimizer.step()

        print('Epoch {}: loss : {}'.format(epoch, loss.sum()))


    
if __name__ == '__main__':
    main()
