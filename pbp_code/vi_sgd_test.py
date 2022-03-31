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

    def __init__(self, len_m, len_m_pri, len_v, num_samps, init_var_params, device, lamb, data_dim):
                       #len_m, len_m_pri, len_v, num_samps, ms_vs, device, gam, lamb, d
        super(NN_Model, self).__init__()
        self.parameter = Parameter(torch.Tensor(init_var_params), requires_grad=True)
        self.num_MC_samps = num_samps
        self.device = device
        self.len_m = len_m
        self.len_m_pri = len_m_pri
        self.len_v = len_v
        self.lamb = lamb
        self.data_dim = data_dim
        self.relu = F.relu


    def forward(self, x): # x is mini_batch_size by input_dim

        # unpack ms_vs
        ms_vs = self.parameter
        m = ms_vs[0:self.len_m]
        m_pri = ms_vs[self.len_m:self.len_m + self.len_m_pri]
        # because these variances have to be non-negative
        v_no_softplus=ms_vs[self.len_m + self.len_m_pri:self.len_m + self.len_m_pri + self.len_v]
        v_pri_no_softplus=ms_vs[self.len_m + self.len_m_pri + self.len_v:]
        with torch.no_grad(): 
            v_no_softplus[ v_no_softplus <= 0] = 1e-6
            v_pri_no_softplus[ v_pri_no_softplus <= 0] = 1e-6
        #print("check if v before softplus contains 0: ", 0 in ms_vs[self.len_m + self.len_m_pri:self.len_m + self.len_m_pri + self.len_v])
        #print("check if v contains negative values before softplus: ", v_no_softplus[v_no_softplus <= 0])
        v =F.softplus(ms_vs[self.len_m + self.len_m_pri:self.len_m + self.len_m_pri + self.len_v])
        #print("check if v from softplus contains 0: ", v[v <= 0])
        v=torch.clamp(v, min=1e-6)
        #print("check if v after clamping contains 0: ", 0 in v)
        v_pri = F.softplus(ms_vs[self.len_m + self.len_m_pri + self.len_v:])
        #print("check if v_pri from softplus contains 0: ", 0 in v_pri)
        #v_pri=torch.clamp(v_pri, min=1e-6)
        #print("check if v_pri after clamping contains 0: ", 0 in v_pri)


        d_h = int(self.len_m / (self.data_dim + 1))
        #print("This is d_h: ", d_h)

        samps_from_standard_normal = torch.randn(self.len_m, self.num_MC_samps)
        samps_std_adjusted = torch.einsum('i,ik ->ik', torch.sqrt(v), samps_from_standard_normal)
        #print("This is sqrt(v): ", torch.sqrt(v))
        #print("This is samps_from_std_adjusted ", torch.std(samps_std_adjusted, dim=0))
        samples_W_0 = m[:,None].repeat(1,samps_std_adjusted.shape[1]) + samps_std_adjusted # size = (d_h * (self_data_dim + 1)) by self.num_MC_samps
        #print("this is m[:,None].repeat(1,samps_std_adjusted.shape[1]) : ", m[:,None].repeat(1,samps_std_adjusted.shape[1]))
        #print("this is std  samples_W_0 ", torch.std(samples_W_0))
        #print("this is mean  samples_W_0 ", torch.mean(samples_W_0))
        #print("This is sqrt  v_no_softplus: ",  torch.sqrt(v_no_softplus))

        #print("This is std  v: ",  torch.std(v))
        
        #print("This is v: ",  v.shape)


        W_0 = torch.reshape(samples_W_0[0:-d_h,:], (d_h, self.data_dim, self.num_MC_samps)) # d_h  by self.data_dim by self.num_MC_samps
        bias = samples_W_0[-d_h:,:] # d_h by self.num_MC_samps

        x_W_0 = torch.einsum('nd,jdk -> njk', x, W_0)
        x_W_0_plus_bias = torch.einsum('njk,jk -> njk', x_W_0, bias)
        pred_samps =  self.relu(x_W_0_plus_bias) # num data samples by d_h by num MC samples (N  time d_h times L)

        ### KL term
        #print("This is m.shape: ", m.shape)
        #print("This is torch.sum(m**2): ", torch.sum(m**2))
        #print("This is torch.mul(m.T, m): ", torch.matmul(m, torch.t(m)))
        trm1 = 0.5*(self.lamb*torch.sum(v) + self.lamb*torch.sum(m**2) - (self.data_dim + 1) - self.data_dim*np.log(self.lamb) - torch.sum(v))
        trm2 = 0.5*(self.lamb*torch.sum(v_pri) + self.lamb*torch.sum(m_pri**2) - d_h - d_h*np.log(self.lamb) - torch.sum(v_pri))
        KL_term = trm1 + trm2

        #print("This is torch.sum(v): ", torch.sum(v))
        #print("This is torch.sum(m**2): ", torch.sum(m**2))
        #print("check if self.lamb*v contains 0: ", 0 in v)
        #print("This is torch.log(self.lamb*v)): ", torch.log(self.lamb*v))
        #print("This is -torch.sum(torch.log(self.lamb*v))): ", -torch.sum(torch.log(self.lamb*v)))
        print("This is KL term from W0: ", trm1)
        print("This is KL term from w1: ", trm2)
        
        #print("This is KL div term: ", KL_term )

        return pred_samps, KL_term, m_pri, v_pri

def loss_func(pred_samps, KL_term, y, gam, data_dim, m_pri, v_pri, d_h):

    # size(pred_samps) = num data samples by d_h by num MC samples
    n = pred_samps.shape[0]

    #print("This is data_dim:", data_dim)

    # m_pri times pred_samps
    m_pri_pred_samps = torch.einsum('j, njk -> nk', m_pri, pred_samps) # N by MC_samps
    #y_m_pri_pred_samps = torch.einsum('n, nk -> nk', y, m_pri_pred_samps) # N by MC_samps
    y_m_pri_pred_samps2 = torch.einsum('n, nk -> k', y, m_pri_pred_samps) # MC_samps
    test=torch.matmul(y, m_pri_pred_samps)
    #print("This is test : ", test)
    #print("This is torch.sum(2*y_m_pri_pred_samps, 0) shape: ", torch.sum(2*y_m_pri_pred_samps, 0).shape)
    #print("This is y_m_pri_pred_samps2 shape: ", y_m_pri_pred_samps2.shape)
    #print("This is 2*torch.matmul(y, m_pri_pred_samps): ",  2*torch.matmul(y, m_pri_pred_samps))
    #print("Check torch.sum(2*y_m_pri_pred_samps, 0) and 2*torch.matmul(y, m_pri_pred_samps) are equal: ", torch.eq(torch.sum(y_m_pri_pred_samps, 0),  test))
    print("Check 2*torch.einsum('n, nk -> k', y, m_pri_pred_samps) and 2*torch.matmul(y, m_pri_pred_samps) are equal: ",  torch.all(torch.eq(y_m_pri_pred_samps2, test)))
    if not torch.all(torch.eq(2*y_m_pri_pred_samps2, 2*test)):
        print("This is test : ", test)
        print("This is y_m_pri_pred_samps2 : ", y_m_pri_pred_samps2)
    else:
        pass
    trm1 = torch.sum(y**2)
    print("This is trm1: ", trm1)
    #trm2 = torch.mean(torch.sum(2*y_m_pri_pred_samps, 0))
    trm2=torch.mean(2*y_m_pri_pred_samps2)
    print("This is trm2: ", trm2)
    
    trm3 = torch.mean(torch.sum(torch.einsum('njk,j -> nk', pred_samps**2, v_pri),0)) #Sums over number of datapoints and mean wrt the MC_samps.

    #print("This is pred_samps: ", pred_samps)  #Has shape (N, d_h, MC_samps)
    #print("This is pred_samps**2: ", (pred_samps**2)) #Has shape (N, d_h, MC_samps) 

    #torch.einsum('njk,j -> nk', pred_samps**2, v_pri) computes the TRACE[(sigma(W0*x_n)sigma(W0*x_n).T)(V')]
   
    print("This is trm3: ", trm3)

    trm4 = torch.mean(torch.sum(m_pri_pred_samps**2,0))

    trm5 = 0.5*n*data_dim*torch.log(2*torch.pi/gam)

    out1 = gam*0.5*(trm1 - trm2 + trm3 + trm4) + trm5
    out = out1 + KL_term
    # print("This is out: ", out)

    print("This is the Expectation term: ", out1)

    return out

def generate_data(n, data_dim, n_hidden, lamb, gam):

    #Generate data features from N(0,I)

    X_train=np.random.randn(n, data_dim)

    #Weights true mean
    len_W0=(data_dim+1)*n_hidden
    len_w1=n_hidden
    w_true_mean=np.random.normal(0, 1, len_W0 + len_w1) 
    #print("This is w_true_mean: ", w_true_mean)
    #print("This is w_true_mean repeated: ", np.tile(w_true_mean, (n, 1)).shape)

    #Generate true W0 and w1 (n by len_W0 + len w1)
    w_i_true_std_adjusted=np.random.randn(n, len_W0 + len_w1)*np.sqrt(1./lamb) 
    w_i_true=w_i_true_std_adjusted + np.tile(w_true_mean, (n, 1))
    #+ np.tile(w_true_mean, (n, 1))
    #print("This is mean w_true_mean: ", np.mean(w_true_mean))
    print("This is mean w_i_true: ", np.mean(w_true_mean))
    print("This is var w_i_true: ", np.var(w_i_true))

    w_i_0_plus_bias=np.reshape(w_i_true[:, 0:len_W0],(n, data_dim + 1, n_hidden))
    w_i_0_true=w_i_0_plus_bias[:, 0:data_dim, :]
    print("This is mean  w_i_0_true: ", np.mean(w_i_0_true))
    print("This is var  w_i_0_true: ", np.var(w_i_0_true))
    bias=w_i_0_plus_bias[:, -1, :]
    print("This is mean  bias: ", np.mean(bias))
    print("This is std  bias: ", np.std(bias))
    #print(w_i_0_true.shape)
    #print(bias.shape)
    w_i_1_true=w_i_true[:, len_W0:]
    #print(w_i_1_true.T.shape)

    x_W0=np.einsum('ij,ijk -> ik', X_train, w_i_0_true) 
    #print("This is x_W0_true: ", x_W0.shape)
    X_W0_bias=x_W0 + bias
    relu_act=np.maximum(X_W0_bias, 0)
    #print(relu_act)
    mean_y=np.einsum('dn,nd -> n', w_i_1_true.T, relu_act) #w1.T * sig(W0*x + b)
    #Now we draw the labels from the model (N(y_i | f_{w_i}(x_i, gamma^{-1})))
    y_labels=np.random.randn(n)*np.sqrt(1./gam) +  mean_y
    #print(y_labels)

    return X_train, y_labels


def main():

    d=4
    n=10000
    n_test=1000
    lamb=100 #weights precision
    gam=10 #noise precision
    d_h=50
    num_samps=500 #MC samples
    epochs=4
    normalize_data=False

    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_train, y_train = generate_data(n, d, d_h, lamb, gam)
    X_test, y_test=generate_data(n_test, d, d_h, lamb, gam)

    if normalize_data:
        print("We are normalizing the data")
        std_X_train = np.std(X_train, 0)
        std_X_train[std_X_train == 0] = 1
        mean_X_train = np.mean(X_train, 0)

        X_train = (X_train - np.full(X_train.shape, mean_X_train)) / \
                np.full(X_train.shape, std_X_train)

        X_test = (X_test - np.full(X_test.shape, mean_X_train)) / \
                np.full(X_test.shape, std_X_train)

        mean_y_train = np.mean(y_train)
        std_y_train = np.std(y_train)
        print("This is std_y_train: ", std_y_train)

        y_train = (y_train - mean_y_train) / std_y_train
    else:
        print("We aren't normalizing the data")
        

    len_m = d_h * (d + 1)  # length of mean parameters for W_0, where the size of W_0 is d_h by (d+1)
    #print("This is len_m: ", len_m)
    len_v = d_h * (d + 1)  # length of variance parameters for W_0
    len_m_pri = d_h  # length of mean parameters for w_1, where the size of w_1 is d_h
    len_v_pri = d_h # length of variance parameters for w_1
    init_ms = 0.01*torch.randn(len_m + len_m_pri) # initial values for all means
    init_vs = 0.01*torch.randn(len_v + len_v_pri) # initial values for all variances
    ms_vs = torch.cat((init_ms, init_vs), 0)
    
    model = NN_Model(len_m, len_m_pri, len_v, num_samps, ms_vs, device, lamb, d)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    if normalize_data:
        v_noise =  (1./gam) * std_y_train ** 2 #the variance for the noise and std_y_train ** 2 is because it's reescaling it.
    else:
        v_noise =  (1./gam)

    print("This is v_noise: ", v_noise)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred_samps, KL_term, m_pri, v_pri = model(torch.Tensor(X_train))
        loss = loss_func(pred_samps, KL_term, torch.Tensor(y_train), torch.tensor(gam), d, m_pri, v_pri, d_h)

        loss.backward()
        optimizer.step()

        print('Epoch {}: loss : {}'.format(epoch, loss.sum()))

        #### testing in every epoch ####
        #pred_samps_y_tst, KL_term, m_pri, v_pri = model(torch.Tensor(X_test))

        #samps_from_standard_normal = torch.randn(d_h, num_samps)
        #samps_std_adjusted = torch.einsum('i,ik ->ik', torch.sqrt(v_pri), samps_from_standard_normal)
        #samples_w_1 = m_pri[:, None].repeat(1, samps_std_adjusted.shape[1]) + samps_std_adjusted

        #w_1_times_pred_samps_y_tst = torch.einsum('jk,  njk -> nk', samples_w_1, pred_samps_y_tst)
        #m_prd = (torch.mean(w_1_times_pred_samps_y_tst, 1)).detach().numpy()
        #v_prd = (torch.var(w_1_times_pred_samps_y_tst, 1)).detach().numpy()

        #if normalize_data:
        #    m_prd = m_prd * std_y_train + mean_y_train #Reescale due to normalization of data.
        #    v_prd = v_prd * std_y_train ** 2 #Reescale due to normalization of data.

        #test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v_prd + v_noise)) - \
        #                  0.5 * (y_test - m_prd) ** 2 / (v_prd + v_noise))
        #print("test_ll: ", test_ll)

        #rmse = np.sqrt(np.mean((y_test - m_prd) ** 2))
        #print("rmse: ", rmse)





if __name__ == '__main__':
    main()