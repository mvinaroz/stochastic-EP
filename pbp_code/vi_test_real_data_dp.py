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
import math
from aux_files import load_data

from autodp import rdp_acct, rdp_bank

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


def kl_div_term(m0, v0, lamb):

    # KL(N0, N1) = 0.5*(TR(\precision1 \cov0) + (\mu1 - \mu0).T \precision1 (\mu1 - \mu0) -d + ln( det(\cov1) / det(\cov0) ) )
    #print("This is v0: ", v0)
    cov0=torch.diag(v0)
    #print("This is cov0: ", cov0)
    cov1=torch.eye(v0.shape[0])*(1/torch.tensor(lamb))
    
    precision1=torch.linalg.inv(cov1)

    m1=torch.zeros(m0.shape)

    trace_trm=torch.trace(torch.matmul(precision1, cov0))
    #print("This is trace_term: ", trace_trm)

    mu_diff=m1 - m0
    #print("This is m1 - m0: ", mu_diff)
    squared_term=torch.matmul(torch.matmul(mu_diff.T, precision1), mu_diff)
    #print("This is squared_term: ", squared_term)

    d=m0.shape[0]
    #print("This is dim in kl:", d)

    
    log_term=torch.log(torch.det(cov1))  - torch.log(torch.det(cov0))
    #print("This is torch.det(cov1): ", torch.det(cov1))
    #print("This is torch.log(torch.det(cov1)): ", torch.log(torch.det(cov1)))
    #print("This is torch.log(torch.det(cov0)): ", torch.log(torch.det(cov0)))
    #print("This is torch.det(cov1) / torch.det(cov0): ", torch.det(cov1) / torch.det(cov0))
    #print("This is log_term: ", log_term)

    kl_div=0.5*(trace_trm + squared_term - d + log_term )

    return kl_div


def kl_div_univariate(m_0, v_0, lamb):
    # KL(N0, N1) =0.5*((std0/std1)**2 + (mu1-mu0)**2/std1**2 - 1 + log(std1/std0))

    std0=torch.sqrt(v_0)
    std1=torch.sqrt(1/torch.tensor(lamb))

    mu1=torch.zeros(v_0.shape)

    term1=(std0/std1)**2
    mu_diff=(mu1 - m_0)**2

    term2=mu_diff/(std1**2)
    
    log_term=torch.log(std1) - torch.log(std0)
    kl_term= 0.5*(term1 + term2 - 1 + log_term)

    return  kl_term

        
def loss_func(pred_samps, y, gam, kl_div):

    n, n_MC_samps_w0, n_MC_samps_w1 = pred_samps.shape  

    out1=torch.zeros((n,  n_MC_samps_w0, n_MC_samps_w1))


    for j in range(0, n_MC_samps_w0):
        for k in range(0, n_MC_samps_w1):

            out1[:,j,k]=0.5*gam*(y.squeeze()- pred_samps[:,j,k])**2


    out=torch.mean(out1) + 0.5*torch.log(2*torch.pi/gam)

    return out + kl_div

def privacy_param_func(sigma, delta, n_epochs, batch_size, n_data):
    """ input arguments """
    
    k = n_epochs  # k is the number of steps during the entire training
    prob = batch_size / n_data  # prob is the subsampling probability

    """ end of input arguments """

    """ now use autodp to calculate the cumulative privacy loss """
    # declare the moment accountants
    acct = rdp_acct.anaRDPacct()

    # define the functional form of uppder bound of RDP
    func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)

    eps_seq = []
    print_every_n = 100
    for i in range(1, k + 1):
        acct.compose_subsampled_mechanism(func, prob)
        eps_seq.append(acct.get_eps(delta))
        if i % print_every_n == 0 or i == k:
            print("[", i, "]Privacy loss is", (eps_seq[-1]))

    print("The final epsilon delta values after the training is over: ", (acct.get_eps(delta), delta))

    final_epsilon = acct.get_eps(delta)
    return final_epsilon


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='wine', \
                        help='choose the data name among naval, robot, power, wine, protein')

    # OPTIMIZATION
    parser.add_argument('--n-hidden', type=int, default=50, help='number of hidden units in the layer')
    parser.add_argument('--batch-size', '-bs', type=int, default=100, help='batch size during training')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument("--normalize-data", action='store_true', default=True)
    parser.add_argument('--lr', type=float, default=1e-2,  help='learning rate' )
    parser.add_argument('--lamb', type=float, default=200,  help='precision on weights' )
    parser.add_argument('--gam', type=float, default=2,  help='precision on noise' )
    parser.add_argument('--mc-samps-w0', type=int, default=10,  help='number of mc samples to generate for w0' )
    parser.add_argument('--mc-samps-w1', type=int, default=10,  help='number of mc samples to generate for w1' )
    parser.add_argument('--beta', type=float, default=0.01,  help='' )

    parser.add_argument('--clip', type=float, default=10,  help='' )
    parser.add_argument('--delta', type=float, default=1e-5,  help='' )
    parser.add_argument('--sigma', type=float, default=30,  help='' )





    ar = parser.parse_args()

    return ar


def main():

    ar = get_args()
    print(ar)
    np.random.seed(ar.seed)
    torch.manual_seed(ar.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    d_h=ar.n_hidden
    n_MC_samps_w0=ar.mc_samps_w0
    n_MC_samps_w1=ar.mc_samps_w1
    lamb=ar.lamb
    gam=ar.gam
    epochs=ar.epochs
    batch_size=ar.batch_size
    c=ar.clip
    sigma=ar.sigma
    delta=ar.delta



    """Load data"""
    # Load data.
    X_train, X_test, y_train, y_test = load_data(ar.data_name, ar.seed)

     # Normalizing data
    if ar.normalize_data:
        print('normalizing the data')

        std_X_train = np.std(X_train, 0)
        std_X_train[std_X_train == 0] = 1
        mean_X_train = np.mean(X_train, 0)

        X_train = (X_train - np.full(X_train.shape, mean_X_train)) / \
                  np.full(X_train.shape, std_X_train)

        X_test = (X_test - np.full(X_test.shape, mean_X_train)) / \
                 np.full(X_test.shape, std_X_train)

        mean_y_train = np.mean(y_train)
        std_y_train = np.std(y_train)

        y_train = (y_train - mean_y_train) / std_y_train


    else:
        print('testing non-standardized data')

    v_noise =  (1./ar.gam) * std_y_train ** 2 #the variance for the noise and std_y_train ** 2 is because it's reescaling it.

    print("This is v_noise: ", v_noise)

    n, d = X_train.shape
    n_test=y_test.shape[0]

    X_train=torch.from_numpy(X_train)
    X_train=X_train.to(torch.float32)
    X_test=torch.from_numpy(X_test)
    X_test=X_test.to(torch.float32)

    print("This is X_train shape: ", X_train.shape)

    y_train=torch.from_numpy(y_train)
    y_test=torch.from_numpy(y_test)
    
    X_train=torch.cat((torch.ones(n,1), X_train), 1)
    X_test=torch.cat((torch.ones(n_test,1), X_test), 1)

    """Compute epsilon for a given gamma"""
    final_epsilon=privacy_param_func(sigma , delta, epochs, batch_size, n)

    """initialize model and it's parameters"""
    len_m = d_h * (d + 1)  # length of mean parameters for W_0, where the size of W_0 is d_h by (d+1)
    len_v = d_h * (d + 1)  # length of variance parameters for W_0
    len_m_pri = d_h + 1 # length of mean parameters for w_1
    len_v_pri = d_h + 1 # length of variance parameters for w_1
    init_ms = 1*torch.randn(len_m + len_m_pri) # initial values for all means
    init_vs = 1*torch.randn(len_v + len_v_pri) # initial values for all variances
    #init_vs = 1*torch.ones(len_v + len_v_pri)
    ms_vs = torch.cat((init_ms, init_vs), 0)


    #print("This is ms_vs: ", ms_vs.shape)

    model = NN_Model(len_m, len_v, n_MC_samps_w0, n_MC_samps_w1, ms_vs, device, lamb, d_h)
    optimizer = optim.SGD(model.parameters(), lr=ar.lr)


    how_many_iter = np.int(n / batch_size)

    for epoch in range(1, epochs + 1):
        model.train()

        for i in range(how_many_iter):
            # get the inputs
            inputs = X_train[i * batch_size:(i + 1) * batch_size, :]
            labels = y_train[i * batch_size:(i + 1) * batch_size]

            optimizer.zero_grad()

            pred_samps, m_w0, v_w0, m_w1, v_w1= model(inputs)

            kl_w1_dh=torch.zeros(d_h + 1)
            for a in range(0, d_h + 1):
                kl_w1_dh[a]=kl_div_univariate(m_w1[a], v_w1[a], lamb)
            
            kl_w1=torch.sum(kl_w1_dh)
            #print("This is kl_w1: ", kl_w1)
            #kl_w1=0

            kl_w0_dh=torch.zeros(m_w0.shape[0])

            for dim in range(0, m_w0.shape[0]):
                kl_w0_dh[dim]=kl_div_univariate(m_w0[dim], v_w0[dim], lamb)

            kl_w0=torch.sum(kl_w0_dh)
            #kl_w0=0
            #print("This is kl_w0: ", kl_w0)
            kl_term=(ar.beta/batch_size)*(kl_w0 + kl_w1)
            #kl_term=0

            loss_list=[]
            accumulated_grads=[]

            for item in range(0, batch_size):
                #print("This is pred_samps[item]: ", labels[item])
                loss_item=loss_func( torch.unsqueeze(pred_samps[item], 0), labels[item], torch.tensor(gam), kl_term)
                loss_list.append(loss_item)
                loss_item.backward(retain_graph=True)


                # Clip each parameter's per-sample gradient
                for p in model.parameters():
                    per_sample_grad = p.grad.detach().clone()
                    #print("The model params grads before clipping: ", per_sample_grad)
                    torch.nn.utils.clip_grad_norm_(per_sample_grad, max_norm=c)
                    #print("The model params grads after clipping: ", p.grad)
                    accumulated_grads.append(per_sample_grad)
                model.zero_grad() # p.grad is accumulative, so we need to manually reset

            # Aggregate clipped gradients of all samples in a batch, and add DP noise
            clip_gradients=torch.stack(accumulated_grads)
            #print("This is accumulated_grads: ", clip_gradients.shape)

            aggregated_grad=torch.sum(clip_gradients, 0) 
            #print("This is aggregated_grad: ", aggregated_grad.shape)
            
            gaussian_noise=torch.randn(aggregated_grad.shape)*c*sigma

            

            priv_grads=  (aggregated_grad + gaussian_noise)/batch_size
            
            for p in model.parameters():
                p.grad=priv_grads
                #print("This is the priv_grads: ", priv_grads)

            optimizer.step()

        loss=torch.stack(loss_list)

        print('Epoch {}: loss : {}'.format(epoch, loss.sum()))

        """Compute y predicted from model parameters to compare it with y_test"""

        #print("This are the mean and std for y: ,", torch.mean(y), torch.std(y))
        #print("This are the mean and std for y_pred: ,", torch.mean(y_pred), torch.std(y_pred))

        pred_samps_test, m_w0_test, v_w0_test, m_w1_test, v_w1_test = model(X_test)

        W0_test=torch.randn((d+1)*d_h)*torch.sqrt(v_w0_test) +  m_w0_test
        W0_test=torch.reshape(W0_test, (d + 1, d_h))

        x_w0_test=torch.mm(X_test,  W0_test)

        z0_test=F.relu(x_w0_test)
        z0_test=torch.cat((torch.ones(n_test,1), z0_test), 1)

        w1_test=torch.randn(d_h + 1)*torch.sqrt(v_w1_test) + m_w1_test

        z1_test=torch.matmul(z0_test,  w1_test) 

        #print("This is pred_samps_test: ", pred_samps_test.shape)

        m_prd = (torch.mean(pred_samps_test, (1, 2))).detach().numpy()
        v_prd = (torch.var(pred_samps_test, (1, 2))).detach().numpy()

        if ar.normalize_data:
            m_prd = m_prd * std_y_train + mean_y_train
            v_prd = v_prd * std_y_train ** 2

        y_test_numpy=y_test.detach().numpy()
    

        y_test_pred=torch.randn(n_test)*torch.sqrt(1/torch.tensor(gam)) + z1_test

        test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v_prd + v_noise)) - \
                          0.5 * ( y_test_numpy - m_prd) ** 2 / (v_prd + v_noise))
        print("test_ll: ", test_ll)

        rmse = torch.sqrt(torch.mean((y_test - m_prd) ** 2))
        print("rmse: ", rmse)

 

        
    #matplotlib.pyplot.figure(figsize = [10, 5]) # larger figure size for subplots

    # example of somewhat too-large bin size
    #matplotlib.pyplot.subplot(1, 2, 1) # 1 row, 2 cols, subplot 1

    #matplotlib.pyplot.hist(y_test.squeeze().detach().numpy(), bins=20)
    #matplotlib.pyplot.xlabel('y_test')

    # example of somewhat too-small bin size
    #matplotlib.pyplot.subplot(1, 2, 2) # 1 row, 2 cols, subplot 2


    #matplotlib.pyplot.hist(y_test_pred.squeeze().detach().numpy(),  bins=20)
    #matplotlib.pyplot.xlabel('y_test_pred')
    #matplotlib.pyplot.show()





if __name__ == '__main__':
    main()