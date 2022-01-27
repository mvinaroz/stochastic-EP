
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
import torch
from aux_files import load_data
import argparse
import numpy as np
import torch.optim as optim

class NN_Model(nn.Module):

    def __init__(self, len_m, len_m_pri, len_v, num_samps, init_var_params, device, gamma, lamb, data_dim):
        super(NN_Model, self).__init__()
        self.parameter = Parameter(torch.Tensor(init_var_params), requires_grad=True)
        self.num_samps = num_samps
        self.device = device
        self.len_m = len_m
        self.len_m_pri = len_m_pri
        self.len_v = len_v
        self.gamma = gamma
        self.lamb = lamb
        self.data_dim = data_dim

    def forward(self, x, y): # x is mini_batch_size by input_dim

        # unpack ms_vs
        ms_vs = self.parameter
        init_m = ms_vs[0:self.len_m]
        init_m_pri = ms_vs[self.len_m:self.len_m + self.len_m_pri]
        init_v = ms_vs[self.len_m + self.len_m_pri:self.len_m + self.len_m_pri + self.len_v]
        init_v_pri = ms_vs[self.len_m + self.len_m_pri + self.len_v:]
        print("this is init_v shape: ", init_v.shape[0])

        #reshape vars into (d_h, dim +1)
        reshape_v_init=torch.reshape(init_v, (init_v_pri.shape[0], self.data_dim+1 ))
        #reshape means into (d_h, dim +1)
        reshape_m_init=torch.reshape(init_m, (init_m_pri.shape[0], self.data_dim+1 ))
        #print('This is reshape_v_init ', reshape_v_init)


        # to do : implement actual loss here using samples drawn from the posterior, i.e., eq(24)
        #le's draw W_0 samples
        W_0=torch.zeros(init_v.shape[0])
        for elem in range(init_v.shape[0]):
            W_0[elem]=np.random.normal(init_m[elem].item(), init_v[elem].item()) #Add size= argument to draw the desired samples (L).
        reshape_W_0=torch.reshape(W_0, (init_m_pri.shape[0], self.data_dim+1 ))

        for x_n in range(x.shape[0]):
            print("This is datapoint x_n shape: ", x[x_n].shape)
        

        print("This is the susampled W_0: ", reshape_W_0.shape)




        # to do : add KL term, i.e., eq(20)

        #Sum over columns(data dim) lambda*v_ij
        prod_lamb_v= self.lamb*reshape_v_init
        sum_lamb_v=torch.sum(prod_lamb_v, 1) #Shape: d_h

        #lambda*m_i.T m_i
        means_prod=torch.einsum('ij,ji->i', reshape_m_init, reshape_m_init.T)
        lamb_means_prod=self.lamb*means_prod

        #Sum over columns(data dim) log(1/lambda*v'_ij)
        inv_lambv_ij=torch.div(torch.ones(init_m_pri.shape[0], self.data_dim+1), prod_lamb_v)
        log_inv_lambv_ij=torch.log(inv_lambv_ij)
        sum_log_inv_lambv_ij=torch.sum(prod_lamb_v, 1) #Shape: d_h
        

        kl_term1= 0.5*(sum_lamb_v + lamb_means_prod + sum_log_inv_lambv_ij - self.data_dim)
        kl_term1=torch.sum(kl_term1) 
        print("this is kl_term1: ", kl_term1)

        #Sum lambda*v'_i
        lamb1=self.lamb*torch.ones(self.len_m_pri)
        #print("this is lamb shape: ", lamb1)

        mult_lamb_vi=torch.matmul(lamb1, init_v_pri)

        #lamda*m'.T m'
        mult_m_pri=self.lamb*torch.matmul(init_m_pri, init_m_pri)

        #Sum log(1/lambda*v'_i)
        aux=self.lamb*init_v_pri
        inv_lambv=torch.div(torch.ones(self.len_m_pri), aux)
        log_inv_lambv=torch.log(inv_lambv)
        sum_log_inv_lambv=torch.sum(log_inv_lambv)

        kl_term2=0.5*(mult_lamb_vi + mult_m_pri +  sum_log_inv_lambv -  self.len_m_pri)
        print("this is kl_term2: ", kl_term2)


        pred_samps=0
        KL_term=kl_term1 + kl_term2
        print("this is KL_term: ", KL_term)

        return pred_samps, KL_term

    def sigmoid(X):
        return 1/(1+np.exp(-X))


def loss(pred_samps, KL_term, y, gam, lamb, num_samps):

    # write eq.(18) here
    out = 0


    return out


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='naval', \
                        help='choose the data name among naval, robot, power, wine, protein')

    # OPTIMIZATION
    parser.add_argument('--n-hidden', type=int, default=50, help='number of hidden units in the layer')
    parser.add_argument("--batch-size, --bs", type=float, default=200)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument("--normalize-data", action='store_true', default=True)

    ar = parser.parse_args()

    return ar


def main():
    ar = get_args()
    np.random.seed(ar.seed)
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'

    # Load data.
    X_train, X_test, y_train, y_test = load_data(ar.data_name, ar.seed)
    print("this is y_train shape: ", y_train.shape)

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


    # model specs and loss

    n, d = X_train.shape  # input dimension of data, depending on the dataset
    print("This is data dimension: ", d)
    d_h = 50  # number of hidden units in the hidden layer
    # define the length of variational parameters
    len_m = d_h * (d + 1)  # length of mean parameters for W_0, where the size of W_0 is d_h by (d+1)
    print("This is len_m: ", len_m)
    len_v = d_h * (d + 1)  # length of variance parameters for W_0
    len_m_pri = d_h  # length of mean parameters for w_1, where the size of w_1 is d_h
    len_v_pri = d_h  # length of variance parameters for w_1
    init_ms = torch.randn(len_m + len_m_pri, requires_grad=True) # initial values for all means
    init_vs = torch.rand(len_v + len_v_pri, requires_grad=True) # initial values for all variances, can't be negative
    ms_vs = torch.cat((init_ms, init_vs), 0)

    # these hyperparameters gamma and lambda are taken from PBP results
    gam = 0.1
    lamb = 0.2
    num_samps = 10

    model = NN_Model(len_m, len_m_pri, len_v, num_samps, ms_vs, device, gam, lamb, d)
    # optimizer = optim.SGD(importance.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training routine should start here.
    # in every training step, you compute this
    pred_samps, KL_term = model(torch.tensor(X_train), y_train) # some portion of X_train if mini-batch learning is happening

    output = loss(pred_samps, KL_term, torch.tensor(y_train), gam, lamb, num_samps)
    #output.backward()

    print('output: ', output)


    # # We obtain the test RMSE and the test ll
    # m, v, v_noise = pbp_instance.get_predictive_mean_and_variance(X_test)
    # rmse = np.sqrt(np.mean((y_test - m) ** 2))
    # print("rmse: ", rmse)
    # test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
    #                   0.5 * (y_test - m) ** 2 / (v + v_noise))
    # print("test_ll: ", test_ll)


if __name__ == '__main__':
    main()
