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
        v = F.softplus(ms_vs[self.len_m + self.len_m_pri:self.len_m + self.len_m_pri + self.len_v])
        v_pri = F.softplus(ms_vs[self.len_m + self.len_m_pri + self.len_v:])

        d_h = int(self.len_m / (self.data_dim + 1))

        samps_from_standard_normal = torch.randn(self.len_m, self.num_MC_samps)
        samps_std_adjusted = torch.einsum('i,ik ->ik', torch.sqrt(v), samps_from_standard_normal)
        samples_W_0 = m[:,None].repeat(1,samps_std_adjusted.shape[1]) + samps_std_adjusted # size = (d_h * (self_data_dim + 1)) by self.num_MC_samps
        W_0 = torch.reshape(samples_W_0[0:-d_h,:], (d_h, self.data_dim, self.num_MC_samps)) # d_h  by self.data_dim by self.num_MC_samps
        bias = samples_W_0[-d_h:,:] # d_h by self.num_MC_samps

        x_W_0 = torch.einsum('nd,jdk -> njk', x, W_0)
        x_W_0_plus_bias = torch.einsum('njk,jk -> njk', x_W_0, bias)
        pred_samps =  self.relu(x_W_0_plus_bias) # num data samples by d_h by num MC samples (N  time d_h times L)

        ### KL term
        trm1 = 0.5*(self.lamb*torch.sum(v) + self.lamb*torch.sum(m**2) - self.data_dim - torch.sum(torch.log(self.lamb*v)))
        trm2 = 0.5*(self.lamb*torch.sum(v_pri) + self.lamb*torch.sum(m_pri**2) - d_h - torch.sum(torch.log(self.lamb*v_pri)))
        KL_term = trm1 + trm2

        # print("this is KL_term: ", KL_term)
        # print("this is pred_samps: ", pred_samps)

        return pred_samps, KL_term, m_pri, v_pri

def loss_func_per_sample(pred_samps, KL_term, y, gam, data_dim, m_pri, v_pri):

    # size(pred_samps) = num data samples by d_h by num MC samples
    n = pred_samps.shape[0]

    # m_pri times pred_samps
    m_pri_pred_samps = torch.einsum('j, njk -> nk', m_pri, pred_samps) # N by MC_samps
    y_m_pri_pred_samps = torch.einsum('n, nk -> nk', y, m_pri_pred_samps) # N by MC_samps
    trm1 = y**2
    trm2=torch.mean(2*y_m_pri_pred_samps, 1)
    trm3=torch.mean(torch.einsum('njk,j -> nk', pred_samps**2, v_pri), 1)
    trm4=torch.mean(m_pri_pred_samps**2, 1)
    trm5=0.5*data_dim*torch.log(2*torch.pi/gam)
    out1 = gam*0.5*(trm1 - trm2 + trm3 + trm4) + trm5
    out = out1 + KL_term

    print("This is out: ", out.shape)

    return out



def loss_func(pred_samps, KL_term, y, gam, data_dim, m_pri, v_pri):

    # size(pred_samps) = num data samples by d_h by num MC samples
    n = pred_samps.shape[0]

    # m_pri times pred_samps
    m_pri_pred_samps = torch.einsum('j, njk -> nk', m_pri, pred_samps) # N by MC_samps
    y_m_pri_pred_samps = torch.einsum('n, nk -> nk', y, m_pri_pred_samps) # N by MC_samps
    trm1 = torch.sum(y**2)
    trm2 = torch.mean(torch.sum(2*y_m_pri_pred_samps, 0))
    trm3 = torch.mean(torch.sum(torch.einsum('njk,j -> nk', pred_samps**2, v_pri),0))
    trm4 = torch.mean(torch.sum(m_pri_pred_samps**2,0))
    trm5 = 0.5*n*data_dim*torch.log(2*torch.pi/gam)
    out1 = gam*0.5*(trm1 - trm2 + trm3 + trm4) + trm5
    out = out1 + KL_term
    # print("This is out: ", out)

    return out


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='wine', \
                        help='choose the data name among naval, robot, power, wine, protein')

    # OPTIMIZATION
    parser.add_argument('--n-hidden', type=int, default=50, help='number of hidden units in the layer')
    parser.add_argument('--batch-size', '-bs', type=int, default=1000, help='batch size during training')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument("--normalize-data", action='store_true', default=True)
    parser.add_argument('--clf-batch-size', type=int, default=200)

    parser.add_argument('--is-private', action='store_true', default=True, help='produces a DP-VI')
    parser.add_argument('--dp-clip', type=float, default=0.01, help='the clipping norm for the gradients')

    ar = parser.parse_args()

    return ar


def main():
    ar = get_args()
    print(ar)
    np.random.seed(ar.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device='cpu'e
    batch_size = ar.batch_size

    # Load data.
    X_train, X_test, y_train, y_test = load_data(ar.data_name, ar.seed)
    # print("this is y_train shape: ", y_train.shape)

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
    num_iter = np.int(n / batch_size)
    # print("This is num datapoints: ", n)
    d_h = 50  # number of hidden units in the hidden layer
    # define the length of variational parameters
    len_m = d_h * (d + 1)  # length of mean parameters for W_0, where the size of W_0 is d_h by (d+1)
    #print("This is len_m: ", len_m)
    len_v = d_h * (d + 1)  # length of variance parameters for W_0
    len_m_pri = d_h  # length of mean parameters for w_1, where the size of w_1 is d_h
    len_v_pri = d_h  # length of variance parameters for w_1
    init_ms = 0.01*torch.randn(len_m + len_m_pri) # initial values for all means
    init_vs = 0.01*torch.randn(len_v + len_v_pri) # initial values for all variances
    ms_vs = torch.cat((init_ms, init_vs), 0)

    # these hyperparameters gamma and lambda are taken from PBP results
    # gam = 0.1 # gamma is the noise precision
    # lamb = 0.2 # lambda is the precision on weights
    num_samps = 500

    #Used on wine params.
    if ar.data_name == 'wine':
        a = 550.883042
        b = 288.7679997
    elif ar.data_name == 'naval':
        a=4780.383185
        b=144.3312336
    elif ar.data_name =='robot':
        a=3271.898334
        b=331.122069
    elif ar.data_name =='power':
        a=4084.001869
        b=236.8769553 
    elif ar.data_name =='protein':
        a=20080.58288
        b=11029.92738
    elif ar.data_name == 'year':
        a=224608.67015
        b=136304.09595
    else:
        a = 6.0
        b = 6.0

    gam = b/a # setting gamma to the posterior mean
    lamb = 0.1 # this is questionable.


    model = NN_Model(len_m, len_m_pri, len_v, num_samps, ms_vs, device, lamb, d)
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=0.00001)
    v_noise = b / a * std_y_train ** 2


    # training routine should start here.
    for epoch in range(1, ar.epochs + 1):
        model.train()

        for i in range(num_iter):

            inputs = X_train[i * ar.clf_batch_size:(i + 1) * ar.clf_batch_size, :]
            labels = y_train[i * ar.clf_batch_size:(i + 1) * ar.clf_batch_size]

            optimizer.zero_grad()

            pred_samps, KL_term, m_pri, v_pri = model(torch.Tensor(inputs)) # some portion of X_train if mini-batch learning is happening
            
            if ar.is_private:
                loss=loss_func_per_sample(pred_samps, KL_term, torch.Tensor(labels), torch.tensor(gam), d, m_pri, v_pri)
                print("Loss before backward: ", loss)
                for i in range(loss.size()[0]):
                    loss[i].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm(model.parameters(), ar.dp_clip)

            else:
                loss = loss_func(pred_samps, KL_term, torch.Tensor(labels), torch.tensor(gam), d, m_pri, v_pri)
                            # pred_samps, KL_term, y, gam, data_dim, m_pri, v_pri
            
                loss.backward()
                optimizer.step()

        print('Epoch {}: loss : {}'.format(epoch, loss))


        #### testing in every epoch ####
        pred_samps_y_tst, KL_term, m_pri, v_pri = model(torch.Tensor(X_test))


        samps_from_standard_normal = torch.randn(d_h, num_samps)
        samps_std_adjusted = torch.einsum('i,ik ->ik', torch.sqrt(v_pri), samps_from_standard_normal)
        samples_w_1 = m_pri[:, None].repeat(1, samps_std_adjusted.shape[1]) + samps_std_adjusted

        w_1_times_pred_samps_y_tst = torch.einsum('jk,  njk -> nk', samples_w_1, pred_samps_y_tst)
        m_prd = (torch.mean(w_1_times_pred_samps_y_tst, 1)).detach().numpy()
        v_prd = (torch.var(w_1_times_pred_samps_y_tst, 1)).detach().numpy()
        m_prd = m_prd * std_y_train + mean_y_train
        v_prd = v_prd * std_y_train ** 2

        test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v_prd + v_noise)) - \
                          0.5 * (y_test - m_prd) ** 2 / (v_prd + v_noise))
        #print("test_ll: ", test_ll)

        rmse = np.sqrt(np.mean((y_test - m_prd) ** 2))
        #print("rmse: ", rmse)




    # # We obtain the test RMSE and the test ll
    # m, v, v_noise = pbp_instance.get_predictive_mean_and_variance(X_test)
    # rmse = np.sqrt(np.mean((y_test - m) ** 2))
    # print("rmse: ", rmse)


    #### following PBP code:
    #         v_noise = b/a * std_y_train**2
    # m, v = self.predict_probabilistic(X_test[ i, : ])
    #             (self.network.a.get_value() - 1) * self.std_y_train**2
    #             m = m * self.std_y_train + self.mean_y_train
    #             v = v * self.std_y_train**2
    #


if __name__ == '__main__':
    main()