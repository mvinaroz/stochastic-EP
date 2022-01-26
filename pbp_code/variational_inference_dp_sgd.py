
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

    def __init__(self, len_m, len_m_pri, len_v, num_samps, init_var_params, device):
        super(NN_Model, self).__init__()
        self.parameter = Parameter(torch.Tensor(init_var_params), requires_grad=True)
        self.num_samps = num_samps
        self.device = device
        self.len_m = len_m
        self.len_m_pri = len_m_pri
        self.len_v = len_v

    def forward(self, x): # x is mini_batch_size by input_dim

        # unpack ms_vs
        ms_vs = self.parameter
        init_m = ms_vs[0:self.len_m]
        init_m_pri = ms_vs[self.len_m:self.len_m + self.len_m_pri]
        init_v = ms_vs[self.len_m + self.len_m_pri:self.len_m + self.len_m_pri + self.len_v]
        init_v_pri = ms_vs[self.len_m + self.len_m_pri + self.len_v:]

        # to do : implement actual loss here using samples drawn from the posterior, i.e., eq(24)

        # to do : add KL term, i.e., eq(20)

        return pred_samps, KL_term


def loss(pred_samps, KL_term, y, gam, lamb):

    # write eq.(18) here
    # out =


    return out


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='naval', \
                        help='choose the data name among naval, robot, power, wine, protein')

    # OPTIMIZATION
    parser.add_argument('--n-hidden', type=int, default=50, help='number of hidden units in the layer')
    # parser.add_argument("--batch-rate", type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument("--normalize-data", action='store_true', default=True)

    ar = parser.parse_args()

    return ar


def main():
    ar = get_args()
    np.random.seed(ar.seed)

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
    d_h = 50  # number of hidden units in the hidden layer
    # define the length of variational parameters
    len_m = d_h * (d + 1)  # length of mean parameters for W_0, where the size of W_0 is d_h by (d+1)
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

    model = NN_Model(len_m, len_m_pri, len_v, num_samps, ms_vs, ar.device)
    # optimizer = optim.SGD(importance.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training routine should start here.
    # in every training step, you compute this
    pred_samps, KL_term = model(torch.tensor(X_train)) # some portion of X_train if mini-batch learning is happening

    output = loss(pred_samps, KL_term, torch.tensor(y_train), gam, lamb, num_samps)
    output.backward()

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
