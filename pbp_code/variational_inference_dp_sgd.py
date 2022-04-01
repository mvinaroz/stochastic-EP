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
from autodp import rdp_acct, rdp_bank

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
        z_0_n =  self.relu(x_W_0_plus_bias) # num data samples by d_h by num MC samples (N  time d_h times L)

        ### KL term
        trm1 = 0.5*(self.lamb*torch.sum(v) + self.lamb*torch.sum(m**2) - (self.data_dim + 1) - (self.data_dim + 1)*np.log(self.lamb) - torch.sum(torch.log(v)))
        trm2 = 0.5*(self.lamb*torch.sum(v_pri) + self.lamb*torch.sum(m_pri**2) - (d_h +1) - (d_h+1)*np.log(self.lamb) -  torch.sum(torch.log(v_pri)))
        KL_term = trm1 + trm2

        # print("this is KL_term: ", KL_term)
        # print("this is pred_samps: ", pred_samps)

        return z_0_n, KL_term, m_pri, v_pri


def loss_func(pred_samps, KL_term, y, gam, m_pri, v_pri):

    # size(z_0_n) = num data samples by d_h by num MC samples
    n = pred_samps.shape[0]

    #Add +1 to z_0_n d_h dim
    z_0_n = F.pad(input=pred_samps, pad=(0, 0, 0, 1), mode='constant', value=1)
    #print("This is z_0_n shape: ", z_0_n.shape)
    #print("This is z_0_n_bias shape: ", z_0_n_bias)

    # m_pri times z_0_n
    m_pri_z_0_n = torch.einsum('j, njk -> nk', m_pri, z_0_n) # N by MC_samps
    y_m_pri_z_0_n = torch.einsum('n, nk -> nk', y, m_pri_z_0_n) # N by MC_samps
    trm1 = torch.sum(y**2)
    trm2 = torch.mean(torch.sum(2*y_m_pri_z_0_n, 0))
    trm3 = torch.mean(torch.sum(torch.einsum('njk,j -> nk', z_0_n**2, v_pri),0))
    trm4 = torch.mean(torch.sum(m_pri_z_0_n**2,0))
    trm5 = 0.5*n*torch.log(2*torch.pi/gam)
    out1 = gam*0.5*(trm1 - trm2 + trm3 + trm4) + trm5
    out = out1 + KL_term
    # print("This is out: ", out)

    return out

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
    parser.add_argument('--data-name', type=str, default='naval', \
                        help='choose the data name among naval, robot, power, wine, protein')

    # OPTIMIZATION
    parser.add_argument('--n-hidden', type=int, default=50, help='number of hidden units in the layer')
    parser.add_argument('--batch-size', '-bs', type=int, default=200, help='batch size during training')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument("--normalize-data", action='store_true', default=True)
    parser.add_argument('--clf-batch-size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001,  help='learning rate' )
    parser.add_argument('--lamb', type=float, default=10,  help='precision on weights' )
    parser.add_argument('--gam', type=float, default=0.1,  help='precision on noise' )
    parser.add_argument('--mc-samps', type=int, default=500,  help='number of mc samples to generate' )

    parser.add_argument('--is-private', action='store_true', default=False, help='produces a DP-VI')
    parser.add_argument('--dp-clip', type=float, default=0.0001, help='the clipping norm for the gradients')
    parser.add_argument('--dp-sigma', type=float, default=50., help='sigma for dp-vi')

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
    d_h = 50 # number of hidden units in the hidden layer 
    # define the length of variational parameters
    len_m = d_h * (d + 1)  # length of mean parameters for W_0, where the size of W_0 is d_h by (d+1)
    #print("This is len_m: ", len_m)
    len_v = d_h * (d + 1)  # length of variance parameters for W_0
    len_m_pri = d_h + 1 # length of mean parameters for w_1, where the size of w_1 is d_h
    len_v_pri = d_h + 1# length of variance parameters for w_1
    init_ms = 0.01*torch.randn(len_m + len_m_pri) # initial values for all means
    init_vs = 0.01*torch.randn(len_v + len_v_pri) # initial values for all variances
    ms_vs = torch.cat((init_ms, init_vs), 0)
    

    # these hyperparameters gamma and lambda are taken from PBP results
    # gam = 0.1 # gamma is the noise precision
    # lamb = 0.2 # lambda is the precision on weights
    num_samps = ar.mc_samps

    model = NN_Model(len_m, len_m_pri, len_v, num_samps, ms_vs, device, ar.lamb, d)
    optimizer = optim.SGD(model.parameters(), lr=ar.lr)
    # optimizer = optim.Adam(model.parameters(), lr=0.00001)
    v_noise =  (1./ar.gam) * std_y_train ** 2 #the variance for the noise and std_y_train ** 2 is because it's reescaling it.
    print("This is v_noise: ", v_noise)

    #Compute DP budget.
    delta=1e-5
    if ar.is_private:
        final_epsilon=privacy_param_func(ar.dp_sigma , delta, ar.epochs, ar.clf_batch_size, n)
        #privacy_param_func(sigma, delta, n_epochs, batch_size, n_data):


    # training routine should start here.
    for epoch in range(1, ar.epochs + 1):
        model.train()

        for i in range(num_iter):

            inputs = X_train[i * ar.clf_batch_size:(i + 1) * ar.clf_batch_size, :]
            labels = y_train[i * ar.clf_batch_size:(i + 1) * ar.clf_batch_size]

            optimizer.zero_grad()

            z_0_n, KL_term, m_pri, v_pri = model(torch.Tensor(inputs)) # some portion of X_train if mini-batch learning is happening
            
            if ar.is_private:
                #TO DO: compute loss per sample with a loop
                for samp in range(labels.shape[0]):
                    loss_per_sample=loss_func(z_0_n[samp, :, :], KL_term, torch.Tensor(labels[samp]), torch.tensor(ar.gam), m_pri, v_pri) #This loss has size minibatc_size
                    #print("Loss before backward: ", loss)
                    save_clipped_grad=torch.zeros(ar.clf_batch_size, ms_vs.shape[0]) #tensor that will contain the clipped gradients where size is batch_size per parameters size
                    #print('Thi is save_clipped_grad.shape: ', save_clipped_grad.shape)
                for i in range(loss.size()[0]):
                    loss[i].backward(retain_graph=True) #Compute per sample gradient.
                    """Check the gradient norm before clipping"""
                    #total_norm = 0
                    #for p in model.parameters():
                    #    print(p.grad)
                    #    param_norm = p.grad.detach().data.norm(2)
                    #    total_norm += param_norm.item() ** 2
                    #    total_norm = total_norm ** 0.5
                    #print("This is the total norm of the parameter gradients: ", total_norm)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), ar.dp_clip) #Clip per sample gradient.
                    """Check the gradient norm after clipping"""
                    #total_norm2 = 0
                    #for p in model.parameters():
                    #    print(p.grad)
                    #    param_norm2 = p.grad.detach().data.norm(2)
                    #    total_norm2 += param_norm2.item() ** 2
                    #    total_norm2 = total_norm2 ** 0.5
                    #print("This is the total norm of the parameter gradients after clipping: ", total_norm2)
                    """Save the clipped grad into a tensor so after we can add noise and do the mean"""
                    for p in model.parameters():
                        save_clipped_grad[i, :]= p.grad
                #print("This are the clipped gradients saved: ", save_clipped_grad)
                sum_clipped_grad=torch.sum(save_clipped_grad, dim=0) #Sum over all minibatch gradients (ms_vs size).
                #print("This is sum_clipped_grad shape: ", sum_clipped_grad.shape)

                """Now we have to add noise"""
                noise_sdev = ar.dp_sigma * 2 * ar.dp_clip  # gaussian noise standard dev is computed (sensitivity is 2*clip)...
                perturbed_grad = (sum_clipped_grad + torch.randn_like(sum_clipped_grad, device=device) * noise_sdev) / loss.size()[0] # ...and applied


                """Version 1. Update the model parameters as in SGD and the perturbed gradients"""
#                for p in model.parameters():
#                    p = p - ar.lr*perturbed_grad
#                    p.grad=perturbed_grad # now we set the parameter gradient to what we just computed

                """Version 2. Update the model with perturbed gradients and perform SGD by optimizer"""
                for p in model.parameters():
                    p.grad=perturbed_grad
                optimizer.step()

                
            else:
                loss = loss_func(z_0_n, KL_term, torch.Tensor(labels), torch.tensor(ar.gam), m_pri, v_pri)
                            # pred_samps, KL_term, y, gam, data_dim, m_pri, v_pri
            
                loss.backward()
                optimizer.step()

        print('Epoch {}: loss : {}'.format(epoch, loss))


        #### testing in every epoch ####
        pred_samps_y_tst, KL_term, m_pri, v_pri = model(torch.Tensor(X_test))


        samps_from_standard_normal = torch.randn(d_h + 1, num_samps)
        samps_std_adjusted = torch.einsum('i,ik ->ik', torch.sqrt(v_pri), samps_from_standard_normal)
        samples_w_1 = m_pri[:, None].repeat(1, samps_std_adjusted.shape[1]) + samps_std_adjusted

        pred_sampls_y_tst_bias = F.pad(input=pred_samps_y_tst, pad=(0, 0, 0, 1), mode='constant', value=1) #add bias term

        w_1_times_pred_samps_y_tst = torch.einsum('jk,  njk -> nk', samples_w_1, pred_sampls_y_tst_bias)
        m_prd = (torch.mean(w_1_times_pred_samps_y_tst, 1)).detach().numpy()
        v_prd = (torch.var(w_1_times_pred_samps_y_tst, 1)).detach().numpy()
        m_prd = m_prd * std_y_train + mean_y_train
        v_prd = v_prd * std_y_train ** 2

        test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v_prd + v_noise)) - \
                          0.5 * (y_test - m_prd) ** 2 / (v_prd + v_noise))
        print("test_ll: ", test_ll)

        rmse = np.sqrt(np.mean((y_test - m_prd) ** 2))
        print("rmse: ", rmse)




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