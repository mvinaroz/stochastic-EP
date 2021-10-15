import sys
sys.path.append('PBP_net_dpsep/')

import numpy as np
import argparse
from aux_files import load_data
from sklearn import preprocessing
import pbp
from autodp import privacy_calibrator

import pandas as pd
import math
import os


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='navalt', \
                        help='choose the data name among naval, robot, power, wine, protein')

    # OPTIMIZATION
    parser.add_argument('--n-hidden', type=int, default=50, help='number of hidden units in the layer')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument("--normalize-data", action='store_true', default=True)
    parser.add_argument("--clip", action='store_true', default=False)
    parser.add_argument('--c', type=float, default=45.0)
    parser.add_argument('--gamma', type=float, default=1, help='for computing the damping factor')
 
    # DP SPEC
    parser.add_argument('--is-private',action='store_true', default=False, help='produces a DP-SEP')
    parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon in (epsilon, delta)-DP')
    parser.add_argument('--delta', type=float, default=1e-5, help='delta in (epsilon, delta)-DP')

   
    ar = parser.parse_args()

    return ar


def preprocess_args(ar):

    if ar.is_private is True:
        ar.clip=True

def main():
    ar = get_args()
    
    # We fix the random seed

    np.random.seed(ar.seed)

    #Load data.
    X_train, X_test, y_train, y_test=load_data(ar.data_name, ar.seed)
    #print("Before normalizing data: ", y_train)
    print("this is y_train shape: ", y_train.shape)

    mean_y_train = np.mean(y_train)
    std_y_train=np.std(y_train)

    #Normalizing data
    if ar.normalize_data:
        print('normalizing the data')
        std_X_train = np.std(X_train, 0)
        std_X_train[ std_X_train == 0 ] = 1
        mean_X_train = np.mean(X_train, 0)
       
        X_train = (X_train - np.full(X_train.shape, mean_X_train)) / \
            np.full(X_train.shape, std_X_train)

        X_test = (X_test - np.full(X_test.shape, mean_X_train)) / \
            np.full(X_test.shape, std_X_train)

        y_train = (y_train - mean_y_train) / std_y_train


    else:
        print('testing non-standardized data')


    #Check if we are doing probit regression (n_hidden=0) or PBP (n_hidden>0)
    if ar.n_hidden>0:
        print("We are doing PBP with {} hidden units".format(ar.n_hidden))
        n_units_per_layer = \
            np.concatenate(([ X_train.shape[ 1 ] ], [ar.n_hidden], [ 1 ]))
    else:
        print("We are doing probit regression")
        n_units_per_layer = \
            np.concatenate(([ X_train.shape[ 1 ] ], [ 1 ]))

    if ar.is_private is True:
        N=X_train.shape[0]
        print("Computing the calibrated noise")
        prob=1. / N #Sampling without replacement mong all the possible datapoints in the training set.
        k=2*ar.epochs*N #Number of times we are running the algorithm.
        privacy_param=privacy_calibrator.gaussian_mech(ar.epsilon, ar.delta, prob=prob, k=k)
        
        learning_rate=ar.gamma / N

        noise= 2*ar.c*learning_rate*privacy_param['sigma']
        print("The noise calculated is: {}".format(noise))
    else:
        noise=0.0

    fullEP=X_train.shape[0]

    pbp_instance = \
        pbp_instance=pbp.PBP(n_units_per_layer, mean_y_train, std_y_train, fullEP, ar.clip, ar.c, ar.data_name)

    # We iterate the learning process

    pbp_instance.do_pbp(X_train, y_train, ar.epochs, ar.clip, ar.c, ar.data_name, ar.seed, ar.n_hidden, ar.is_private, noise)
     

    # We obtain the test RMSE and the test ll
    m, v, v_noise = pbp_instance.get_predictive_mean_and_variance(X_test)
    rmse = np.sqrt(np.mean((y_test - m)**2))
    print("rmse: ", rmse)
    test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
    0.5 * (y_test - m)**2 / (v + v_noise))
    print("test_ll: ", test_ll)

    #Save results
    cur_path = os.path.dirname(os.path.abspath(__file__))
    save_path=os.path.join(cur_path, 'results')
    if not os.path.exists(save_path):
	        os.makedirs(save_path)

    filename_error= "rmse_dpsep_{}_clip={}_c={}_num_iter={}_n_hidden={}_seed={}_is_private={}_eps={}_delta={}.txt".format(ar.data_name, ar.clip, ar.c, ar.epochs, ar.n_hidden, ar.seed, ar.is_private, ar.epsilon, ar.delta)
    filename_ll= "test_ll_dpsep_{}_clip={}_c={}_num_iter={}_n_hidden={}_seed={}_is_private={}_eps={}_delta={}.txt".format(ar.data_name, ar.clip, ar.c, ar.epochs, ar.n_hidden, ar.seed, ar.is_private, ar.epsilon, ar.delta)
    
    open("results/"+filename_error, 'w').close()
    open("results/"+filename_ll, 'w').close()


    with open("results/"+filename_error, "a") as myfile:
        myfile.write(repr(rmse) + '\n')

    with open("results/"+filename_ll, "a") as myfile:
        myfile.write(repr(test_ll) + '\n')
    


if __name__ == '__main__':
    main()
