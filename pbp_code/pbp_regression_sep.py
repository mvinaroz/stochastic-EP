import sys
sys.path.append('PBP_net_sep/')

import numpy as np
import argparse
from aux_files import load_data
from sklearn import preprocessing
import pbp
from autodp import privacy_calibrator

import pandas as pd
import math


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='naval', \
                        help='choose the data name among naval, robot, power, wine, protein')

    # OPTIMIZATION
    parser.add_argument('--n-hidden', type=int, default=50, help='number of hidden units in the layer')
    #parser.add_argument("--batch-rate", type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument("--normalize-data", action='store_true', default=True)

    ar = parser.parse_args()

    return ar


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


    fullEP=X_train.shape[0]

    pbp_instance = \
        pbp_instance=pbp.PBP(n_units_per_layer, mean_y_train, std_y_train, fullEP)

    # We iterate the learning process

    pbp_instance.do_pbp(X_train, y_train, ar.epochs)

    # We obtain the test RMSE and the test ll
    m, v, v_noise = pbp_instance.get_predictive_mean_and_variance(X_test)
    rmse = np.sqrt(np.mean((y_test - m)**2))
    print("rmse: ", rmse)
    test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
    0.5 * (y_test - m)**2 / (v + v_noise))
    print("test_ll: ", test_ll)



if __name__ == '__main__':
    main()
