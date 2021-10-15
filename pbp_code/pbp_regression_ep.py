import sys
sys.path.append('PBP_net_ep/')

import numpy as np
import argparse
from aux_files import load_data
from sklearn import preprocessing
import pbp
import math
from scipy.stats import norm


import pandas as pd


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='year', \
                        help='choose the data name among naval, robot, power, wine, protein or year')

    # OPTIMIZATION
    parser.add_argument('--n-hidden', type=int, default=100, help='number of hidden units in the layer')
    parser.add_argument("--batch-rate", type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument("--normalize-data", action='store_true', default=True)
    # parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    #parser.add_argument('--lr', type=float, default=0.01, help='learning rate') # for covtype data
    #parser.add_argument('--lr-decay', type=float, default=0.9, help='per epoch learning rate decay factor')

    # DP SPEC
    parser.add_argument('--is-private', default=False, help='produces a DP mean embedding of data')
    parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon in (epsilon, delta)-DP')
    parser.add_argument('--delta', type=float, default=1e-5, help='delta in (epsilon, delta)-DP')

   
    ar = parser.parse_args()

    return ar


#def preprocess_args(ar):


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
        #X_train = preprocessing.minmax_scale(X_train, feature_range=(0, 1), axis=0, copy=True)
        #y_train=preprocessing.minmax_scale(y_train, feature_range=(0, 1), axis=0, copy=True)
        #X_test = preprocessing.minmax_scale(X_test, feature_range=(0, 1), axis=0, copy=True)
        #y_test=preprocessing.minmax_scale(y_test, feature_range=(0, 1), axis=0, copy=True)

        std_X_train = np.std(X_train, 0)
        std_X_train[std_X_train == 0 ] = 1
        mean_X_train = np.mean(X_train, 0)

        X_train = (X_train - np.full(X_train.shape, mean_X_train)) / \
            np.full(X_train.shape, std_X_train)

        #std_X_test = np.std(X_test, 0)
        #std_X_test[std_X_test == 0 ] = 1
        

        X_test = (X_test - np.full(X_test.shape, mean_X_train)) / \
            np.full(X_test.shape, std_X_train)


        y_train = (y_train - mean_y_train) / std_y_train

    else:
        print('testing non-standardized data')

    #print("this is y_train shape: ", y_train.shape)
    #print("After normalizing data: ", y_train)
    #df2=pd.DataFrame(y_train)
    #train_stats2 = df2.describe()
    #print(train_stats2)


    #Check if we are doing probit regression (n_hidden=0) or PBP (n_hidden>0)
    if ar.n_hidden>0:
        print("We are doing PBP with {} hidden units".format(ar.n_hidden))
        n_units_per_layer = \
            np.concatenate(([ X_train.shape[ 1 ] ], [ar.n_hidden], [ 1 ]))
    else:
        print("We are doing probit regression")
        n_units_per_layer = \
            np.concatenate(([ X_train.shape[ 1 ] ], [ 1 ]))

    #mean_y_train = 0
    #std_y_train=1.0
    fullEP=X_train.shape[0]

    pbp_instance = \
        pbp_instance=pbp.PBP(n_units_per_layer, mean_y_train, std_y_train, fullEP)

    # We iterate the learning process

    pbp_instance.do_pbp(X_train, y_train, ar.epochs)

    # We obtain the test RMSE and the test ll

    #m, v, v_noise = predict(X_test)
    m, v, v_noise = pbp_instance.get_predictive_mean_and_variance(X_test)
    #print("this is m: ", m)
    rmse = np.sqrt(np.mean((y_test - m)**2))
    print("rmse: ", rmse)
    #error = np.mean(y_test != np.sign(m))
    #test_ll = np.mean(np.log(norm.cdf(y_test * m / np.sqrt(v + 1.0))))
    test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
    0.5 * (y_test - m)**2 / (v + v_noise))
    print("test_ll: ", test_ll)

if __name__ == '__main__':
    main()
