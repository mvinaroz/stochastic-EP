
import sys

import math

import numpy as np

import theano

import theano.tensor as T

import network

import prior

import os

class PBP:

    def __init__(self, layer_sizes, mean_y_train, std_y_train, fullEP, clip, c, data_name):

        var_targets = 1
        self.std_y_train = std_y_train
        self.mean_y_train = mean_y_train

        # We initialize the prior

        self.prior = prior.Prior(layer_sizes, var_targets, data_name)

        # We create the network

        params = self.prior.get_initial_params()
        

        if clip is True:
            print("We are clipping the initial prior parameters")
            self.clip_norm(params, c)

        #Set the inital prior parameters to the initial posterior.
        self.network = network.Network(params[ 'm_w' ], params[ 'v_w' ],
            params[ 'a' ], params[ 'b' ], fullEP)


        # We create the input and output variables in theano

        self.x = T.vector('x')
        self.y = T.scalar('y')
        self.alpha = T.scalar('alpha')
        
        # A function for computing the value of logZ, logZ1 and logZ2

        self.logZ, self.logZ1, self.logZ2 = \
            self.network.logZ_Z1_Z2(self.x, self.y, self.alpha)

        # We create a theano function for updating the posterior

        self.adf_update = theano.function([ self.x, self.y, self.alpha ], 
            self.logZ, updates = self.network.generate_updates(self.logZ, 
            self.logZ1, self.logZ2))

        # We greate a theano function for the network predictive distribution

        self.predict_probabilistic = theano.function([ self.x ],
            self.network.output_probabilistic(self.x))

        self.predict_deterministic = theano.function([ self.x ],
            self.network.output_deterministic(self.x))

    def do_pbp(self, X_train, y_train, n_iterations, clip, c, data_name, seed, n_hidden, is_private, noise):

        if n_iterations > 0:

            #Load prior parameters.
            prior_params = self.prior.get_params(data_name, seed, n_hidden)
            #print('Prior mean parameters before clipping: ', prior_params['m_w'])
            if clip is True:
                self.clip_norm(prior_params, c)
                #print('Prior mean parameters bafter clipping: ', prior_params['m_w'])

            #print('Prior mean parameters shape: ', len(prior_params['m_w']))

            # We first do a single pass

            self.do_first_pass(data_name, prior_params, seed, n_hidden, X_train, y_train, True, clip, c, is_private, noise)


            # We refine the prior

            #params = self.network.get_params()
            #params = self.prior.refine_prior(params)

            #if clip is True:
            #    N=X_train.shape[0]
            #    clip_bound=(N+1)*c
            #    self.clip_norm(params, clip_bound)
           

            #self.network.set_params(params)


            #print('param', params['m_w'][0][0, 0], params['v_w'][0][0, 0])

            #sys.stdout.write('{}\n'.format(0))
            #sys.stdout.flush()

            for i in range(int(n_iterations)-1):

                # We do one more pass

                
                self.do_first_pass(data_name, prior_params, seed, n_hidden, X_train, y_train, True, clip, c, is_private, noise)

                # We refine the prior

                #params = self.network.get_params()
                #params = self.prior.refine_prior(params)

                #if clip is True:
                    #We need to clip the posterior parameters with the refined prior.
                #    N=X_train.shape[0]
                #    clip_bound=(N+1)*c
                #    self.clip_norm(params, c)
                    
                #self.network.set_params(params)
                
                #print('param', params['m_w'][0][0, 0], params['v_w'][0][0, 0])
                
                sys.stdout.write('{}\n'.format(i + 1))
                sys.stdout.flush()
            #params_last=self.prior.get_params(data_name, seed, n_iterations, n_hidden)
            #print("The mean prior parameters last run: ", params_last['m_w'])
            #print("The variance prior parameter last run: ", params_last['v_w'])
    

    def get_deterministic_output(self, X_test):

        output = np.zeros(X_test.shape[ 0 ])
        for i in range(X_test.shape[ 0 ]):
            output[ i ] = self.predict_deterministic(X_test[ i, : ])
            output[ i ] = output[ i ] * self.std_y_train + self.mean_y_train

        return output

    def get_predictive_mean_and_variance(self, X_test):

        mean = np.zeros(X_test.shape[ 0 ])
        variance = np.zeros(X_test.shape[ 0 ])
        for i in range(X_test.shape[ 0 ]):
            m, v = self.predict_probabilistic(X_test[ i, : ])
            m = m * self.std_y_train + self.mean_y_train
            v = v * self.std_y_train**2
            mean[ i ] = m
            variance[ i ] = v

        v_noise = self.network.b.get_value() / \
            (self.network.a.get_value() - 1) * self.std_y_train**2

        return mean, variance, v_noise

    def do_first_pass(self, data_name, prior_params, seed, n_hidden, X, y, stochastic = False, clip=False, c=10.0, is_private=False, noise=0.0):

        permutation = np.random.choice(range(X.shape[ 0 ]), X.shape[ 0 ],
            replace = False)

        alpha = 1.0
        counter = 0
        N = float(X.shape[0])
        for i in permutation:

            old_params = self.network.get_params()

            #print("The old mean parameters: ", old_params['m_w'])

            #prior_params = self.prior.get_params(data_name, seed, n_hidden)
   
            
            #if clip is True:
                #self.clip_norm(prior_params, c)
                #print("prior mean after clipping: ", prior_params['m_w'])

            cavity_params_n = self.network.compute_cavity(prior_params, alpha, N, stochastic)
            logZ = self.adf_update(X[ i, : ], y[ i ], alpha)
            new_params = self.network.get_params() #Gives us q_{new}.
            new_params = self.network.update_local(new_params, old_params, cavity_params_n, prior_params, alpha, N, stochastic, clip, c, is_private, noise)
            self.network.set_params(new_params)
            #new_params2=self.network.get_params()
            #print("The new mean parameters: ", new_params2['m_w'])
            #print(np.array(new_params2['m_w']).shape)
            #for i in range(len(new_params2['m_w'])):
            #    print(new_params2['m_w'][i].shape)

            if counter % 1000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()

            counter += 1
        
        print(self.network.a.get_value(), self.network.b.get_value())
        #prior_params = self.prior.get_params()
        #print("The prior a parameter: ", prior_params['a'])
        
        sys.stdout.write('\n')
        sys.stdout.flush()

    def sample_w(self):

        self.network.sample_w()

    def clip_norm(self, params, c):
        
        #Clip the mean and variance NATURAL parameters.
        #print("var_parameters before clipping: ", params['v_w'])
        for item in range(len(params['v_w'])):
            v_nat_params=1. / params['v_w'][item]
            #print("Natural params for the variance: " ,v_nat_params.shape)
            m_nat_params=params['m_w'][item] / params['v_w'][item]
            norm_var=np.linalg.norm(v_nat_params)
            norm_mean=np.linalg.norm(m_nat_params)
            #print("Variance norm: ", norm_var)

            v_nat_params= v_nat_params / max(1, norm_var / c)
            m_nat_params=m_nat_params / max(1, norm_mean / c)
            
            params['v_w'][item] = 1. / v_nat_params
            params['m_w'][item]= m_nat_params*params['v_w'][item] 
        #print("Parameters after clipping: ", params['v_w']) 

    def clip_nat(self, params, c):
        for item in range(len(params['v_w'])):
            norm_var=np.linalg.norm(params['v_w'][item])
            norm_mean=np.linalg.norm(params['m_w'][item])

            params['v_w'][item]= params['v_w'][item] / max(1, norm_var / c)
            params['m_w'][item]= params['m_w'][item] / max(1, norm_mean / c)



        
