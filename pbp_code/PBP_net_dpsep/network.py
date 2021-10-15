
import math

import numpy as np

import theano

import theano.tensor as T

import network_layer

class Network:

    def __init__(self, m_w_init, v_w_init, a_init, b_init, fullEP = None):

        # We create the different layers

        self.layers = []

        if len(m_w_init) > 1:
            for m_w, v_w in zip(m_w_init[ : -1 ], v_w_init[ : -1 ]):
                self.layers.append(network_layer.Network_layer(m_w, v_w, True, fullEP))

        self.layers.append(network_layer.Network_layer(m_w_init[ -1 ],
            v_w_init[ -1 ], False, fullEP))

        # We create mean and variance parameters from all layers

        self.params_m_w = []
        self.params_v_w = []
        self.params_w = []
        for layer in self.layers:
            self.params_m_w.append(layer.m_w)
            self.params_v_w.append(layer.v_w)
            self.params_w.append(layer.w)

        # We create the theano variables for a and b

        self.a = theano.shared(float(a_init))
        self.b = theano.shared(float(b_init))
        
        # for full EP we need local paramters 
        if fullEP is not None:
            self.n_data = fullEP
            self.a_n = np.zeros(self.n_data)
            self.b_n = np.zeros(self.n_data)

    def output_deterministic(self, x):

        # Recursively compute output

        for layer in self.layers:
            x = layer.output_deterministic(x)

        return x

    def output_probabilistic(self, m):

        v = T.zeros_like(m)

        # Recursively compute output

        for layer in self.layers:
            m, v = layer.output_probabilistic(m, v)

        return (m[ 0 ], v[ 0 ])

    @staticmethod
    def n_cdf(x):

        return 0.5 * (1.0 + T.erf(x / T.sqrt(2.0)))

    def logZ_Z1_Z2(self, x, y, alpha):

        n_a = 2.0 / (alpha + 1.0)

        m, v = self.output_probabilistic(x)

        #logZ = n_a * T.log(Network.n_cdf(y * m / T.sqrt(v + 1.0)))

        #return (logZ, logZ, logZ)

        v_final = v + self.b / (self.a - 1)
        v_final1 = v + self.b / self.a
        v_final2 = v + self.b / (self.a + 1)

        logZ = n_a*(-0.5 * (T.log(v_final) + (y - m)**2 / v_final))
        logZ1 = n_a*(-0.5 * (T.log(v_final1) + (y - m)**2 / v_final1))
        logZ2 = n_a*(-0.5 * (T.log(v_final2) + (y - m)**2 / v_final2))

        return (logZ, logZ1, logZ2)

    def generate_updates(self, logZ, logZ1, logZ2):

        updates = []
        for i in range(len(self.params_m_w)):
            updates.append((self.params_m_w[ i ], self.params_m_w[ i ] + \
                self.params_v_w[ i ] * T.grad(logZ, self.params_m_w[ i ])))
            updates.append((self.params_v_w[ i ], self.params_v_w[ i ] - \
               self.params_v_w[ i ]**2 * \
                (T.grad(logZ, self.params_m_w[ i ])**2 - 2 * \
                T.grad(logZ, self.params_v_w[ i ]))))

        #updates.append((self.a, 1.0 / (T.exp(logZ2 - 2 * logZ1 + logZ) * \
        #    (self.a + 1) / self.a - 1.0)))
        #updates.append((self.b, 1.0 / (T.exp(logZ2 - logZ1) * (self.a + 1) / \
        #    (self.b) - T.exp(logZ1 - logZ) * self.a / self.b)))

        return updates
    
    def get_params(self):

        m_w = []
        v_w = []
        for layer in self.layers:
            m_w.append(layer.m_w.get_value())
            v_w.append(layer.v_w.get_value())

        return { 'm_w': m_w, 'v_w': v_w , 'a': self.a.get_value(),
            'b': self.b.get_value() }

    def set_params(self, params):

        for i in range(len(self.layers)):
            self.layers[ i ].m_w.set_value(params[ 'm_w' ][ i ])
            self.layers[ i ].v_w.set_value(params[ 'v_w' ][ i ])

        self.a.set_value(params[ 'a' ])
        self.b.set_value(params[ 'b' ])
    
    def compute_cavity(self, prior_params, alpha, N, stochastic):
        # compute cavity distribution
        # by removing the ith site

        n_a = 2.0 / (alpha + 1.0)

        m_w = []; m_w_n = []
        v_w = []; v_w_n = []
        l = 0
        N = float(N)
        for layer in self.layers:
            # global params
            v_w_q = 1.0 / layer.v_w.get_value()
            m_w_q = layer.m_w.get_value() * v_w_q
            # prior params
            v_w_p = 1.0 / prior_params[ 'v_w' ][ l ]
            m_w_p = prior_params[ 'm_w' ][ l ] / prior_params[ 'v_w' ][ l ]
            # compute cavity with fraction 1/N
            if stochastic:
                m_w_c = m_w_q - (m_w_q - m_w_p) / (N * n_a)
                v_w_c = v_w_q - (v_w_q - v_w_p) / (N * n_a)
            else:
                m_w_c = m_w_q
                v_w_c = v_w_q
            m_w_n.append(m_w_c)
            v_w_n.append(v_w_c)
            m_w_c = m_w_c / v_w_c
            v_w_c = 1.0 / v_w_c
            m_w.append(m_w_c)
            v_w.append(v_w_c)
            l += 1
        
        # no cavity computations for a and b since they're from the prior?
        a_q = self.a.get_value() - 1.0	# natural param
        a_p = prior_params[ 'a' ] - 1.0
        if stochastic:
            a_n = a_q - (a_q - a_p) / (n_a * N)
        else:
            a_n = a_q
        a_c = a_n + 1.0

        b_q = -self.b.get_value()		# natural param
        b_p = -prior_params[ 'b' ]
        if stochastic:
            b_n = b_q - (b_q - b_p) / (n_a * N)
        else:
            b_n = b_q
        b_c = -b_n
        
        # set params
        params = { 'm_w': m_w, 'v_w': v_w , 'a': a_c, 'b': b_c}
        self.set_params(params)
        
        # return natural params
        return {'m_w': m_w_n, 'v_w': v_w_n, 'a': a_n, 'b': b_n}
    
        
    def update_local(self, new_params, old_params, cavity_params_n, prior_params, alpha, N, stochastic, clip, c, is_private, noise):
        # update the ith local site 
        # taking the difference between 
        # new update and cavity
        # we assume we're doing normal EP with power 1
        
        # cavity params = { 'm_w': m_w, 'v_w': v_w , 'a': a_c, 'b': b_c }
        # all in natural parameters

        n_a = 2.0 / (alpha + 1.0)
        step = 0.4
        l = 0
        for layer in self.layers:
        
            # We obtain the prior parameters

            v_w_p = 1.0 / prior_params[ 'v_w' ][ l ]
            m_w_p = prior_params[ 'm_w' ][ l ] / prior_params[ 'v_w' ][ l ]
 
            # We obtain the new natural parameters

            v_w_n_new = 1.0 / new_params[ 'v_w' ][ l ]
            m_w_n_new = new_params[ 'm_w' ][ l ] / new_params[ 'v_w' ][ l ]

            # We obtain the old natural parameters before the update

            v_w_n_old = 1.0 / old_params[ 'v_w' ][ l ]
            m_w_n_old = old_params[ 'm_w' ][ l ] / old_params[ 'v_w' ][ l ]

            # We obtain the cavity natural parameters

            v_w_n_cavity = cavity_params_n[ 'v_w' ][ l ]
            m_w_n_cavity = cavity_params_n[ 'm_w' ][ l ]

            # We compute the new approximation parameters

            #The f_n.
            m_f_n= n_a*(m_w_n_new - m_w_n_cavity)
            v_f_n=n_a*(v_w_n_new - v_w_n_cavity)

            if clip is True:
            #    #We first compute f_n natural parameters.
            #    v_f_n_nat=1.0 / v_f_n
            #    m_f_n_nat=m_f_n / v_f_n

                #Ensure f_n natural parameters are bounded by c.
                clipped_v_f_n_nat=self.clip_norm(v_f_n, c)
                clipped_m_f_n_nat=self.clip_norm(m_f_n, c)

                #Go back to the previous parameters
            #    v_f_n=1.0 / clipped_v_f_n_nat
            #    m_f_n=clipped_m_f_n_nat/clipped_v_f_n_nat

                 

            if stochastic:
                m_w_n_new_final = m_w_n_old - (m_w_n_old - m_w_p) / N + m_f_n
                v_w_n_new_final = v_w_n_old - (v_w_n_old - v_w_p) / N +v_f_n 
                

            else:
                m_w_n_new_final = m_w_n_old + n_a * (m_w_n_new - m_w_n_cavity)
                v_w_n_new_final = v_w_n_old + n_a * (v_w_n_new - v_w_n_cavity)

            # We eliminate invalid updates

            index1 = np.where(v_w_n_new <= 1e-20)
            index2 = np.where(v_w_n_new >= 1e20)
            index3 = np.where(np.isnan(v_w_n_new))
            index4 = np.where(np.isnan(m_w_n_new))

            index5 = np.where(v_w_n_cavity <= 1e-20)
            index6 = np.where(v_w_n_cavity >= 1e20)
            index7 = np.where(np.isnan(v_w_n_cavity))
            index8 = np.where(np.isnan(m_w_n_cavity))

            index9 = np.where(v_w_n_new_final <= 1e-20)
            index10 = np.where(v_w_n_new_final >= 1e20)
            index11 = np.where(np.isnan(v_w_n_new_final))
            index12 = np.where(np.isnan(m_w_n_new_final))

            index = [ np.concatenate((index1[ 0 ], index2[ 0 ], index3[ 0 ], \
                index4[ 0 ], index5[ 0 ], index6[ 0 ], index7[ 0 ], index8[ 0 ], index9[ 0 ], index10[ 0 ], index11[ 0 ], index12[ 0 ])), \
                np.concatenate((index1[ 1 ], index2[ 1 ], index3[ 1 ], index4[ 1 ], index5[ 1 ], index6[ 1 ], index7[ 1 ], index8[ 1 ], \
                index9[ 1 ], index10[ 1 ], index11[ 1 ], index12[ 1 ])) ]

            if len(index[ 0 ]) > 0:
                #print("This is index: , ", index)
                m_w_n_new_final[tuple(index)] = m_w_n_old[tuple(index)]
                v_w_n_new_final[tuple(index)] = v_w_n_old[tuple(index)]

            #Once invalid parameters are gone we ensure that the natural parameters of q_new are bounded by (N+1)*c
            if clip is True:
                    #print("We are clipping the updated posterior")
                    clip_bound=(N+1)*c
                    #Compute natural parameters.
                    #v_w_n_new_final_nat= 1.0 / v_w_n_new_final
                    #m_w_n_new_final_nat=m_w_n_new_final/v_w_n_new_final
                    #Clip natural parameters
                    clipped_m_w_new_final=self.clip_norm(m_w_n_new_final, clip_bound)
                    clipped_v_w_new_final=self.clip_norm(v_w_n_new_final, clip_bound)
                    
                    if is_private is True:
                        clipped_m_w_new_final= clipped_m_w_new_final + self.perturb_mean(clipped_m_w_new_final, noise)
                        clipped_v_w_new_final= clipped_v_w_new_final + self.perturb_var(clipped_v_w_new_final, noise)

            # We update the posterior approximation

            #if clip is True:
            #   new_params[ 'm_w' ][ l ] = clipped_m_w_new_final
            #   new_params[ 'v_w' ][ l ] = clipped_v_w_new_final
            #else:
            #    new_params[ 'm_w' ][ l ] = m_w_n_new_final / v_w_n_new_final
            #    new_params[ 'v_w' ][ l ] = 1.0 / v_w_n_new_final
            new_params[ 'm_w' ][ l ] = m_w_n_new_final / v_w_n_new_final
            new_params[ 'v_w' ][ l ] = 1.0 / v_w_n_new_final
            l += 1
        
        # We obtain the prior parameters

        a_p = prior_params[ 'a' ] - 1.0
        b_p = -prior_params[ 'b' ]

        # We obtain the new natural parameters

        a_n_new = new_params[ 'a' ] - 1.0
        b_n_new = -new_params[ 'b' ]

        # We obtain the old natural parameters

        a_n_old = old_params[ 'a' ] - 1.0
        b_n_old = -old_params[ 'b' ]

        # We obtain the cavity natural parameters
        
        a_n_cavity = cavity_params_n[ 'a' ]
        b_n_cavity = cavity_params_n[ 'b' ]

        # We compute the new approximation parameters

        if stochastic:
            a_n_new_final = a_n_old - (a_n_old - a_p) / N + n_a * (a_n_new - a_n_cavity)
            b_n_new_final = b_n_old - (b_n_old - b_p) / N + n_a * (b_n_new - b_n_cavity)
        else:
            a_n_new_final = a_n_old + n_a * (a_n_new - a_n_cavity)
            b_n_new_final = b_n_old + n_a * (b_n_new - b_n_cavity)

        # We undo invalid updates

        if np.isnan(a_n_new_final) or np.isnan(b_n_new_final) or a_n_new_final < 0 or b_n_new_final > 0 or \
            np.isnan(a_n_cavity) or np.isnan(b_n_cavity) or a_n_cavity < 0 or b_n_cavity > 0 or \
            np.isnan(a_n_new) or np.isnan(b_n_new) or a_n_new < 0 or b_n_new > 0:

            a_n_new_final = a_n_old
            b_n_new_final = b_n_old

        # We update the posterior approximation

        new_params[ 'a' ] = a_n_new_final + 1.0
        new_params[ 'b' ] = -b_n_new_final
        
        return new_params
        
    def sample_w(self):

        w = []
        for i in range(len(self.layers)):
            w.append(self.params_m_w[ i ].get_value() + \
                np.random.randn(self.params_m_w[ i ].get_value().shape[ 0 ], \
                self.params_m_w[ i ].get_value().shape[ 1 ]) * \
                np.sqrt(self.params_v_w[ i ].get_value()))

        for i in range(len(self.layers)):
            self.params_w[ i ].set_value(w[ i ])

    def clip_norm(self, params, c):
        
        #Clip the mean and variance NATURAL parameters.
        norm_item=np.linalg.norm(params)
        clipped_nat_params= params / max(1, norm_item / c)
        #print("Parameters after clipping: ", clipped_nat_params) 

        return clipped_nat_params

    def perturb_mean(self, param, noise):
    
       #Draw noise for perturbing the mean.
       noise=np.random.standard_normal(param.shape)*noise
       noised_mean = param + noise
       return noised_mean

    def perturb_var(self, param, noise):

        #Draw noise for perturbing the variance.
       noise=np.random.standard_normal(param.shape)*noise
       noised_var= param + noise

       #Ensure that all values are positive.
       noised_var[noised_var<=0] = 0.0001

       return noised_var


    
