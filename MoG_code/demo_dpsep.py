import numpy as np
import argparse
from mog import mog
import matplotlib
from matplotlib import pyplot
matplotlib.pyplot.switch_backend('agg')
from gmm import model as GMM
from sampling import gmm_sampling
import os


def get_args():
	parser = argparse.ArgumentParser()

    	# BASICS
    	parser.add_argument('--seed', type=int, default=None, help='sets random seed')
    	parser.add_argument('--num-data', type=int, default=200, help='snumber of data points to generate')
    	parser.add_argument('--dimension', type=int, default=4, help='data dimension') 
    	parser.add_argument('--num_group', type=int, default=4, help='number of gaussian components') 
    	#parser.add_argument('--w', type=float, default=None, help='') 
    	parser.add_argument('--std', type=float, default=0.5, help='') 
    	parser.add_argument('--full_cov', default=True, help='') 
    	parser.add_argument('--var_prior', type=float, default=10.0 , help='') 
    	parser.add_argument('--num_test', type=int, default=1 , help='') 

    	parser.add_argument('--num-iter', type=int, default=100, help='number of iterations to run the algorithm')
	parser.add_argument('--model-name', type=str, default='stochastic', help='you can use stochastic')
    	parser.add_argument('--clip', default=True, help='for clipping the approximating factors')
	parser.add_argument('--c', nargs='+', type=int,default=[20],  help=' list of clipping values', required=False)
    	#parser.add_argument('--c', default=20, help='clipping value')

   	# DP SPEC
   	parser.add_argument('--is-private', default=False, help='produces DP-SEP version')
   	parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon in (epsilon, delta)-DP')
   	parser.add_argument('--delta', type=float, default=1e-6, help='delta in (epsilon, delta)-DP')
    	parser.add_argument('--gamma', type=float, default=1.0, help='damping factor used in the global approximation update')

    	ar = parser.parse_args()

    	preprocess_args(ar)
    	return ar

def preprocess_args(ar):

	if ar.seed is None:
        	#ar.seed = np.random.randint(0, 1000)
        	ar.seed=[60]

    	assert ar.model_name in {'stochastic'}

def demo_clutter(seed, step, num_data, num_group, size, prior_precision, w,
		std_noise, c=10, clip=False, learning_rate=0.1):
	print"C value ${0}".format(c)
	# generate data
	np.random.seed(seed*10)
	scale = 1.0
	MU_B = np.random.randn(num_group, size) * scale
	SIGMA_B = np.abs(np.random.randn(num_group, size))
	model = GMM(num_group, size, std_noise, MU_B, SIGMA_B, w)
    
    	# Simulate_data
	print 'simulating training data...'
	data = model.simulate_data(num_data, seed=seed)
	X = data['X']	# observations
	y = data['y']	# cluster labels
	num_data = X.shape[0]
	
	# test data
	num_data_test = 1000
	print 'simulating test data...'
	data_test = model.simulate_data(num_data_test, seed=seed)
	X_test = data_test['X']
	y_test = data_test['y']


	np.random.seed(0)
	prior_mu = np.random.randn(num_group, size)
	
	# sample from the true posterior
	level = 0.98
	sampling = True
	linewidth = 2.0
	if sampling:
		print 'computing the true posterior...'
		m_samp, var_samp, samp = gmm_sampling(X, y, num_group, prior_mu, \
			prior_precision, w, std_noise) #you can add save_res=True to save results.
	
	mode=['stochastic']

	name={'stochastic' : 'SEP'}
	for m in xrange(len(mode)):
		print "fitting with %s..." % name[mode[m]]
		
		clutter_train = mog(size, prior_mu, prior_precision, std_noise, w)
		
		for i in xrange(len(step)):
			clutter_train.train_ep(X, step[i], learning_rate, mode[m], c, clip=clip, is_private=False, epsilon=1.0, delta=1e-6)
			y_pred, logZ_pred = clutter_train.predict(X_test)
			y_pred_train, _ = clutter_train.predict(X)
			
			if i==(len(step)-1):
				print "Computing avraged F-norm"
				err_mean, err_cov=clutter_train.compute_mse(m_samp, var_samp)
				print"The averaged F-norm for  the mean is {}".format(err_mean)
				print"The averaged F-norm for  the covariance is {}".format(err_cov)
	
	return err_mean, err_cov


def main():
	"""Load settings"""
    	ar = get_args()
   	print(ar)

    	w = np.ones(ar.num_group)
    	#w = np.array(w)
    	w = w / float(w.sum())

    	if ar.full_cov is False:
		prior_precision = np.ones([ar.num_group, ar.dimension]) / ar.var_prior
    	else:
		prior_precision = np.eye(ar.dimension) / ar.var_prior
		prior_precision = np.tile(prior_precision, (ar.num_group, 1, 1))

    	learning_rate= float(ar.gamma) / ar.num_data
    
	fnorm_mean=[]
  	fnorm_cov=[]
	c=np.array(ar.c)
	step = np.array([1, 2, 3, 4, 10, 30, 50, 100])
	

	print 'settings:'
	print 'N_train_data = %d, dim = %d, N_clusters = %d, full Cov matrix = %s, learning_rate=%s' \
		% (ar.num_data, ar.dimension, ar.num_group, ar.full_cov, learning_rate)
	print 'total number of iterations = %d' % ar.num_iter
    	for i in xrange(ar.num_test):
        	for j in xrange(len(c)):
			err_mean, err_cov= demo_clutter(ar.seed[i], step, ar.num_data, ar.num_group, \
					ar.dimension, prior_precision, w, ar.std, c=c[j], clip=ar.clip, learning_rate=learning_rate)
			fnorm_mean.append(err_mean)
			fnorm_cov.append(err_cov)
		sep_err_mean, sep_err_cov=demo_clutter(ar.seed[i], step, ar.num_data, ar.num_group, \
                	ar.dimension, prior_precision, w, ar.std, c=c[j], clip=False, learning_rate=learning_rate)
	
	sep_mean=np.repeat(sep_err_mean, len(c))
	sep_cov=np.repeat(sep_err_cov, len(c))
	fnorm_mean=np.array(fnorm_mean)
	fnorm_cov=np.array(fnorm_cov)
	
	#Plotting the results
	fig, ax = pyplot.subplots(2, 1, figsize=(5, 4))
	ax[0].plot(c, fnorm_mean, label="clipping SEP")
	ax[0].plot(c, sep_mean, label="SEP")
	ax[0].set_ylabel('Mean')  # Add a y-label to the axes.
	ax[0].get_xaxis().set_ticks(c)

	ax[1].plot(c, fnorm_cov, label="clipping SEP")
	ax[1].plot(c, sep_cov, label="SEP")
	ax[1].set_xlabel('Clipping value C')
	ax[1].set_ylabel('Covariance')
	ax[1].get_xaxis().set_ticks(c)
	ax[1].set_yscale("log")
	fig.suptitle("Averaged F-norm") 
	ax[0].legend()
	ax[1].legend()
	
	#Save the generated plot
	cur_path = os.path.dirname(os.path.abspath(__file__))
	save_path=os.path.join(cur_path, 'results')
	if not os.path.exists(save_path):
	        os.makedirs(save_path)
	filename='SEP_fnorm_clipping_step={}_learning_rate={}_private={}_eps={}_delta={}.pdf'.format(step[-1], learning_rate, ar.is_private, ar.epsilon, ar.delta)
	fig.savefig(os.path.join(save_path, filename), format="pdf")







if __name__ == '__main__':
	main()
