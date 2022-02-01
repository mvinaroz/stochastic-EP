import numpy as np
import argparse
from mog import mog
import matplotlib
from matplotlib import pyplot
matplotlib.pyplot.switch_backend('agg')
from gmm import model as GMM
from sampling import gmm_sampling
import os
from autodp import privacy_calibrator
from plot_error import find_cluster

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

	# DP SPEC
	parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon in (epsilon, delta)-DP')
	parser.add_argument('--delta', type=float, default=1e-6, help='delta in (epsilon, delta)-DP')

	ar = parser.parse_args()

	preprocess_args(ar)
	return ar

def preprocess_args(ar):

	if ar.seed is None:
		#ar.seed = np.random.randint(0, 1000)
		ar.seed=[6]
	

def compute_mse_dpem(true_mean, true_var, mean_dpem, sig_dpem):
	# compute the mse.
	J=mean_dpem.shape[0]

	#Compute the natural parameters for the global approximation.
	#for j in range(J):
	#	sig_dpem[j] = np.linalg.inv(sig_dpem[j])
	#	mean_dpem[j] = np.dot(sig_dpem[j],mean_dpem[j])

	#First find the cluster (Gaussian component) by finding the minimum distance between aech true means and the approximated ones.
	label=find_cluster(true_mean, mean_dpem, J)
	print("These are the correct labels: ", label)
	#Compute the mse for each gaussian component and average them
	err_mean=0
	err_var=0
	for i in range(J):
		err_mean += ((true_mean[i] - mean_dpem[label[i]]) ** 2).sum() / float(J)
		err_var += ((true_var[i] - sig_dpem[label[i]]) ** 2).sum() / float(J)
	return err_mean, err_var

def demo_clutter(seed, num_data, num_group, size, prior_precision, w,
		std_noise, epsilon=1.0, delta=1e-6):
	# generate data
	np.random.seed(seed*10)
	scale = 1.0
	MU_B = np.random.randn(num_group, size) * scale
	SIGMA_B = np.abs(np.random.randn(num_group, size))
	model = GMM(num_group, size, std_noise, MU_B, SIGMA_B, w)
    
    	# Simulate_data
	print('simulating training data...')
	data = model.simulate_data(num_data, seed=seed)
	X = data['X']	# observations
	y = data['y']	# cluster labels
	num_data = X.shape[0]
	
	# test data
	num_data_test = 1000
	print('simulating test data...')
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
		print('computing the true posterior...')
		m_samp, var_samp, samp = gmm_sampling(X, y, num_group, prior_mu, \
			prior_precision, w, std_noise) #you can add save_res=True to save results.
	
	
	path_file=os.path.join(os.getcwd(),'dpem_results/')

	mean_dpem=np.load(os.path.join(path_file,'dpme_mu_eps=' + str(int(epsilon)) + '_delta=1e-5_lap=0_comp=4.npy'))
	mean_dpem=mean_dpem.T
	std_dpem=np.load(os.path.join(path_file,'dpme_sigma_eps=' + str(int(epsilon)) + '_delta=1e-5_lap=0_comp=4.npy'))


	
	print("Computing averaged F-norm")
	err_mean, err_cov=compute_mse_dpem(m_samp, var_samp, mean_dpem, std_dpem)
	print("The averaged F-norm for  the mean is {}".format(err_mean))
	print("The averaged F-norm for  the covariance is {}".format(err_cov))
	f_norm_total=(err_mean + err_cov)/2
	print("The averaged F-norm is {}".format(f_norm_total))

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
    
	fnorm_mean=[]
	fnorm_cov=[]
	fnorm_mean_dp=[]
	fnorm_cov_dp=[]

	print('settings:')
	print('N_train_data = %d, dim = %d, N_clusters = %d, full Cov matrix = %s' \
		% (ar.num_data, ar.dimension, ar.num_group, ar.full_cov))
	#print('total number of iterations = %d' % ar.num_iter)
	
	err_mean_dp, err_cov_dp= demo_clutter(ar.seed[0], ar.num_data, ar.num_group, \
			ar.dimension, prior_precision, w, ar.std, epsilon=ar.epsilon, delta=ar.delta)

if __name__ == '__main__':
	main()

