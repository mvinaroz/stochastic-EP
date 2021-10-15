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
	parser.add_argument('--is-private', action='store_true', help='produces DP-SEP version, by default is False')
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
		std_noise, learning_rate=0.1, c=10, clip=False, is_private=False, epsilon=1.0, delta=1e-6):
	print("C value ${0}".format(c))
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
	
	mode=['stochastic']

	name={'stochastic' : 'SEP'}
	for m in range(len(mode)):
		print("fitting with %s..." % name[mode[m]])
		
		clutter_train = mog(size, prior_mu, prior_precision, std_noise, w)
		
		if is_private is True:
			print("Computing the calibrated noise")
			#Using RDP, calibrating the noise for multiple rounds of subsampled mechanisms
			total_iter=step.sum()
			prob= 1. / num_data #Uniformly sample without replacement.

			#Non-splitted privacy budget version 
			#k= 2*num_data*total_iter #Times we rund the algorithm (times we compose dp mechanisms)
			#privacy_param= privacy_calibrator.gaussian_mech(epsilon,delta,prob=prob,k=k)

			#Splitting the privacy budget among the parameters
			k=num_data*total_iter
			eps_mean=4*epsilon/5
			eps_cov=epsilon - eps_mean
			delta_mean=4*delta/5
			delta_cov= delta - delta_mean
			privacy_param_mean= privacy_calibrator.gaussian_mech(eps_mean,delta_mean,prob=prob,k=k)
			privacy_param_cov= privacy_calibrator.gaussian_mech(eps_cov,delta_cov,prob=prob,k=k)

			#We need to multiply the sigma value by the sensitivity (2C*gamma/N) where gamma/N = learning_rate.
			#noise= 2*c*learning_rate*privacy_param['sigma']
			#print("The noise calculated is: {}".format(noise))
			
			noise_mean=2*c*learning_rate*privacy_param_mean['sigma']
			noise_cov=2*c*learning_rate*privacy_param_cov['sigma']
			print("The noise for the mean is: {}".format(noise_mean))
			print("The noise for the covariance matrix is: {}".format(noise_cov))
		else:
			#noise=0.0
			noise_mean=0.0
			noise_cov=0.0

		for i in range(len(step)):
			clutter_train.train_ep(X, step[i], learning_rate, mode[m], noise_mean, noise_cov, c, clip=clip, \
				is_private=is_private)
			y_pred, logZ_pred = clutter_train.predict(X_test, m_samp)
			y_pred_train, _ = clutter_train.predict(X, m_samp)
			
			if i==(len(step)-1):
				print("Computing averaged F-norm")
				err_mean, err_cov=clutter_train.compute_mse(m_samp, var_samp)
				print("The averaged F-norm for  the mean is {}".format(err_mean))
				print("The averaged F-norm for  the covariance is {}".format(err_cov))
				kl_div=clutter_train.averaged_KL(m_samp, var_samp)
				print("The averaged KL-divergence is {}".format(kl_div))

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
	fnorm_mean_dp=[]
	fnorm_cov_dp=[]

	c=np.array(ar.c)
	step = np.array([ar.num_iter])
	

	print('settings:')
	print('N_train_data = %d, dim = %d, N_clusters = %d, full Cov matrix = %s, learning_rate=%s, is_private=%s' \
		% (ar.num_data, ar.dimension, ar.num_group, ar.full_cov, learning_rate, ar.is_private))
	#print('total number of iterations = %d' % ar.num_iter)
	for i in range(ar.num_test):
		for j in range(len(c)):
			err_mean_dp, err_cov_dp= demo_clutter(ar.seed[i], step, ar.num_data, ar.num_group, \
					ar.dimension, prior_precision, w, ar.std, c=c[j], clip=ar.clip, \
					learning_rate=learning_rate, is_private=ar.is_private, epsilon=ar.epsilon, delta=ar.delta)
#			fnorm_mean_dp.append(err_mean_dp)
#			fnorm_cov_dp.append(err_cov_dp)
			
			#Non-private clipped version
#			err_mean, err_cov= demo_clutter(ar.seed[i], step, ar.num_data, ar.num_group, \
#                                        ar.dimension, prior_precision, w, ar.std, c=c[j], clip=ar.clip, \
#                                        learning_rate=learning_rate, is_private=False, epsilon=ar.epsilon, delta=ar.delta)
#			fnorm_mean.append(err_mean)
#			fnorm_cov.append(err_cov)

#		sep_err_mean, sep_err_cov=demo_clutter(ar.seed[i], step, ar.num_data, ar.num_group, \
#                	ar.dimension, prior_precision, w, ar.std, c=c[j],  clip=False, \
#			learning_rate=learning_rate, is_private=False, epsilon=ar.epsilon, delta=ar.delta)
	
	sep_mean=np.repeat(sep_err_mean, len(c))
	sep_cov=np.repeat(sep_err_cov, len(c))
	fnorm_mean=np.array(fnorm_mean)
	fnorm_cov=np.array(fnorm_cov)
	fnorm_mean_dp=np.array(fnorm_mean_dp)
	fnorm_cov_dp=np.array(fnorm_cov_dp)

	#Plotting the results
#	fig, ax = pyplot.subplots(2, 1, figsize=(5, 4))
#	ax[0].plot(c, fnorm_mean_dp, label="DP-SEP")
#	ax[0].plot(c, fnorm_mean, label="clipping SEP")
#	ax[0].plot(c, sep_mean, label="SEP")
#	ax[0].set_ylabel('Mean')  # Add a y-label to the axes.
#	ax[0].get_xaxis().set_ticks(c)

#	ax[1].plot(c, fnorm_cov_dp, label="DP-SEP")
#	ax[1].plot(c, fnorm_cov, label="clipping SEP")
#	ax[1].plot(c, sep_cov, label="SEP")
#	ax[1].set_xlabel('Clipping value C')
#	ax[1].set_ylabel('Covariance')
#	ax[1].get_xaxis().set_ticks(c)
#	ax[1].set_yscale("log")
#	fig.suptitle("Averaged F-norm") 
#	ax[0].legend()
#	ax[1].legend()
	
	#Save the generated plot
#	cur_path = os.path.dirname(os.path.abspath(__file__))
#	save_path=os.path.join(cur_path, 'results')
#	if not os.path.exists(save_path):
#	        os.makedirs(save_path)
#	filename='SEP_fnorm_clipping_step={}_learning_rate={}_num_iter={}_private={}_eps={}_delta={}.pdf'.format(step[-1], learning_rate,step.sum(), ar.is_private, ar.epsilon, ar.delta)
#	fig.savefig(os.path.join(save_path, filename), format="pdf")







if __name__ == '__main__':
	main()
