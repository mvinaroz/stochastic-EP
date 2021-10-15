import numpy as np
from mog import mog
import sys, time
from comp_ground_truth import moment_numerical, kl_approx
import matplotlib
from matplotlib import pyplot
matplotlib.pyplot.switch_backend('agg')
from gmm import model as GMM
from try_bvn_ellipse import BVN, make_ellipses, gauss_ell
from sampling import gmm_sampling
import os
import argparse
from autodp import privacy_calibrator

color = ['r.', 'y.', 'g.', 'c.']


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
	parser.add_argument('--var-prior', type=float, default=10.0 , help='') 
	parser.add_argument('--num-test', type=int, default=1 , help='') 

	parser.add_argument('--num-iter', type=int, default=100, help='number of iterations to run the algorithm')
	#parser.add_argument('--model-name', type=str, default='stochastic', help='you can use stochastic')
	parser.add_argument('--clip', default=True, help='for clipping the approximating factors')
	parser.add_argument('--c', nargs='+', type=int,default=[20],  help=' list of clipping values', required=False)
	#parser.add_argument('--c', default=20, help='clipping value')

	# DP SPEC
	parser.add_argument('--is-private', action='store_true', help='produces DP-SEP version, by default is False')
	#parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon in (epsilon, delta)-DP')
	parser.add_argument('--delta', type=float, default=1e-6, help='delta in (epsilon, delta)-DP')
	parser.add_argument('--gamma', type=float, default=1.0, help='damping factor used in the global approximation update')

	ar = parser.parse_args()

	preprocess_args(ar)
	return ar

def preprocess_args(ar):

	if ar.seed is None:
		#ar.seed = np.random.randint(0, 1000)
		ar.seed=[60]


def demo_clutter(seed, step, num_data, num_group, size, prior_precision, w, 
		std_noise, show=False, dim = [0, 1], learning_rate=0.1, c=10, clip=False, is_private=False, delta=1e-6):
	
	#print("this is epsilon={}, delta={}, c={}, learning_rate={}".format(epsilon, delta, c, learning_rate))
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
	
	if show:
		alpha = 0.55
		width = 5; hight = 3
		if is_private is False:
			#We only show the ground truth, ep and sep predictions
			fig1, ax1 = pyplot.subplots(1, 3, figsize=(width, hight))
		else:
			fig1, ax1 = pyplot.subplots(2, 3, figsize=(width, hight))
		pyplot.title('test')
		for n in range(X_test.shape[0]):
			#print("This is y_test[n]: ", y_test[n])
			#print("This is color y_test: ", color[y_test[n]])
			ax1[0, 0].plot(X_test[n, dim[0]], X_test[n, dim[1]], \
				color[y_test[n]], alpha=alpha)
			ax1[0, 0].set_title('truth')
			ax1[0, 0].axis('off')

	np.random.seed(0)
	prior_mu = np.random.randn(num_group, size)
	
	# sample from the true posterior
	level = 0.98
	sampling = True
	linewidth = 2.0
	if sampling:
		print('computing the true posterior...')
		m_samp, var_samp, samp = gmm_sampling(X, y, num_group, prior_mu, \
			prior_precision, w, std_noise)
		# draw true posteior
		if show:
			bvn_full = BVN()
			cov_full = []; mean_full = []
			for k in range(num_group):
				cov_full.append(var_samp[k][np.ix_(dim, dim)])
				mean_full.append(m_samp[k][dim])			
			bvn_full.covars = cov_full
			bvn_full.mean = np.array(mean_full)
			make_ellipses(bvn_full, ax1[0, 0], level=level)
			for k in range(num_group):
				e1, e2 = gauss_ell(mean_full[k], cov_full[k], dim = [0,1], \
					npoints = 200, level =level)
				ax1[0, 0].plot(e1, e2, 'k', linewidth=linewidth)
				#ax2[0, 0].plot(e1, e2, 'k', linewidth=linewidth)

	mode = ['full', 'stochastic']
	#learning_rate = 0.1
	noise=0
	# learning 'stochastic'
	name = {'full':'EP', 'stochastic':'SEP'}
	for m in range(len(mode)):
		print("fitting with %s..." % name[mode[m]])
		clutter_train = mog(size, prior_mu, prior_precision, std_noise, w)
		for i in range(len(step)):
			clutter_train.train_ep(X, step[i], learning_rate, mode[m], noise, noise,  c, clip=False, is_private=False)
			y_pred, logZ_pred = clutter_train.predict(X_test, m_samp)
			y_pred_train, _ = clutter_train.predict(X, m_samp)
			#ll[m, i] = logZ_pred.mean()
		
		if show:
			j = int(m >= 1)+1; i = 0
			for n in range(X_test.shape[0]):
				#print("This is y_pred[n]: ", y_pred[n])
				#print("This is color y_pred: ", color[y_pred[n]])
				ax1[i, j].plot(X_test[n, dim[0]], X_test[n, dim[1]], \
					color[y_pred[n]], alpha=alpha)
				ax1[i, j].set_title(name[mode[m]])
				ax1[i, j].axis('off')

			bvn = BVN()
			cov = []; mean = []
			for k in range(num_group):
				if clutter_train.full_cov is True:
					cov.append(clutter_train.pISx[k][np.ix_(dim, dim)] \
						+ clutter_train.ISx[k][np.ix_(dim, dim)])
					cov[-1] = np.linalg.inv(cov[-1])
					mean.append(clutter_train.pMISx[k][dim] \
						+ clutter_train.MISx[k][dim])
					mean[-1] = np.dot(cov[-1], mean[-1])
				else:
					cov.append(clutter_train.pISx[k][dim] \
						+ clutter_train.ISx[k][dim])
					mean.append(clutter_train.pMISx[k][dim] \
						+ clutter_train.MISx[k][dim])
					mean[-1] = mean[-1] / cov[-1]
					cov[-1] = np.eye(2) / cov[-1]

			bvn.covars = cov
			bvn.mean = np.array(mean)
			make_ellipses(bvn, ax1[i, j], level=level)

			for k in range(num_group):
				e1, e2 = gauss_ell(mean[k], cov[k], dim = [0,1], \
					npoints = num_data, level =level)
				ax1[i, j].plot(e1, e2, 'k', linewidth=linewidth)
				#ax2[i, j].plot(e1, e2, 'k', linewidth=linewidth)

	#Now the private version
	name = {'stochastic':'SEP'}
	if is_private is True:
		#print("Fitting with DPSEP for epsilon={}, delta={}, c={}, learning_rate={}".format(epsilon, delta, c, learning_rate))
		#clutter_train = mog(size, prior_mu, prior_precision, std_noise, w)

		print("Computing the calibrated noise")
		#Using RDP, calibrating the noise for multiple rounds of subsampled mechanisms
		total_iter=step.sum()
		prob= 1. / num_data #Uniformly sample without replacement.

		#Non-splitted privacy budget version 
		#k= 2*num_data*total_iter #Times we rund the algorithm (times we compose dp mechanisms)
		#privacy_param= privacy_calibrator.gaussian_mech(epsilon,delta,prob=prob,k=k)

		#We need to multiply the sigma value by the sensitivity (2C*gamma/N) where gamma/N = learning_rate.
		#noise= 2*c*learning_rate*privacy_param['sigma']
		#print("The noise calculated is: {}".format(noise))

		list_epsilon=[1.0,  5.0, 50.0]
		
		for h in range(len(list_epsilon)): 

			#print("Fitting with DPSEP for epsilon={}, delta={}, c={}, learning_rate={}".format(epsilon, delta, c, learning_rate))
			clutter_train_dp = mog(size, prior_mu, prior_precision, std_noise, w)



			#Splitted privacy budget
			epsilon=list_epsilon[h]
			k=num_data*total_iter
			eps_mean=4*epsilon/5
			eps_cov=epsilon - eps_mean
			delta_mean=4*delta/5
			delta_cov= delta - delta_mean
			privacy_param_mean= privacy_calibrator.gaussian_mech(eps_mean,delta_mean,prob=prob,k=k)
			privacy_param_cov= privacy_calibrator.gaussian_mech(eps_cov,delta_cov,prob=prob,k=k)

			noise_mean=2*c*learning_rate*privacy_param_mean['sigma']
			noise_cov=2*c*learning_rate*privacy_param_cov['sigma']

			for i in range(len(step)):
				clutter_train_dp.train_ep(X, step[i], learning_rate, mode[m], noise_mean, noise_cov, c, clip=clip, is_private=is_private)
				y_pred_dp, logZ_pred = clutter_train_dp.predict(X_test, m_samp)
				y_pred_train, _ = clutter_train_dp.predict(X, m_samp)

			if show:
				i = 1
				j = h
				for n in range(X_test.shape[0]):
					#print("This is y_pred[n]: ", y_pred[n])
					#print("This is color y_pred: ", color[y_pred[n]])
					ax1[i, j].plot(X_test[n, dim[0]], X_test[n, dim[1]], \
						color[y_pred_dp[n]], alpha=alpha)
					ax1[i, j].set_title('DP-SEP eps={}'.format(epsilon))
					ax1[i, j].axis('off')

				bvn = BVN()
				cov = []; mean = []
				for k in range(num_group):
					if clutter_train_dp.full_cov is True:
						cov.append(clutter_train_dp.pISx[k][np.ix_(dim, dim)] \
							+ clutter_train_dp.ISx[k][np.ix_(dim, dim)])
						cov[-1] = np.linalg.inv(cov[-1])
						mean.append(clutter_train_dp.pMISx[k][dim] \
							+ clutter_train_dp.MISx[k][dim])
						mean[-1] = np.dot(cov[-1], mean[-1])
					else:
						cov.append(clutter_train_dp.pISx[k][dim] \
							+ clutter_train_dp.ISx[k][dim])
						mean.append(clutter_train_dp.pMISx[k][dim] \
							+ clutter_train_dp.MISx[k][dim])
						mean[-1] = mean[-1] / cov[-1]
						cov[-1] = np.eye(2) / cov[-1]

				bvn.covars = cov
				bvn.mean = np.array(mean)
				make_ellipses(bvn, ax1[i, j], level=level)
				print("THIS IS THE MEAN FOR THE PLOTS: ",  mean)

				for k in range(num_group):
					e1, e2 = gauss_ell(mean[k], cov[k], dim = [0,1], \
						npoints = num_data, level =level)
					ax1[i, j].plot(e1, e2, 'k', linewidth=linewidth)
					#ax2[i, j].plot(e1, e2, 'k', linewidth=linewidth)

	fig1.tight_layout()
	#Saving image results.
	cur_path = os.path.dirname(os.path.abspath(__file__))
	save_path=os.path.join(cur_path, 'results')

	if not os.path.exists(save_path):
		os.makedirs(save_path)
	filename="cluster_is_private={}_c={}_delta_{}_lr={}_num_iter={}_num_data={}_dim={}.pdf".format(is_private, c, delta, learning_rate, step.sum(), num_data, dim)
	fig1.savefig(os.path.join(save_path, filename), format="pdf")



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
    
	#fnorm_mean=[]
	#fnorm_cov=[]
	#fnorm_mean_dp=[]
	#fnorm_cov_dp=[]

	c=np.array(ar.c)
	step = np.array([ar.num_iter])
	dim=[[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
	
	print('settings:')
	print('N_train_data = %d, dim = %d, N_clusters = %d, full Cov matrix = %s,' \
		% (ar.num_data, ar.dimension, ar.num_group, ar.full_cov))
	print('total number of epochs = %d' % step.sum())
	for j in range(len(dim)):
		for i in range(ar.num_test):
			show = (i >= (ar.num_test-1))
			demo_clutter(ar.seed[i], step, ar.num_data, ar.num_group, ar.dimension, prior_precision, w, ar.std, show=show, dim=dim[j], \
					learning_rate=learning_rate, c=c, clip=ar.clip, is_private=ar.is_private, delta=ar.delta)



if __name__ == '__main__':
	main()
