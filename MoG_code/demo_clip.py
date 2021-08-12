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

color = ['r.', 'y.', 'g.', 'c.']

def kl_approx(mu1, sig1, mu2, sig2):
	# compute KL divergence between 2 Gaussians with full
	mu_diff = mu2 - mu1
	if len(sig2.shape) == 1:
	    KL = (sig1 / sig2).sum() - sig1.shape[0]
	    KL += (mu_diff ** 2 / sig2).sum()
	    KL += np.log(sig2).sum() - np.log(sig1).sum()
	else:
	    inv_sig2 = np.linalg.inv(sig2)
	    KL = np.trace(np.dot(inv_sig2, sig1)) - sig1.shape[0]
	    KL += np.dot(np.dot(mu_diff, inv_sig2), mu_diff)
	    KL += np.log(np.linalg.det(sig2)) - np.log(np.linalg.det(sig1)) 
	
	return KL / 2.0

def demo_clutter(seed, step, num_data, num_group, size, prior_precision, w, 
		std_noise, show=False, dim = [0, 1], c=10, clip=False):
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
	
#	if show:
#		alpha = 0.55
#		width = 5; hight = 3
#		fig1, ax1 = pyplot.subplots(2, 2, figsize=(width, hight))
#		pyplot.title('test')
#		for n in xrange(X_test.shape[0]):
#			ax1[0, 0].plot(X_test[n, dim[0]], X_test[n, dim[1]], \
#				color[y_test[n]], alpha=alpha)
#			ax1[0, 0].set_title('truth')
#			ax1[0, 0].axis('off')
			
#		# for training
#		fig2, ax2 = pyplot.subplots(2, 2, figsize=(width, hight))
#		pyplot.title('train')
#		for n in xrange(X.shape[0]):
#			ax2[0, 0].plot(X[n, dim[0]], X[n, dim[1]], color[y[n]], alpha=alpha)
#			ax2[0, 0].set_title('truth')
#			ax2[0, 0].axis('off')
		#pyplot.show()
	
	np.random.seed(0)
	prior_mu = np.random.randn(num_group, size)
	
	# sample from the true posterior
	level = 0.98
	sampling = True
	linewidth = 2.0
	if sampling:
		print 'computing the true posterior...'
		m_samp, var_samp, samp = gmm_sampling(X, y, num_group, prior_mu, \
			prior_precision, w, std_noise,  save_res=True) #Added save_res=True to save results.
		# draw true posteior
	#	if show:
	#		bvn_full = BVN()
	#		cov_full = []; mean_full = []
	#		for k in xrange(num_group):
	#			cov_full.append(var_samp[k][np.ix_(dim, dim)])
	#			mean_full.append(m_samp[k][dim])			
	#		bvn_full.covars = cov_full
	#		bvn_full.mean = np.array(mean_full)
	#		make_ellipses(bvn_full, ax1[0, 0], level=level)
	#		for k in xrange(num_group):
	#			e1, e2 = gauss_ell(mean_full[k], cov_full[k], dim = [0, 1], \
	#				npoints = 200, level =level)
	#			ax1[0, 0].plot(e1, e2, 'k', linewidth=linewidth)
	#			ax2[0, 0].plot(e1, e2, 'k', linewidth=linewidth)
	
	# learning options
	#mode = ['full', 'stochastic', 'adf']
	mode=['stochastic']
	learning_rate = 0.1
	#ll = np.zeros([len(mode), num_track])	# for test likelihood
	#t = np.zeros([len(mode), num_track])		# for training time

	# learning 'stochastic'
	#name = {'full':'EP', 'stochastic':'SEP', 'adf':'ADF'}
	name={'stochastic' : 'SEP'}
	for m in xrange(len(mode)):
		print "fitting with %s..." % name[mode[m]]

		#if (mode[m] == 'stochastic' and clip==True):
			#Check that the norm  of the prior mean is bounded by c, otherwise we clip it.
		#	norm_prior_mu=np.linalg.norm(prior_mu)
		#	prior_mu = prior_mu / max(1, norm_prior_mu / c)
			#Check that the norm of the prior covariance is bounded by c, otherwise clip it.
			
		clutter_train = mog(size, prior_mu, prior_precision, std_noise, w)
		time_ep = time.time()
		for i in xrange(len(step)):
			clutter_train.train_ep(X, step[i], learning_rate, mode[m], c, clip=clip)
		#	t[m, i] = t[m, i] + time.time() - time_ep
			y_pred, logZ_pred = clutter_train.predict(X_test)
			y_pred_train, _ = clutter_train.predict(X)
		#	ll[m, i] = logZ_pred.mean()
		#	time_ep = time.time()
			if i==(len(step)-1):
		#		if (clip is True and mode[m] == "stochastic"):
				print "Computing avraged F-norm"
				err_mean, err_cov=clutter_train.compute_mse(m_samp, var_samp)
				print"The averaged F-norm for  the mean is {}".format(err_mean)
				print"The averaged F-norm for  the covariance is {}".format(err_cov)

		#if show:
		#	i = int(m >= 1); j = int(np.mod(m, 2) == 0)
		#	for n in xrange(X_test.shape[0]):
		#		ax1[i, j].plot(X_test[n, dim[0]], X_test[n, dim[1]], \
		#			color[y_pred[n]], alpha=alpha)
		#		ax1[i, j].set_title(name[mode[m]])
		#		ax1[i, j].axis('off')
				
		#	for n in xrange(X.shape[0]):
		#		current_color = y_pred_train[n]
		#		ax2[i, j].plot(X[n, dim[0]], X[n, dim[1]], \
		#			color[current_color], alpha=alpha)
		#		ax2[i, j].set_title(name[mode[m]])
		#		ax2[i, j].axis('off')
				
		#	bvn = BVN()
		#	cov = []; mean = []
		#	for k in xrange(num_group):
		#		if clutter_train.full_cov is True:
		#			cov.append(clutter_train.pISx[k][np.ix_(dim, dim)] \
		#				+ clutter_train.ISx[k][np.ix_(dim, dim)])
		#			cov[-1] = np.linalg.inv(cov[-1])
		#			mean.append(clutter_train.pMISx[k][dim] \
		#				+ clutter_train.MISx[k][dim])
		#			mean[-1] = np.dot(cov[-1], mean[-1])
		#		else:
		#			cov.append(clutter_train.pISx[k][dim] \
		#				+ clutter_train.ISx[k][dim])
		#			mean.append(clutter_train.pMISx[k][dim] \
		#				+ clutter_train.MISx[k][dim])
		#			mean[-1] = mean[-1] / cov[-1]
		#			cov[-1] = np.eye(2) / cov[-1]
					
		#	bvn.covars = cov
		#	bvn.mean = np.array(mean)
		#	make_ellipses(bvn, ax1[i, j], level=level)

		#	for k in xrange(num_group):
		#		e1, e2 = gauss_ell(mean[k], cov[k], dim = [0, 1], \
		#			npoints = 200, level =level)
		#		ax1[i, j].plot(e1, e2, 'k', linewidth=linewidth)
		#		ax2[i, j].plot(e1, e2, 'k', linewidth=linewidth)
	
#	if show:
		#pyplot.show()
#		pyplot.savefig('demo_figure_clipping_c={}.png'.format(c))
		
	return err_mean, err_cov

		
if __name__ == '__main__':
	num_data = int(sys.argv[1])
	if len(sys.argv) > 2:
		size = int(sys.argv[2])
	else:
		size = 4
	if len(sys.argv) > 3:
		num_group = int(sys.argv[3])
	else:
		num_group = 4
	
	w = np.ones(num_group)
	if len(w) != num_group:
		w = np.ones(num_group)
	w = np.array(w); w = w / float(w.sum())
	std_noise = 0.5
	
	full_cov = True
	var_prior = 10.0
	if full_cov is False:
		prior_precision = np.ones([num_group, size]) / var_prior
	else:
		prior_precision = np.eye(size) / var_prior
		prior_precision = np.tile(prior_precision, (num_group, 1, 1))
	
	#c=np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30]) #The clipping norm to apply
	c=np.arange(1,21)
	len_c=len(c)
	num_test = 1
	# for SEP paper figure, seed = [60]
	seed = [60]#np.arange(num_test) * 50
	num_mode = 3
	step = np.array([1, 2, 3, 4, 10, 30, 50, 100])
	num_track = len(step)
	#ll = np.zeros([len_c, num_test, num_mode, num_track])
	#time_ep = np.zeros([len_c, num_test, num_mode, num_track])
	dim = [0, 1]
	fnorm_mean=[]
	fnorm_cov=[]
	
	
	print 'settings:'
	print 'N_train_data = %d, dim = %d, N_clusters = %d, full Cov matrix = %s,' \
		% (num_data, size, num_group, full_cov)
	print 'total number of epochs = %d' % step.sum()
	for i in xrange(num_test):
		for j in xrange(len_c):
			show = (i >= (num_test-1))
			err_mean, err_cov= demo_clutter(seed[i], step, num_data, num_group, \
				size, prior_precision, w, std_noise, show=show, dim=dim, c=c[j], clip=True)
			fnorm_mean.append(err_mean)
			fnorm_cov.append(err_cov)
		sep_err_mean, sep_err_cov=demo_clutter(seed[i], step, num_data, num_group, \
                                size, prior_precision, w, std_noise, show=show, dim=dim, c=c[j], clip=False)

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
	filename='SEP_fnorm_clipping_step={}.pdf'.format(step[-1])
	fig.savefig(os.path.join(save_path, filename), format="pdf")

