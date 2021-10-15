import numpy as np
from comp_ep_functions import *
from plot_error import find_cluster
from autodp import privacy_calibrator

class mog(object):
	def __init__(self, size, prior_mu, prior_precision, std = 1.0, w = 0.5):
		self.size = size
		self.sig_noise = std ** 2
		if type(w) == float or (type(w) == list and len(w) == 1):
			self.w = np.array([w, 1 - w])
		else:
			self.w = np.array(w)
		self.w = self.w / float(self.w.sum())
		self.J = self.w.shape[0]	# number of groups
		if len(prior_precision.shape) > len(prior_mu.shape):
			self.full_cov = True
		else:
			self.full_cov = False
		self._init_params_prior(prior_mu, prior_precision)
		self._ep_param_initialsed = False

			
	def _init_params_prior(self, prior_mu, prior_precision):
		# params for prior
		self.pISx = prior_precision
		if self.full_cov is False:
			self.pMISx = prior_mu * self.pISx
		else:
			self.pMISx = prior_mu
			for j in range(self.J):
				self.pMISx[j] = np.dot(self.pISx[j], self.pMISx[j])
		
	def _init_ep_params(self, num_data, mode = 'full'):
		# local parameters for W
		self.num_data = num_data
		self.mode = mode
		# diagonal matrix first
		if self.full_cov is False:
			if mode == 'full':
				shape_is = [num_data, self.J, self.size]
				shape_mis = [num_data, self.J, self.size]
			else:
				shape_is = [self.J, self.size]
				shape_mis = [self.J, self.size]
		else:
			if mode == 'full':
				shape_is = [num_data, self.J, self.size, self.size]
				shape_mis = [num_data, self.J, self.size]
			else:
				shape_is = [self.J, self.size, self.size]
				shape_mis = [self.J, self.size]
		# local parameters
		self.isx = np.zeros(shape_is)
		self.misx = np.zeros(shape_mis)
		
		# global parameters
		if mode == 'full':
			self.ISx = self.isx.sum(0)
			self.MISx = self.misx.sum(0)
		else:
			self.ISx = self.isx * num_data
			self.MISx = self.misx * num_data
		self._ep_param_initialsed = True
		
	def check_positive_definiteness(self, matrix):
		if self.full_cov is False:
			success = (matrix > 0).all()
		else:
			#print("The eigenvalues for the matrix are {}".format(np.linalg.eigvals(matrix)))
			success = not (np.any(np.linalg.eigvals(matrix) <= 0))
		return success
	
	def comp_current_moments(self):
		# return current moments
		if self.full_cov is False:
			Sn = 1.0 / (self.pISx + self.ISx)
			Mn = (self.pMISx + self.MISx) * Sn
		else:
			Sn = self.pISx + self.ISx
			Mn = (self.pMISx + self.MISx)
			for j in range(self.J):
				Sn[j] = np.linalg.inv(Sn[j])
				Mn[j] = np.dot(Sn[j], Mn[j])
		return Mn, Sn
		
	def comp_cavity(self, ind):
		if self.full_cov is False:
			Sn = 1.0 / (self.pISx + self.ISx - self.isx[ind])
			Mn = (self.pMISx + self.MISx - self.misx[ind]) * Sn
		else:
			Sn = self.pISx + self.ISx - self.isx[ind]
			Mn = (self.pMISx + self.MISx - self.misx[ind])
			for j in range(self.J):
				Sn[j] = np.linalg.inv(Sn[j])
				Mn[j] = np.dot(Sn[j], Mn[j])
		return Mn, Sn
	
	def comp_cavity_stochastic(self):
		if self.full_cov is False:
			Sn = 1.0 / (self.pISx + self.ISx - self.isx)
			Mn = (self.pMISx + self.MISx - self.misx) * Sn
		else:
			Sn = (self.pISx + self.ISx - self.isx)
			Mn = (self.pMISx + self.MISx - self.misx)
			for j in range(self.J):
				Sn[j] = np.linalg.inv(Sn[j])
				Mn[j] = np.dot(Sn[j], Mn[j])
		return Mn, Sn
		
	def update(self, ind, misx, isx, xi):
		tmp = self.ISx + xi * (isx - self.isx[ind])
		for j in range(self.J):
			if not self.check_positive_definiteness(tmp[j]):
				return 0
		self.ISx = self.ISx - self.isx[ind]
		self.MISx = self.MISx - self.misx[ind]
		self.isx[ind] = (1 - xi) * self.isx[ind] + xi * isx
		self.misx[ind] = (1 - xi) * self.misx[ind] + xi * misx
		self.ISx = self.ISx + self.isx[ind]
		self.MISx = self.MISx + self.misx[ind]
		
	def update_stochastic(self, minibatch, misx, isx, xi, is_private, noise_mean, noise_cov, c):
		tmp = self.ISx + xi * (isx - self.isx) * minibatch
		for j in range(self.J):
			if not self.check_positive_definiteness(tmp[j]):
				return 0
		self.ISx = self.ISx - self.isx * minibatch
		self.MISx = self.MISx - self.misx * minibatch
		self.isx = (1 - xi) * self.isx + xi * isx
		self.misx = (1 - xi) * self.misx + xi * misx
		self.ISx = self.ISx + self.isx * minibatch
		self.MISx = self.MISx + self.misx * minibatch

		if is_private is True:
			#print("The global mean before adding noise {}".format(self.MISx))
			#print("We are privatizing the global parameters")
			self.ISx= self.perturb_cov(self.ISx, noise_cov)
			self.MISx= self.perturb_mean(self.MISx, noise_mean)
			#print("The global mean after adding noise {}".format(self.MISx))
		else:
			pass

		self.isx = self.ISx / float(self.num_data)
		self.misx = self.MISx / float(self.num_data)

		if is_private is True:
			#Ensure that the new updated approximating factor is bounded by C. (post-processing step)
			self.isx=self.clip_norm( self.isx, c)
			self.misx=self.clip_norm( self.misx, c)

		

	def comp_local_updates(self, mu_t, sig_t, mu_c, sig_c):
		# compute local update
		# return no update if non-positive definite matrix
		
		# test1: diagonal matrix
		if self.full_cov is False:
			isx = 1.0 / sig_t - 1.0 / sig_c
			misx = mu_t / sig_t - mu_c / sig_c
			success = True
		else:
			isx = np.zeros(sig_c.shape)
			misx = np.zeros(mu_c.shape)
			for j in range(self.J):
				sig_t[j] = np.linalg.inv(sig_t[j])
				sig_c[j] = np.linalg.inv(sig_c[j])
				isx[j] = sig_t[j] - sig_c[j]
				misx[j] = np.dot(sig_t[j], mu_t[j]) - np.dot(sig_c[j], mu_c[j])
			success = True
				
		return misx, isx, success
		
	def comp_logZ(self, y):
		# TODO: need to be changed
		if self.mode == 'full':
			logZ = 0
			for ind in range(self.num_data):
				MU, SIG = self.comp_cavity(ind)
				pdf1 = norm.pdf(y[ind], loc = MU, scale = np.sqrt(SIG + self.sig1))
				pdf2 = norm.pdf(y[ind], loc = 0, scale = np.sqrt(self.sig2))
				logZ += np.log((1 - self.w) * pdf1 + self.w * pdf2)
		else:
			MU, SIG = self.comp_cavity_stochastic()
			pdf1 = norm.pdf(y, loc = MU, scale = np.sqrt(SIG + self.sig1))
			pdf2 = norm.pdf(y, loc = 0, scale = np.sqrt(self.sig2))
			logZ = np.log((1 - self.w) * pdf1 + self.w * pdf2).sum()
		return logZ
		
	def comp_gradient_logZ(self, y, delta = 0.001):
		SIG = 1 / (self.pISx + self.isx * (self.num_data - 1))
		MU = (self.pMISx + self.misx * (self.num_data - 1)) * SIG
		pdf1 = norm.pdf(y, loc = MU, scale = np.sqrt(SIG + self.sig1))
		pdf2 = norm.pdf(y, loc = 0, scale = np.sqrt(self.sig2))
		logZ1 = np.log((1 - self.w) * pdf1 + self.w * pdf2).sum()
		
		SIG = 1 / (self.pISx + (self.isx + delta) * (self.num_data - 1))
		MU = (self.pMISx + self.misx * (self.num_data - 1)) * SIG
		pdf1 = norm.pdf(y, loc = MU, scale = np.sqrt(SIG + self.sig1))
		pdf2 = norm.pdf(y, loc = 0, scale = np.sqrt(self.sig2))
		logZ2 = np.log((1 - self.w) * pdf1 + self.w * pdf2).sum()
		
		return (logZ2 - logZ1) / delta
		
	def train_ep(self, y, num_iter, learning_rate, mode, noise_mean, noise_cov,  c, clip=False, is_private=False):
		num_data = y.shape[0]
		
		print("The value of is_private inside train_ep function is {}".format(is_private))
		print("this is c={}, learning_rate={}".format(c, learning_rate))
		# initialising ep parameters
		if self._ep_param_initialsed == False:
			self._init_ep_params(num_data, mode)
		# start training:
		if mode == 'adf':
			for epoch in range(num_iter):
				for ind in range(num_data):
					if self.full_cov is False:
						SIG = 1.0 / (self.pISx + self.ISx)
						MU = (self.pMISx + self.MISx) * SIG
					else:
						SIG = (self.pISx + self.ISx)
						MU = (self.pMISx + self.MISx)
						for j in range(self.J):
							SIG[j] = np.linalg.inv(SIG[j])
							MU[j] = np.dot(SIG[j], MU[j])
					mean, cov, r = gmm_updates([y[ind], self.w, self.sig_noise], SIG, MU, \
						approx_x = True, full_cov = self.full_cov)
					misx, isx, success = self.comp_local_updates(mean, cov, MU, SIG)
					if success:
						self.MISx += learning_rate * misx
						self.ISx += learning_rate * isx
		
		if mode == 'full':
			for epoch in range(num_iter):
				for ind in range(num_data):
					MU, SIG = self.comp_cavity(ind)
					mean, cov, r = gmm_updates([y[ind], self.w, self.sig_noise], SIG, MU, \
						approx_x = True, full_cov = self.full_cov)
					misx, isx, success = self.comp_local_updates(mean, cov, MU, SIG)
					if success:
						self.update(ind, misx, isx, learning_rate)
					
		if mode == 'stochastic':
			if clip is True:
				print("SEP with clipping")
                		#Ensure prior parameters are clipped                
				self.pMISx=self.clip_norm( self.pMISx, c)
				self.pISx= self.clip_norm( self.pISx, c)
				#Ensure that the initial parameters of the approximating factors are clipped
				self.misx= self.clip_norm( self.misx, c)
				self.isx= self.clip_norm( self.isx, c)


			for epoch in range(num_iter):
				for i in range(num_data):
					ind = i#np.random.randint(num_data)
					MU, SIG = self.comp_cavity_stochastic()	
					mean, cov, r = gmm_updates([y[ind], self.w, self.sig_noise], SIG, MU, \
						approx_x = True, full_cov = self.full_cov)
					misx, isx, success = self.comp_local_updates(mean, cov, MU, SIG)
					if clip is True:
						#Ensure the new f_n parameters are bounded by c, otherwise we clip them.
                        			misx=self.clip_norm( misx, c)
                        			isx=self.clip_norm( isx, c)
					if success:
						#self.update_stochastic(1, misx, isx, learning_rate, is_private, noise, c)
						self.update_stochastic(1, misx, isx, learning_rate, is_private, noise_mean, noise_cov, c)

	def predict(self, X, true_mean):
		# predict the cluster label
		SIG = (self.pISx + self.ISx)
		MU = self.pMISx + self.MISx
		if self.full_cov is False:
			SIG = 1.0 / SIG
			MU = MU * SIG
		else:
			for j in range(self.J):
				SIG[j] = np.linalg.inv(SIG[j])
				MU[j] = np.dot(SIG[j], MU[j])
		y_pred = np.zeros(X.shape[0], dtype = int)
		logZ_pred = np.zeros(X.shape[0])

		#First find the cluster (Gaussian component) by finding the minimum distance between aech true means and the approximated ones.
		label=find_cluster(true_mean, MU, self.J)
		print("These are the correct labels: ", label)
		#Rearrange the components as in the truth ones.
		MU=MU[label]
		SIG=SIG[label]

		# TODO: need efficient implementation for processing multiple inputs together
		for i in range(X.shape[0]):
			r_k, logZ = gmm_updates([X[i], self.w, self.sig_noise], SIG, MU, pred = True, full_cov = self.full_cov)
			# take the max
			y_pred[i] = int(np.argmax(r_k))
			logZ_pred[i] = logZ
			
		return y_pred, logZ_pred

	def clip_norm(self, item, c):
		norm_item = np.linalg.norm(item)
		item= item / max( 1, norm_item / c)

		return item		

	def compute_mse(self, true_mean, true_var):
		# compute the mse.

		#Compute the natural parameters for the global approximation.
		SIG = (self.pISx + self.ISx)
		MU = self.pMISx + self.MISx
		if self.full_cov is False:
			SIG = 1.0 / SIG
			MU = MU * SIG
		else:
			for j in range(self.J):
				SIG[j] = np.linalg.inv(SIG[j])
				MU[j] = np.dot(SIG[j], MU[j])

		#First find the cluster (Gaussian component) by finding the minimum distance between aech true means and the approximated ones.
		label=find_cluster(true_mean, MU, self.J)
		print("These are the correct labels: ", label)
		#Compute the mse for each gaussian component and average them
		err_mean=0
		err_var=0
		for i in range(self.J):
			err_mean += ((true_mean[i] - MU[label[i]]) ** 2).sum() / float(self.J)
			err_var += ((true_var[i] - SIG[label[i]]) ** 2).sum() / float(self.J)
		#mse_mean.append(err_mean)
		#mse_var.append(err_var)

		return err_mean, err_var

	def averaged_KL(self, true_mean, true_var):
		#Compute the KL divergence KL(global approx, ground truth) for each gaussian component and average them.

		#Compute the natural parameters for the global approximation.
		SIG = (self.pISx + self.ISx)
		MU = self.pMISx + self.MISx
		if self.full_cov is False:
			SIG = 1.0 / SIG
			MU = MU * SIG
		else:
			for j in range(self.J):
				SIG[j] = np.linalg.inv(SIG[j])
				MU[j] = np.dot(SIG[j], MU[j])

		#First find the cluster (Gaussian component) by finding the minimum distance between aech true means and the approximated ones
		label=find_cluster(true_mean, MU, self.J)
		
		kl_div=0
		#Compute the KL divergence for a multivariate gaussian KL(global approx, ground truth).
		for i in range(self.J):
			mean_dif=true_mean[i]-MU[label[i]]
			
			inv_truth=np.linalg.inv(true_var[i])
			inv_approx=np.linalg.inv(SIG[label[i]])

			trace_term=np.trace(np.dot(inv_approx, SIG[label[i]]))

			prod_term=np.dot(np.dot(mean_dif, inv_truth), mean_dif)
			#prod_term=np.dot(mean_dif, inv_truth, mean_dif.transpose())
			
			det_truth=np.linalg.det(true_var[i])
			det_approx=np.linalg.det(SIG[label[i]])

			log_det=np.log(det_truth/det_approx)
			
			sum_term=0.5*(trace_term + prod_term + log_det - self.size)/self.J
			kl_div+=sum_term

		return kl_div

	def perturb_mean(self, param, std_noise):

		"""Redraw noise and add it to the natural parameter"""
		noise= np.random.standard_normal(param.shape)*std_noise
		noised_param = param + noise

		return noised_param

	def perturb_cov(self, param, std_noise):

		noise= np.random.standard_normal(param.shape)*std_noise
		noise_triu=np.triu(noise, 0)

		#Make the noise matrix symmetric for the covariance matrix.
		for i in range(noise_triu.shape[0]):
			for j in range(noise_triu.shape[1]):
				for k in range(j, noise_triu.shape[2]):
					noise_triu[i][k][j] = noise_triu[i][j][k]
		
		noised_param = param + noise_triu
		
		#Make the noised covariance matrix  positive definite
		for i in range(noised_param.shape[0]):
			w, v = np.linalg.eig(noised_param[i])
			neg_idx = np.nonzero(w<=0)
			w[neg_idx] = 0.0001
			noised_param[i]=np.dot(v, np.dot(np.diag(w), v.transpose()))
		#print("The noised cov matrix: ", noised_param)
		return noised_param
