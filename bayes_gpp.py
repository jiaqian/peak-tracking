
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.special import erf
from linkf_trunc import linkf_trunc
from numpy.linalg import inv
#from bayes_gpp_target import bayes_gpp_target
from linkf_trunc import linkf_trunc
import pdb
from scipy import optimize 
def bayes_gpp(X, freqs, theta_ini, dbg):

	params = {'aF':0,'bF':1000,'muF':theta_ini[:,0],'sigma2F':1e6}

	# GPP
	y = np.arange(X.shape[0])
	y = np.expand_dims(y,axis=0)
	
	y_dist2 = cdist(np.transpose(y),np.transpose(y))
	# user input
	#pdb.set_trace()
	params['sigf2_S']=1 
	params['sigf2_Q'] = 1
	params['lenscl_S'] = 3
	params['lenscl_Q'] = 5
	params['sign2_S'] = 1 # mapped space
	params['sign2_Q'] = 1 # mapped space

	# square exponential kernel definition
	#pdb.set_trace()
	#I = np.eye(y.shape[0],y.shape[1])
	# Peak location, S
	K_S = params['sigf2_S'] * np.exp(-0.5 * y_dist2 / params['lenscl_S']**2)
	#C_S = K_S+params.sign2_S*I; % covariance matrix
	C_S = K_S + np.diag(params['sign2_S'] * np.ones(y.shape))
	# Inverse Q factor, 1/Q
	K_Q = params['sigf2_Q'] * np.exp(-0.5 * y_dist2 / params['lenscl_Q']**2)
	#C_Q = K_Q+params.sign2_Q*I; % covariance matrix
	C_Q = K_Q + np.diag(params['sign2_Q'] * np.ones(y.shape))



	# if dbg.['print_kernels']:
 #    	#figure('Name', 'Kernels')
 #    	subplot(2,1,1)
 #    	plt.imagesc(1:size(K_S,1), 1:size(K_S,1), K_S)
 #    	plt.title('Resonance Frequency Kernel')

 #    	subplot(2,1,2)
 #    	imagesc(1:size(K_Q,1), 1:size(K_Q,1), K_Q)
 #    	title('Q factor Kernel')

 #    	f = findobj('Type', 'Figure', 'Name', 'Main')
 #    	figure(f)


	params['R_Q'] = np.transpose(np.linalg.cholesky(C_Q))
	params['R_S'] = np.transpose(np.linalg.cholesky(C_S))
    
# Truncated for Q
	params['aQ'] = 0.5
	params['bQ'] = 1e6
	params['muQ'] = theta_ini[:,2]
	params['sigma2Q'] = (1e3)**2
	# eq 2
	params['ZQ'] = 0.5 * (1 + erf((params['bQ'] - params['muQ'])/(np.sqrt(2)*params['sigma2Q']))) -0.5*(1 +erf((params['aQ'] - params['muQ'])/(np.sqrt(2)*params['sigma2Q'])))

# Truncated for S
	params['aS'] = np.min(freqs)
	params['bS'] = np.max(freqs)
	params['muS'] = theta_ini[:,1]
	params['sigma2S'] = ((np.max(freqs) - np.min(freqs))/2)**2

	params['ZS'] = 0.5 * (1 + erf((params['bS'] - params['muS'])/(np.sqrt(2)*params['sigma2S']))) - 0.5*(1 +erf((params['aS'] - params['muS'])/(np.sqrt(2)*params['sigma2S'])))
#Sample noise
	params['sig2B'] = (1e-7)**2

	uu = inv(params['R_S']).dot(linkf_trunc(theta_ini[:,1],params['ZS'],params['sigma2S'],theta_ini[:,1],params['aS'],params['sign2_S']))
	#pdb.set_trace()

	vv = inv(params['R_Q']).dot(linkf_trunc(theta_ini[:,2], params['ZQ'], params['sigma2Q'], theta_ini[:,2], params['aQ'], params['sign2_Q']))

	#pdb.set_trace()
	theta = np.hstack((np.hstack((np.expand_dims(theta_ini[:,0],axis=1), uu)),vv))


	params['theta_ini'] = theta_ini
	#func = bayes_gpp_target(X, freqs, theta, theta_ini, params)
	# [theta, fval, exit_flat, output] = minimizer(func, theta, [], [], [], [], ...
	#             repmat([0 -inf -inf], size(theta,1), 1), ...
	#             repmat([inf inf inf], size(theta,1), 1), [], options);

	S_orig = invlinkf_trunc(p['R_S']*theta[:,1], p['ZS'], p['sigma2S'], theta_ini[:,1], p['aS'], p['sign2_S'])
	Q_orig = invlinkf_trunc(p['R_Q']*theta[:,2], p['ZQ'], p['sigma2Q'], theta_ini[:,2], p['aQ'], p['sign2_Q'])
	theta_orig_space = np.hstack((np.expand_dims(theta_ini[:,0],axis=1), S_orig, Q_orig))

	def bayes_gpp_target(X, freqs, theta, theta_ini, p):
		Nt_logpriorF = -np.log((1/2*(1 + (erf((p['bF'] - p['muF'])/(np.sqrt(2)*p['sigma2F'])))) - 
		1/2*(1 + erf((p['aF'] - p['muF'])/(np.sqrt(2)*p['sigma2F']))))) - np.log(2*math.pi*p['sigma2F']) -(0.5/p['sigma2F'])*np.sum((theta[:,0] - p['muF'])**2)
		
		#Vectorized representation of B for all spectra.
		B = (np.expand_dims(theta_orig_space[:,1],axis=1)**2 / np.expand_dims(theta_orig_space[:,2],axis=1) )* np.expand_dims(theta_orig_space[:,0],axis=1) / np.sqrt((np.expand_dims(theta_orig_space[:,1],axis=1)**2-freqs**2)**2 + (np.expand_dims(theta_orig_space[:,1],axis=1)**2*(freqs**2))/np.expand_dims(theta_orig_space[:,2],axis=1)**2)
		cc = np.sum((X - B)**2, axis=1)
		cc = cc.reshape(len(cc),-1)
		loglik = -( X.shape[0]+ X.shape[1]) * 1/2*(np.log(2*math.pi)/2 + np.log(p['sig2B'])) -1/(2*p['sig2B'])*np.sum(cc, axis=0) # check the axis
		#logpost = loglikelihood(X|eta) + dot(eta, eta)
		logpost = loglik +0.5*(theta[:,1].dot(theta[:,1]))+ 0.5*(theta[:,2].dot(theta[:,2]) ) + np.sum(Nt_logpriorF)
		F = logpost
		#pdb.set_trace()
		return F

	#pdb.set_trace()
	optimize_result = optimize.minimize(bayes_gpp_target,X, freqs, theta, theta_ini, params)#(theta, func, freqs, params, X)

	theta_estimate = theta

	











