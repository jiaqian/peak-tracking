import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from synthetic_data import data_generator
from bayes_opt import BayesianOptimization
from invlinkf_trunc import invlinkf_trunc
from damped_model import damped_model
from bayes_gpp import bayes_gpp
import math
import pdb


# generate data
freqs = np.linspace(1, 2000, 8000)
num_spectra = 80
noisy_peaks = 3
freq_limits = [300, 1000]
inflection_points = np.array([[350 ,10], [700, 60], [900, 70]])
main_peaks = {'F':10e-3, 'Q': 100}
params = {'decorr':True}
spectra_shape = {'freqs':freqs, 'num_spectra': num_spectra,
'freq_limits':freq_limits, 'inflection_points':inflection_points}
noises = {'freq': 0, 'intensity': [0, 5e-5], 'width': 0, 'peaks': noisy_peaks}
spectra_idx = np.arange(spectra_shape['num_spectra'])
freq_inflection = spectra_shape['inflection_points'][:,0]
[X, main_centers] = data_generator(freqs,num_spectra,noisy_peaks,freq_limits,inflection_points,
	main_peaks,params,spectra_shape,noises,spectra_idx,freq_inflection)
#print(main_centers.shape)
pdb.set_trace()
main_centers = main_centers.reshape(1,-1)
dbg={'true_values' :False,'print_kernels':True}
main_center_indcs = np.argmin(np.abs(np.matlib.repmat(freqs,X.shape[0],1)-
	np.matlib.repmat(np.transpose(main_centers),1,len(freqs))),axis=1)
main_center_indcs = main_center_indcs.reshape((main_center_indcs.shape[0],-1))
# Initial resonance frequency 
S_ini = main_centers+ 20 * np.random.normal(size=len(main_centers))
#print(main_centers.shape)
Q_ini = np.ones((1, X.shape[0]),dtype=int) * 50
ini_center_idcs = np.argmin(np.abs(np.matlib.repmat(freqs,X.shape[0],1))
	-np.matlib.repmat(np.transpose(S_ini),1,len(freqs)),axis =1)
def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols
#print (X.shape)
F_ini = X.ravel()[sub2ind(X.shape, np.arange(X.shape[0]), np.transpose(ini_center_idcs))]
F_ini = F_ini.reshape((1,-1))

true_S = main_centers
true_Q = np.ones((1, X.shape[0]),dtype=int) * main_peaks['Q']
true_F = np.ones((1,  X.shape[0]), dtype=int) * main_peaks['F']

if dbg['true_values']:
	theta_ini = np.hstack((np.transpose(true_F), np.transpose(true_S), np.transpose(true_Q)))
else:
    theta_ini = np.hstack((np.transpose(F_ini), np.transpose(S_ini), np.transpose(Q_ini)))

print(math.pi)
#theta_est,p = bayes_gpp(X, freqs, theta_ini, dbg)
#prig_s = invlinkf_trunc(p['R_Q'] * theta_est[:,2], p['ZQ'], p['sigma2Q'], p['muQ'],p['aQ'], p['sign2_Q'])
#params['decorr'] = true;
#B = damped_model(freqs, theta_est(1,1), orig_S(1), orig_Q(1), params);
#print(np.eye(2,3))
#print(main_centers)
#plt.plot(X[10,:])
#plt.show()
#data = np.reshape(np.arange(num_spectra)+1, (num_spectra,-1))
#data = np.hstack((np.reshape(np.arange(num_spectra)+1, (num_spectra,-1)), main_centers))
#data = np.hstack((main_centers,A))
#pbounds = {'x':(1,num_spectra),'y':(1,8000)}
#print(data.shape)

# def b_func(w, F, Omega, Q):
# 	B = F/ np.sqrt((Omega**2 - w**2)**2 + (Omega**2)*(w**2)/(Q**2))
# 	B = B * Omega**2 / Q
# 	return B

# optimizer = BayesianOptimization(
#     f=b_func,
#     pbounds={'F': (10e-4, 10e-2), 'Omega': (1, 8000), 'w': (1, 8000),'Q':(10,500)},
#     verbose=2,
#     random_state=1,
# )
# optimizer.maximize(alpha=1e-3)





# 
# name = 'ucb' # or 'ei'or 'poi' 
# if name == 'ucb':
# 	ucb(x,gp,kappa)
# elif name == 'ei':
# 	ei(x, gp, y_max, xi)
# elif name == 'poi':
# 	poi(x, gp, y_max, xi)


# gp = GaussianProcessRegressor(
#             kernel=Matern(nu=0.05,length_scale=1e-5),
#             alpha=1e-6,
#             normalize_y=True,
#             n_restarts_optimizer=25,
            
#         )
# N = data.shape[0]
#m =2
#print(data[:m,1])
# gp_ = gp.fit(np.reshape(data[:m,0],(data[:m,0].shape[0],-1)),np.reshape(data[:m,1],(data[:m,1].shape[0],-1)))
# kappa = 0.1
# xi = 0.1
# y_max = 8000
# re = np.zeros((N-m,1),dtype=float)
# for i in range(m,N-m+1):
# 	mean, std = gp.predict(data[i,0], return_std = True)
# 	z = (mean - y_max - xi)/std
# 	re[i-m,0] = norm.cdf(z)
# print(re)
#mean,std = gp_.predict(3,return_std=True)
#print('mean is ', mean.item())
#print('std is ',std.item())
#print(data[:75,1])

# def ucb (x, gp, kappa):
# 	# GP upper confidence bound
# 	mean, std = gp.predict(x, return_std = True)
# 	return mean + kappa * std 


# def ei(x, gp, y_max, xi):
# 	# expected improvement
# 	mean, std = gp.predict(x, return_std = True)
# 	z = (mean - y_max - xi)/std
# 	return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

# def poi(x, gp, y_max, xi):

# 	#
# 	mean, std = gp.predict(x, return_std=True)
# 	z = (mean - y_max - xi)/std
# 	return norm.cdf(z)




