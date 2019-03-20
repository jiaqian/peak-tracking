import numpy as np
import pandas as pd
from string_data import string_data
import matplotlib.pyplot as plt
import pdb

def data_generator(freqs,num_spectra,noisy_peaks,freq_limits,inflection_points,main_peaks,params,spectra_shape,noises,spectra_idx,freq_inflection):

# P = np.polyfit(np.hstack((np.array([1]),spectra_shape['inflection_points'][:,1].flatten(),
# 		np.array(spectra_shape['num_spectra']))),np.hstack((np.array([spectra_shape['freq_limits'][0]]),
# 		spectra_shape['inflection_points'][:,0].flatten(),np.array(spectra_shape['freq_limits'][-1]))),
# 		len(freq_inflection)+1)

# main_centers = np.polyval(P, spectra_idx)
# print(main_centers)
#print(np.arange(spectra_shape['num_spectra']))   
#np.polyfit([1,spectra_shape['inflection_points'][:,0]],[],)
# X 80x 8000
# main_centers: 80x1
	X, main_centers = string_data(spectra_shape, main_peaks, noises, 'damped-resonator', params)
	main_centers = np.reshape(main_centers,(main_centers.shape[0],-1))
	return X, main_centers
#print(X.shape)
#print(main_centers.shape)
#X = X / max(X, [], 2);
# plt.figure(figsize=(10,8))
# plt.subplot(2,1,1)
#plt.imshow(X , extent=[freqs[0],freqs[-1],1,num_spectra], aspect='auto')
# plt.subplot(2,1,2)
# plt.plot(freqs, X[20,:],freqs, X[60,:])
# #plt.subplot(3,1,3)
#plt.show()
#print(Omega**2)
# w = spectra_shape['freqs']
# F = main_peaks['F']
# Q = main_peaks['Q']
# for i in range(2):
# 	Omega = main_centers[i]
# 	#print(Omega)
# 	B = F/ np.sqrt((Omega**2 - w**2)**2 + (Omega**2)*(w**2)/(Q**2))
# 	print(B.shape)
#yyy=np.reshape(np.random.normal(size=10),(1,-1))
#print(noises['intensity'][0] + np.sqrt(noises['intensity'][1])*np.reshape(np.random.normal(size=5),(1,-1)))
#print(np.random.normal(size=10).shape)



