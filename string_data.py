import numpy as np
import pandas as pd
from damped_model import damped_model
import pdb



def string_data(spectra_shape, main_peaks, noises,func,params):
	# sepctra_shape: dictionary
	# main_peaks: dictionary, where store F and Q
	# noise: dictionary, where store
	# func:
	# parasm: 
	X = np.zeros((spectra_shape['num_spectra'],len(spectra_shape['freqs'])))
	#A = np.zeros((spectra_shape['num_spectra'],1))
	spectra_idx = np.arange(spectra_shape['num_spectra'])
	freq_inflection = spectra_shape['inflection_points'][:,0]
	spectra_inflection = spectra_shape['inflection_points'][:,1]

	P = np.polyfit(np.hstack((np.array([1]),spectra_shape['inflection_points'][:,1].flatten(),
		np.array(spectra_shape['num_spectra']))),np.hstack((np.array([spectra_shape['freq_limits'][0]]),
		spectra_shape['inflection_points'][:,0].flatten(),np.array(spectra_shape['freq_limits'][-1]))),
		len(freq_inflection)+1)

	
	main_centers = np.polyval(P, spectra_idx) # values of y


	if func == 'pseudo-voigt':
		print('******** not consider it now ********')
	elif func == 'damped-resonator':

		for i in range(X.shape[0]):
			X[i,:] = damped_model(spectra_shape['freqs'], main_peaks['F'], main_centers[i] + 
					noises['freq']*np.random.normal(size=1), main_peaks['Q'] + noises['width']*np.random.normal(size=1),params)
			# if noises['peaks'] >0:
			# 	spur_peaks = np.random.randint(noises['peaks']-1,size=1)
			# 	for k in range(spur_peaks):
			# 		print('later complete it')
			X[i,:] = X[i,:] + (noises['intensity'][0] + noises['intensity'][1]* 
				np.reshape(np.random.normal(size=X.shape[1]),(1,-1)))

	return X, main_centers

