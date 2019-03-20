import numpy as np
import pdb
def damped_model(w, F, Omega, Q, params):
	######input######
	# w: freq
	# F: amplitude at peak location
	# Omega: peak location
	# Q: Q factor
	#####output######
	# B: a vector
	#pdb.set_trace()
	B = F/ np.sqrt((Omega**2 - w**2)**2 + (Omega**2)*(w**2)/(Q**2))
	B = B * Omega**2 / Q
	
	return B
