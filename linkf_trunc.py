from scipy.special import erf
from scipy.special import erfinv
import numpy as np
import pdb
def linkf_trunc(param, Z, sigmaT, mu, a, sigmaeta):
	
	param = param.reshape((len(param),-1))
	mu = mu.reshape((len(mu),-1))
	Z = Z.reshape((len(Z),-1))
	f = erfinv(1/Z*(erf((param - mu)/(np.sqrt(2)*sigmaT)) - erf((a - mu)/(np.sqrt(2)*sigmaT))) - 1)*np.sqrt(2)*sigmaeta
	#pdb.set_trace()
	return f