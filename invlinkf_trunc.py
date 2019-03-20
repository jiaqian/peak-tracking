import numpy as np
from scipy.special import erfinv
from scipy.special import erf

def invlinkf_trunc(param, Z, sigmaT, mu, a, sigmaeta):

	f = erfinv(Z*(1 + erf(param/(np.sqrt(2)*sigmaeta))) + erf((a - mu)/(np.sqrt(2)*sigmaT)))*np.sqrt(2)*sigmaT + mu

	return f
