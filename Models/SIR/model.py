"""

	model.py: 

		This file contains functions to operate the Mira-Paredes model of language competition. 

"""

import numpy as np; 

def uncompressParams(params, kwargs={}): 
	"""	uncompressParams function: 

			This function uncompresses the parameters. 

		Inputs: 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> beta: infection rate. 
				> gamma: recovery rate. 

		Returns: 
			<< params["key"]: each of the parameters separately. 

	"""

	return (params["beta"], params["gamma"]); 

def unpackParams(params, kwargs={}): 
	"""	unpackParams function: 

			This function does the same as uncompressParams() so we don't have to remember which of the two options has been
			implemented. 

			Inputs: 
				>> params: dictionary containing the parameters of the model. 
					> These parameters are either provided or initialized somewhere else. 
					> beta: infection rate. 
					> gamma: recovery rate. 

			Returns: 
				<< params["key"]: each of the parameters separately. 

	"""

	return (params["beta"], params["gamma"]); 


def initializeParams(kwargs={}): 
	"""	initializeParams function: 

			This function initializes the parameters for this model. Note that this function is quite specific for each
			model, which might have special rules or boundaries. Whatever the model, this function always returns a
			dictionary with an entry for each of the parameters. 

			Inputs: 
				>> kwargs: dictionary (might be empty) with key word arguments that the function might use. 
					> Key word is "pName" (parameter name), then: 
						-- If kwargs["pName"] is a float or integer, this sets the value of the parameter. 
						-- If kwargs["pName"] is a list, parameter is chosen uniformly from within that range. 

			Returns: 
				<< params: Dictionary containing names and values of all the parameters of the model. 
					< beta: infection rate. 
					< gamma: recovery rate. 

	"""

	params = {}; 
	params["beta"] = np.random.uniform(0., 0.1); 
	params["gamma"] = np.random.uniform(0., 0.1); 

	for key in kwargs.keys(): 
		if (key in params.keys() and isinstance(kwargs[key], (int, long, float, complex))): 
			params[key] = kwargs[key]; 
		if (key in params.keys() and type(kwargs[key])==list): 
			params[key] = np.random.uniform(kwargs[key][0], kwargs[key][1]); 

	return params; 

def initParams(kwargs={}): 
	"""	initParams function: 

			Calls initializeParams(kwargs). 

	"""
	return initializeParams(kwargs); 
	
def F(x, params, kwargs={}): 
	"""	F function: 

			This function implements the differential step for the SIR model. 

			Inputs: 
				>> x: Values at which we wish to evaluate the model. 
					> The caller of this function must make sure to provide the adequate variables. 
					> This is a nVar*nPoints numpy array. 
				>> params: dictionary containing the parameters of the model. 
					> These parameters are either provided or initialized somewhere else. 
					> beta: infection rate. 
					> gamma: recovery rate. 

			Returns: 
				<< F = [Fs, Fi]: differential steps for variables s (susceptible) and i (infected) of the model. 
					> This must be composed with dT to integrate numerically. 
					> Since s is a series of numpy arrays, F is evaluated for each of the values provided. 
					> Hence F has the same dimension as x[:,0]. 

	"""

	# Uncompress params and variables: 
	(beta, gamma) = uncompressParams(params); 
	s_ = x[:,0]; 
	i_ = x[:,1]; 
	r_ = x[:,2]; 

	# Evaluating differential field: 
	Fs = -beta*np.multiply(i_, s_); 
	Fi = beta*np.multiply(i_, s_) - gamma*i_; 
	Fr = gamma*i_; 
	F = np.array([Fs, Fi, Fr]).transpose(); 
	return F; 


def dF_beta(x, params, kwargs={}): 
	"""	dF_beta function: 

			This function computes the derivative of the differential field with respect to the parameter beta. 

			Inputs: 
				>> x Values at which we wish to evaluate the model. 
					> The caller of this function must make sure to provide the adequate variables. 
					> This is a nVar*nPoints numpy array. 
				>> params: dictionary containing the parameters of the model. 
					> These parameters are either provided or initialized somewhere else. 
					> beta: infection rate. 
					> gamma: recovery rate. 

			Returns: 
				<< dF_beta_: numpy array with the evaluation of the derivative of the field w.r.t. parameter beta. 
					> This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(beta, gamma) = uncompressParams(params); 
	s_ = x[:,0]; 
	i_ = x[:,1]; 
	r_ = x[:,2]; 

	# Evaluating derivative of the differential field: 
	dF_beta_s = -np.multiply(s_, i_); 
	dF_beta_i = np.multiply(s_, i_); 
	dF_beta_r = np.zeros(r_.shape); 
	dF_beta_ = np.array([dF_beta_s, dF_beta_i, dF_beta_r]).transpose(); 

	return dF_beta_; 

def dF_gamma(x, params, kwargs={}): 
	"""	dF_gamma function: 

			This function computes the derivative of the differential field w.r.t. the parameter gamma. 

			Inputs: 
				>> x Values at which we wish to evaluate the model. 
					> The caller of this function must make sure to provide the adequate variables. 
					> This is a nVar*nPoints numpy array. 
				>> params: dictionary containing the parameters of the model. 
					> These parameters are either provided or initialized somewhere else. 
					> beta: infection rate. 
					> gamma: recovery rate. 

			Returns: 
				<< dF_gamma_: numpy array with the evaluation of the derivative of the field w.r.t. parameter gamma. 
					> This is evaluated for each of the values of x provided. 
	"""

	# Uncompress params and variables: 
	(beta, gamma) = uncompressParams(params); 
	s_ = x[:,0]; 
	i_ = x[:,1]; 
	r_ = x[:,2]; 

	# Evaluating derivative of the differential field: 
	dF_gamma_s = np.zeros(s_.shape); 
	dF_gamma_i = -i_; 
	dF_gamma_r = i_; 
	dF_gamma_ = np.array([dF_gamma_s, dF_gamma_i, dF_gamma_r]).transpose(); 

	return dF_gamma_; 	


def dF_params(x, params, kwargs={}): 
	"""	dF_params function: 

			This function computes the derivative of the flux w.r.t. all of the parameters, which are returned in a
			dictionary.

			Inputs: 
				>> x: Values at which we wish to evaluate the model. 
					> The caller of this function must make sure to provide the adequate variables. 
					> This is a nVar*nPoints numpy array. 
				>> params: dictionary containing the parameters of the model. 
					> These parameters are either provided or initialized somewhere else. 
					> beta: infection rate. 
					> gamma: recovery rate. 

			Returns: 
				<< dF_params_: dictionary containing the derivative of the model flux w.r.t. all of the parameters. 

	"""

	dFDict = {}; 
	dFDict["beta"] = dF_beta; 
	dFDict["gamma"] = dF_gamma; 

	if ("toFitParams" in kwargs.keys()): 
		toFitParams = kwargs["toFitParams"]; 
	else: 
		toFitParams = dFDict.keys(); 

	dF_params_ = {}; 
	for key in toFitParams: 
		dF_params_[key] = dFDict[key](x, params, kwargs); 

	return dF_params_; 


