"""

	model.py: 

		This file contains functions to operate the Mira-Paredes model of language competition. 

"""

import numpy as np; 

def uncompressParams(params): 
	"""	uncompressParams function: 

			This function uncompresses the parameters. 

		Inputs: 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> alpha_F: production rate of variable F. 
				> n: number of polymers for activation (?). 
				> K_F: (?). 
				> gamma_F: decay rate of variable F. 
				> alpha_E: production rate of variable E. 
				> gamma_E: related to decay of variable E. 

		Returns: 
			<< params["key"]: each of the parameters separately. 

	"""

	return (params["alpha_F"], params["n"], params["K_F"], params["gamma_F"], params["alpha_E"], params["gamma_E"]); 

def initializeParams(kwArgs={}): 
	"""	initializeParams function: 

			This function initializes the parameters for this model. Note that this function is quite specific for each
			model, which might have special rules or boundaries. Whatever the model, this function always returns a
			dictionary with an entry for each of the parameters. 

		Inputs: 
			>> kwArgs: dictionary (might be empty) with key word arguments that the function might use. 
				> Key word is "pName" (parameter name), then: 
					-- If kwArgs["pName"] is a float or integer, this sets the value of the parameter. 
					-- If kwArgs["pName"] is a list, parameter is chosen uniformly from within that range. 

		Returns: 
			<< params: dictionary containing the parameters of the model. 
				< alpha_F: production rate of variable F. 
				< n: number of polymers for activation (?). 
				< K_F: (?). 
				< gamma_F: decay rate of variable F. 
				< alpha_E: production rate of variable E. 
				< gamma_E: related to decay of variable E. 

	"""

	params = {}; 
	params["alpha_F"] = np.random.uniform(0, 1); 
	params["n"] = np.random.choice([1, 2, 3, 4]); 
	params["K_F"] = np.random.uniform(0, 1); 
	params["gamma_F"] = np.random.uniform(0, 1); 
	params["alpha_E"] = np.random.uniform(0, 1); 
	params["gamma_E"] = np.random.uniform(0, 1); 

	for key in kwargs.keys(): 
		if (key in params.keys() and isinstance(kwargs[key], (int, long, float, complex))): 
			params[key] = kwargs[key]; 
		if (key in params.keys() and type(kwargs[key])==list): 
			params[key] = np.random.uniform(kwargs[key][0], kwargs[key][1]); 

	return params; 

def initParams(kwArgs={}): 
	"""	initParams function: 

			Calls initializeParams(kwArgs). 

	"""
	return initializeParams(kwArgs); 

def F(x, params, kwArgs={}): 
	"""	F function: 

			This function implements the differential step for the Toscano models (reference missing, contact Caslos
			Toscano, who lives in Tuscany).

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
				> x = [F, E] in the original model. 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> alpha_F: production rate of variable F. 
				> n: number of polymers for activation (?). 
				> K_F: (?). 
				> gamma_F: decay rate of variable F. 
				> alpha_E: production rate of variable E. 
				> gamma_E: related to decay of variable E. 
			>> kwArgs: 
				> Dictionary containing extra variables needed to compute F. 
				> This model requires that we provide an extra array with the time marks. 

		Returns: 
			<< F = [Fx, Fy]: differential steps for variables x and y of the model. 
				> This must be composed with dT to integrate numerically. 
				> Since x is a series of numpy arrays, F is evaluated for each of the values provided. 
				> Hence F has the same dimension as x[:,0]. 

	"""

	# Uncompress params and variables: 
	(a_F, n, K_F, g_F, a_E, g_E) = uncompressParams(params); 
	t = kwArgs["time"]; 
	E = kwArgs["E"]; 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating differential field: 
	x_powN = np.power(x_, n); 
	y_powN = np.power(y_, n); 
	Fx = a_F*np.divide(y_powN, (K_F**n + y_powN)) - g_F*x_; 
	Fy = (a_E/g_E)*(1. - np.exp(-g_E*t))*E; 

	F = np.array([Fx, Fy]).transpose(); 
	return F; 

def dF_alpha_F(x, params, kwArgs={}): 
	"""	dF_alpha_F function: 

			This function computes the derivative of the differential field with respect to the parameter alpha_K. 

			Inputs: 
				>> x: Values at which we wish to evaluate the model. 
					> The caller of this function must make sure to provide the adequate variables. 
					> This is a nVar*nPoints numpy array. 
				>> params: dictionary containing the parameters of the model. 
					> These parameters are either provided or initialized somewhere else. 
					> alpha_F: production rate of variable F. 
					> n: number of polymers for activation (?). 
					> K_F: (?). 
					> gamma_F: decay rate of variable F. 
					> alpha_E: production rate of variable E. 
					> gamma_E: related to decay of variable E. 

			Returns: 
				<< dF_alpha_F_: numpy array with the evaluation of the derivative of the field w.r.t. parameter alpha_F. 
					< This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(a_F, n, K_F, g_F, a_E, g_E) = uncompressParams(params); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating derivative of the differential field: 
	y_powN = np.power(y_, n); 
	dF_alpha_F_ = np.zeros(x.shape); 
	dF_alpha_F_[:,0] = np.divide(y_powN, (K_F**n + y_powN)); 
	return dF_alpha_F_; 

def dF_n(x, params, kwArgs={}): 
	"""	dF_n function: 

			This function computes the derivative of the differential field with respect to the parameter alpha_K. 

			Inputs: 
				>> x: Values at which we wish to evaluate the model. 
					> The caller of this function must make sure to provide the adequate variables. 
					> This is a nVar*nPoints numpy array. 
				>> params: dictionary containing the parameters of the model. 
					> These parameters are either provided or initialized somewhere else. 
					> alpha_F: production rate of variable F. 
					> n: number of polymers for activation (?). 
					> K_F: (?). 
					> gamma_F: decay rate of variable F. 
					> alpha_E: production rate of variable E. 
					> gamma_E: related to decay of variable E. 

			Returns: 
				<< dF_n_: numpy array with the evaluation of the derivative of the field w.r.t. parameter n. 
					< This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(a_F, n, K_F, g_F, a_E, g_E) = uncompressParams(params); 
	K_F_powN = K_F**n; 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating derivative of the differential field: 
	y_powN = np.power(y_, n); 
	dF_n_ = np.zeros(x.shape); 
	dF_n_[:,0] = np.divide(a_F*K_F_powN*np.multiply(y_powN, np.log(y_/K_F)), np.power((K_F_powN+y_powN),2)); 

	return dF_n_; 

def dF_K_F(x, params, kwArgs={}): 
	"""	dF_K_F function: 

			This function computes the derivative of the differential field with respect to the parameter alpha_K. 

			Inputs: 
				>> x: Values at which we wish to evaluate the model. 
					> The caller of this function must make sure to provide the adequate variables. 
					> This is a nVar*nPoints numpy array. 
				>> params: dictionary containing the parameters of the model. 
					> These parameters are either provided or initialized somewhere else. 
					> alpha_F: production rate of variable F. 
					> n: number of polymers for activation (?). 
					> K_F: (?). 
					> gamma_F: decay rate of variable F. 
					> alpha_E: production rate of variable E. 
					> gamma_E: related to decay of variable E. 

			Returns: 
				<< dF_K_F_: numpy array with the evaluation of the derivative of the field w.r.t. parameter K_F. 
					< This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(a_F, n, K_F, g_F, a_E, g_E) = uncompressParams(params); 
	K_F_powN = K_F**n; 
	K_F_powN_ = K_F**(n-1); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating derivative of the differential field: 
	y_powN = np.power(y_, n); 
	dF_K_F_ = np.zeros(x.shape); 
	dF_K_F_[:,0] = -n*a_F*K_F_powN_*np.divide(y_powN, np.power(K_F_powN+y_powN, 2)); 

	return dF_K_F_; 

def dF_gamma_F(x, params, kwArgs={}): 
	"""	dF_gamma_F function: 

			This function computes the derivative of the differential field with respect to the parameter alpha_K. 

			Inputs: 
				>> x: Values at which we wish to evaluate the model. 
					> The caller of this function must make sure to provide the adequate variables. 
					> This is a nVar*nPoints numpy array. 
				>> params: dictionary containing the parameters of the model. 
					> These parameters are either provided or initialized somewhere else. 
					> alpha_F: production rate of variable F. 
					> n: number of polymers for activation (?). 
					> K_F: (?). 
					> gamma_F: decay rate of variable F. 
					> alpha_E: production rate of variable E. 
					> gamma_E: related to decay of variable E. 

			Returns: 
				<< dF_gamma_F_: numpy array with the evaluation of the derivative of the field w.r.t. parameter gamma_F. 
					< This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(a_F, n, K_F, g_F, a_E, g_E) = uncompressParams(params); 
	x_ = x[:,0]; 

	# Evaluating derivative of the differential field: 
	dF_gamma_F_ = np.zeros(x.shape); 
	dF_gamma_F_[:,0] = -x_; 
	return dF_gamma_F_; 

def dF_alpha_E(x, params, kwArgs={}): 
	"""	dF_alpha_E function: 

			This function computes the derivative of the differential field with respect to the parameter alpha_K. 

			Inputs: 
				>> x: Values at which we wish to evaluate the model. 
					> The caller of this function must make sure to provide the adequate variables. 
					> This is a nVar*nPoints numpy array. 
				>> params: dictionary containing the parameters of the model. 
					> These parameters are either provided or initialized somewhere else. 
					> alpha_F: production rate of variable F. 
					> n: number of polymers for activation (?). 
					> K_F: (?). 
					> gamma_F: decay rate of variable F. 
					> alpha_E: production rate of variable E. 
					> gamma_E: related to decay of variable E. 

			Returns: 
				<< dF_alpha_E_: numpy array with the evaluation of the derivative of the field w.r.t. parameter alpha_E. 
					< This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(a_F, n, K_F, g_F, a_E, g_E) = uncompressParams(params); 
	t = kwArgs["time"]; 
	E = kwArgs["E"]; 

	# Evaluating derivative of the differential field: 
	dF_alpha_E_ = np.zeros(x.shape); 
	dF_alpha_E_[:,1] = (E/g_E)*(1.-np.exp(-g_E*t)); 

	return dF_alpha_E_; 

def dF_gamma_E(x, params, kwArgs={}): 
	"""	dF_gamma_E function: 

			This function computes the derivative of the differential field with respect to the parameter alpha_K. 

			Inputs: 
				>> x: Values at which we wish to evaluate the model. 
					> The caller of this function must make sure to provide the adequate variables. 
					> This is a nVar*nPoints numpy array. 
				>> params: dictionary containing the parameters of the model. 
					> These parameters are either provided or initialized somewhere else. 
					> alpha_F: production rate of variable F. 
					> n: number of polymers for activation (?). 
					> K_F: (?). 
					> gamma_F: decay rate of variable F. 
					> alpha_E: production rate of variable E. 
					> gamma_E: related to decay of variable E. 

			Returns: 
				<< dF_gamma_E_: numpy array with the evaluation of the derivative of the field w.r.t. parameter gamma_E. 
					< This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(a_F, n, K_F, g_F, a_E, g_E) = uncompressParams(params); 
	t = kwArgs["time"]; 
	E = kwArgs["E"]; 

	# Evaluating derivative of the differential field: 
	dF_gamma_E_ = np.zeros(x.shape); 
	dF_gamma_E_[:,1] = -(E/g_E)*(1.-np.multiply(np.exp(-g_E*t), t-1)); 

	return dF_gamma_E_; 

def dF_params(x, params, kwargs={}): 
	"""	dF_params function: 

			This function computes the derivative of the flux w.r.t. all of the parameters, which are returned in a
			dictionary.

			Inputs: 
				>> x: Values at which we wish to evaluate the model. 
					> The caller of this function must make sure to provide the adequate variables. 
					> This is a nVar*nPoints numpy array. 
				>> params: dictionary containing the parameters of the model. 

			Returns: 
				<< dF_params_: dictionary containing the derivative of the model flux w.r.t. all of the parameters. 

	"""

	dFDict = {}; 
	dFDict["alpha_F"] = dF_alpha_F; 
	dFDict["n"] = dF_n; 
	dFDict["K_F"] = dF_K_F; 
	dFDict["gamma_F"] = dF_gamma_F; 
	dFDict["alpha_E"] = dF_alpha_E; 
	dFDict["gamma_E"] = dF_gamma_E; 

	if ("toFitParams" in kwargs.keys()): 
		toFitParams = kwargs["toFitParams"]; 
	else: 
		toFitParams = dFDict.keys(); 

	dF_params_ = {}; 
	for key in toFitParams: 
		dF_params_[key] = dFDict[key](x, params, kwargs); 

	return dF_params_; 

