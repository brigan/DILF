"""

	model.py: 

		This file contains functions to operate the FitzHugh-Nagumo model of neural membrane. 

"""

import numpy as np; 

def uncompressParams(params): 
	"""	uncompressParams function: 

			This function uncompresses the parameters. 

		Inputs: 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> For more details see: 
					Nagumo J, Arimoto S, Yoshizama S. 
					An active pulse transmission line simulating nerve axon. 
					Proc. IRE 50(10), 2061-2070 (1962). 
				> c: rate-like scale factor accompanying the voltage (inverse for variable w). 
				> J: external current. 
					- In this version of the model we assume that this is constant. 
					- J could change over time and be externally provided. 
				> a, b: two additional constants. 
				> In the original paper, constants satisfy: 
					- "1 > b > 0". 
					- "c^2 > b". 
					- "1 > a > 1 - (2/3)b". 

		Returns: 
			<< params["key"]: each of the parameters separately. 

	"""

	return (params["a"], params["b"], params["c"], params["J"]); 

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
				>> params: dictionary containing the parameters of the model. 
					> These parameters are either provided or initialized somewhere else. 
					> For more details see: 
						Nagumo J, Arimoto S, Yoshizama S. 
						An active pulse transmission line simulating nerve axon. 
						Proc. IRE 50(10), 2061-2070 (1962). 
					> c: rate-like scale factor accompanying the voltage (inverse for variable w). 
					> J: external current. 
						- In this version of the model we assume that this is constant. 
						- J could change over time and be externally provided. 
					> a, b: two additional constants. 
					> In the original paper, constants satisfy: 
						- "1 > b > 0". 
						- "c^2 > b". 
						- "1 > a > 1 - (2/3)b". 

	"""

	params = {}; 
	params["b"] = np.random.uniform(0, 1); 
	params["c"] = np.random.uniform(np.sqrt(params["b"]), 2); 
	params["a"] = np.random.uniform(1-(2./3)*params["b"], 1); 
	params["J"] = np.random.uniform(0, 1); 

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

			This function implements the differential step for the FitzHugh-Nagumo model of membrane potential (see [1]).

            [1] Nagumo J, Arimoto S, Yoshizama S. An active pulse transmission line simulating nerve axon. Proc. IRE
50(10), 2061-2070 (1962).

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 

		Returns: 
			<< F = [Fx, Fy]: differential steps for variables x and y of the model. 
				> This must be composed with dT to integrate numerically. 
				> Since x is a series of numpy arrays, F is evaluated for each of the values provided. 
				> Hence F has the same dimension as x[:,0]. 

	"""

	# Uncompress params and variables: 
	(a, b, c, J) = uncompressParams(params); 
	u = x[:,0]; 
	w = x[:,1]; 

	# Evaluating differential field: 
	Fu = c*(u - np.power(u, 3)/3 + w + J); 
	Fw = (-1./c)*(u - a + b*w); 

	F = np.array([Fu, Fw]).transpose(); 
	return F; 

def dF_a(x, params, kwargs={}): 
	"""	dF_a function: 

			This function computes the derivative of the differential field with respect to the parameter a. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 

		Returns: 
			<< dF_a_: numpy array with the evaluation of the derivative of the field w.r.t. parameter a. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(a, b, c, J) = uncompressParams(params); 

	# Evaluating derivative of the differential field: 
	dF_a_ = np.zeros(x.shape); 
	dF_a_[:,1] = 1./c; 
	return dF_a_; 

def dF_b(x, params, kwargs={}): 
	"""	dF_b function: 

			This function computes the derivative of the differential field with respect to the parameter b. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 

		Returns: 
			<< dF_b_: numpy array with the evaluation of the derivative of the field w.r.t. parameter b. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(a, b, c, J) = uncompressParams(params); 

	# Evaluating derivative of the differential field: 
	dF_b_ = np.zeros(x.shape); 
	dF_b_[:,1] = -1.*x[:,1]/c; 
	return dF_b_; 

def dF_c(x, params, kwargs={}): 
	"""	dF_c function: 

			This function computes the derivative of the differential field with respect to the parameter c. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 

		Returns: 
			<< dF_c_: numpy array with the evaluation of the derivative of the field w.r.t. parameter c. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(a, b, c, J) = uncompressParams(params); 
	u = x[:,0]; 
	w = x[:,1]; 

	dF_u_c_ = (u - np.power(u, 3)/3 + w + J); 
	dF_w_c_ = (1./(c**2))*(u - a + b*w); 
	dF_c_ = np.array([dF_u_c_, dF_w_c_]).transpose(); 

	return dF_c_; 

def dF_J(x, params, kwargs={}): 
	"""	dF_J function: 

			This function computes the derivative of the differential field with respect to the parameter J. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 

		Returns: 
			<< dF_c_: numpy array with the evaluation of the derivative of the field w.r.t. parameter J. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(a, b, c, J) = uncompressParams(params); 
	
	dF_J_ = np.zeros(x.shape); 
	dF_J_[:,0] = c; 

	return dF_J_; 

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
	dFDict["a"] = dF_a; 
	dFDict["b"] = dF_b; 
	dFDict["c"] = dF_c; 
	dFDict["J"] = dF_J; 

	if ("toFitParams" in kwargs.keys()): 
		toFitParams = kwargs["toFitParams"]; 
	else: 
		toFitParams = dFDict.keys(); 

	dF_params_ = {}; 
	for key in toFitParams: 
		dF_params_[key] = dFDict[key](x, params, kwargs); 

	return dF_params_; 
