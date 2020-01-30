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
				> a: exponent that weights the attracting population. 
				> c: scale factor that speeds up or slows down the simulations. 
				> k: inter-linguistic similarity. 
				> s: prestige of the language whose time series is stored in x_. 

		Returns: 
			<< params["key"]: each of the parameters separately. 

	"""

	return (params["a"], params["c"], params["k"], params["s"]); 

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
					< a: exponent weighting the attracting population. 
					< c: scale factor to speed up or slow down the time evolution. 
					< k: inter linguistic similarity. 
					< s: prestige of the language associated to the first variable. 

	"""

	params = {}; 
	params["a"] = np.random.uniform(1, 2); 
	params["c"] = np.random.uniform(0.01, 2); 
	params["k"] = np.random.uniform(0, 1); 
	params["s"] = np.random.uniform(0, 1); 

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

			This function implements the differential step for the Mira-Paredes model of language competition (see [1]).

            [1] Otero-Espinar MV, Seoane LF, Nieto JJ, Mira J. An analytic solution of a model of language competition
with bilingualism and interlinguistic similarity. Phys. D, 264, pp.17-26 (2013). 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> a: exponent that weights the attracting population. 
				> c: scale factor that speeds up or slows down the simulations. 
				> k: inter-linguistic similarity. 
				> s: prestige of the language whose time series is stored in x_. 

		Returns: 
			<< F = [Fx, Fy]: differential steps for variables x and y of the model. 
				> This must be composed with dT to integrate numerically. 
				> Since x is a series of numpy arrays, F is evaluated for each of the values provided. 
				> Hence F has the same dimension as x[:,0]. 

	"""

	# Uncompress params and variables: 
	(a, c, k, s) = uncompressParams(params); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating differential field: 
	Fx = c*( s*(1-k)*np.multiply((1-x_), np.power(1-y_, a)) - (1-s)*np.multiply(x_, np.power(1-x_, a))); 
	Fy = c*( (1-s)*(1-k)*np.multiply((1-y_), np.power(1-x_, a)) - s*np.multiply(y_, np.power(1-y_, a))); 
	F = np.array([Fx, Fy]).transpose(); 
	return F; 

def dF_a(x, params, kwargs={}): 
	"""	dF_a function: 

			This function computes the derivative of the differential field with respect to the parameter a. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> a: exponent that weights the attracting population. 
				> c: scale factor that speeds up or slows down the simulations. 
				> k: inter-linguistic similarity. 
				> s: prestige of the language whose time series is stored in x_. 

		Returns: 
			<< dF_a_: numpy array with the evaluation of the derivative of the field w.r.t. parameter a. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(a, c, k, s) = uncompressParams(params); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating derivative of the differential field: 
	dF_a_x = c*( s*(1-k) * np.multiply( np.multiply((1-x_), np.power(1-y_, a)), np.log(1-y_)) 
		- (1-s)*np.multiply( np.multiply(x_, np.power(1-x_, a)), np.log(1-x_))); 
	dF_a_y = c*( (1-s)*(1-k)*np.multiply(np.multiply((1-y_), np.power(1-x_, a)), np.log(1-x_)) 
		- s*np.multiply(np.multiply(y_, np.power(1-y_, a)), np.log(1-y_))); 
	dF_a_ = np.array([dF_a_x, dF_a_y]).transpose(); 
	return dF_a_; 

def dF_c(x, params, kwargs={}): 
	"""	dF_c function: 

			This function computes the derivative of the differential field with respect to the parameter c. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> a: exponent that weights the attracting population. 
				> c: scale factor that speeds up or slows down the simulations. 
				> k: inter-linguistic similarity. 
				> s: prestige of the language whose time series is stored in x_. 

		Returns: 
			<< dF_c_: numpy array with the evaluation of the derivative of the field w.r.t. parameter c. 
				> This is evaluated for each of the values of x provided. 

	"""

	c = params["c"]; 
	dF_c_ = F(x, params)/c; 
	return dF_c_; 

def dF_k(x, params, kwargs={}): 
	"""	dF_k function: 

			This function computes the derivative of the differential field with respect to the parameter k. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> a: exponent that weights the attracting population. 
				> c: scale factor that speeds up or slows down the simulations. 
				> k: inter-linguistic similarity. 
				> s: prestige of the language whose time series is stored in x_. 

		Returns: 
			<< dF_k_: numpy array with the evaluation of the derivative of the field w.r.t. parameter k. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(a, c, k, s) = uncompressParams(params); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating derivative of the differential field: 
	dF_k_x = -c*s*np.multiply( 1-x_, np.power(1-y_, a) ); 
	dF_k_y = -c*(1-s)*np.multiply( 1-y_, np.power(1-x_, a) ); 
	dF_k_ = np.array([dF_k_x, dF_k_y]).transpose(); 
	return dF_k_; 

def dF_s(x, params, kwargs={}): 
	"""	dF_s function: 

			This function computes the derivative of the differential field with respect to the parameter s. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> a: exponent that weights the attracting population. 
				> c: scale factor that speeds up or slows down the simulations. 
				> k: inter-linguistic similarity. 
				> s: prestige of the language whose time series is stored in x_. 

		Returns: 
			<< dF_s_: numpy array with the evaluation of the derivative of the field w.r.t. parameter s. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompress params and variables: 
	(a, c, k, s) = uncompressParams(params); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating derivative of the differential field: 
	dF_s_x = c*( (1-k)*np.multiply(1-x_, np.power(1-y_, a)) + np.multiply(x_, np.power(1-x_, a)) ); 
	dF_s_y = -c*( (1-k)*np.multiply(1-y_, np.power(1-x_, a)) + np.multiply(y_, np.power(1-y_, a)) ); 
	dF_s_ = np.array([dF_s_x, dF_s_y]).transpose(); 
	return dF_s_; 

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
				> a: exponent that weights the attracting population. 
				> c: scale factor that speeds up or slows down the simulations. 
				> k: inter-linguistic similarity. 
				> s: prestige of the language whose time series is stored in x_. 

		Returns: 
			<< dF_params_: dictionary containing the derivative of the model flux w.r.t. all of the parameters. 

	"""

	dFDict = {}; 
	dFDict["a"] = dF_a; 
	dFDict["c"] = dF_c; 
	dFDict["k"] = dF_k; 
	dFDict["s"] = dF_s; 

	if ("toFitParams" in kwargs.keys()): 
		toFitParams = kwargs["toFitParams"]; 
	else: 
		toFitParams = dFDict.keys(); 

	dF_params_ = {}; 
	for key in toFitParams: 
		dF_params_[key] = dFDict[key](x, params, kwargs); 

	return dF_params_; 


