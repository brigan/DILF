"""

	model.py: 

		This file contains functions to operate the Minett-Wang model of language competition. 

"""

import numpy as np; 

def uncompressParams(params, kwargs={}): 
	"""	uncompressParams function: 

			This function uncompresses the parameters from a dictionary into a more handy form. 

		Inputs: 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> a: exponent that weights the attracting population. 
				> c_zx, c_xz, c_zy, c_yz: scale factors weighting transitions to and from the bilingual group. 
				> mu: mortality rate weighting the influence of horizontally- versus vertically-based language shift. 
				> s: prestige of the language whose time series is stored in x[:,0]. 

		Returns: 
			<< a: exponent that weights the attracting population. 
			<< c_zx, c_xz, c_zy, c_yz: scale factors weighting transitions to and from the bilingual group. 
			<< mu: mortality rate weighting the influence of horizontally- versus vertically-based language shift. 
			<< s: prestige of the language whose time series is stored in x[:,0]. 

	"""

	return (params["a"], params["c_zx"], params["c_xz"], params["c_zy"], params["c_yz"], params["mu"], params["s"]); 


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
				<< params: dictionary containing the parameters for this model: 
					< a: exponent that weights the attracting population. 
					< c_zx, c_xz, c_zy, c_yz: scale factors weighting transitions to and from the bilingual group. 
					< mu: mortality rate weighting the influence of horizontally- versus vertically-based language shift. 
					< s: prestige of the language whose time series is stored in x[:,0]. 

	"""

	params = {}; 
	params["a"] = np.random.uniform(1, 2); 
	params["c_zx"] = np.random.uniform(0.01, 2); 
	params["c_xz"] = np.random.uniform(0.01, 2); 
	params["c_zy"] = np.random.uniform(0.01, 2); 
	params["c_yz"] = np.random.uniform(0.01, 2); 
	params["mu"] = np.random.uniform(0, 1); 
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

			This function implements the differential step for the Minett-Wang model of language competition (see [1]).

            [1] Minett JW, Wang WS, Modelling endangered languages: The effects of bilingualism and
social structure. Lingua, 118(1), pp.19-45 (2008).

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> a: exponent that weights the attracting population. 
				> c_zx, c_xz, c_zy, c_yz: scale factors weighting transitions to and from the bilingual group. 
				> mu: mortality rate weighting the influence of horizontally- versus vertically-based language shift. 
				> s: prestige of the language whose time series is stored in x[:,0]. 

		Returns: 
			<< F = [Fx, Fy]: differential steps for variables x and y of the model. 
				> This must be composed with dT to integrate numerically. 
				> Since x is a series of numpy arrays, F is evaluated for each of the values provided. 
				> Hence F has the same dimension as x. 

	"""

	# Uncompressing parameters and variables: 
	(a, c_zx, c_xz, c_zy, c_yz, mu, s) = uncompressParams(params); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating differential field: 
	Fx = mu*c_zx*s*np.multiply(1-x_-y_, np.power(x_, a)) - (1-mu)*c_xz*(1-s)*np.multiply(x_, np.power(y_, a)); 
	Fy = mu*c_zy*(1-s)*np.multiply(1-x_-y_, np.power(y_, a)) - (1-mu)*c_yz*s*np.multiply(y_, np.power(x_, a)); 
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
				> c_zx, c_xz, c_zy, c_yz: scale factors weighting transitions to and from the bilingual group. 
				> mu: mortality rate weighting the influence of horizontally- versus vertically-based language shift. 
				> s: prestige of the language whose time series is stored in x[:,0]. 

		Returns: 
			<< dF_a_: numpy array with the evaluation of the derivative of the field w.r.t. parameter a. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompressing parameters and variables: 
	(a, c_zx, c_xz, c_zy, c_yz, mu, s) = uncompressParams(params); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating derivative of the differential field: 
	dF_a_x = (mu*c_zx*s*np.multiply(np.multiply(1-x_-y_, np.power(x_, a)), np.log(x_)) 
		- (1-mu)*c_xz*(1-s)*np.multiply(np.multiply(x_, np.power(y_, a)), np.log(y_)) ); 
	dF_a_y = (mu*c_zy*(1-s)*np.multiply(np.multiply(1-x_-y_, np.power(y_, a)), np.log(y_))
		- (1-mu)*c_yz*s*np.multiply(np.multiply(y_, np.power(x_,a)), np.log(x_)) ); 
	dF_a_ = np.array([dF_a_x, dF_a_y]).transpose(); 
	return dF_a_; 

def dF_c_zx(x, params, kwargs={}): 
	"""	dF_c_zx function: 

			This function computes the derivative of the differential field with respect to the parameter c_zx. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> a: exponent that weights the attracting population. 
				> c_zx, c_xz, c_zy, c_yz: scale factors weighting transitions to and from the bilingual group. 
				> mu: mortality rate weighting the influence of horizontally- versus vertically-based language shift. 
				> s: prestige of the language whose time series is stored in x[:,0]. 

		Returns: 
			<< dF_c_zx_: numpy array with the evaluation of the derivative of the field w.r.t. parameter c_zx. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompressing parameters and variables: 
	(a, c_zx, c_xz, c_zy, c_yz, mu, s) = uncompressParams(params); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating derivative of the differential field: 
	dF_c_zx_x = mu*s*np.multiply(1-x_-y_, np.power(x_, a)); 
	dF_c_zx_y = np.zeros(y_.shape); 
	dF_c_zx_ = np.array([dF_c_zx_x, dF_c_zx_y]).transpose(); 
	return dF_c_zx_; 

def dF_c_xz(x, params, kwargs={}): 
	"""	dF_c_xz function: 

			This function computes the derivative of the differential field with respect to the parameter c_xz. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> a: exponent that weights the attracting population. 
				> c_zx, c_xz, c_zy, c_yz: scale factors weighting transitions to and from the bilingual group. 
				> mu: mortality rate weighting the influence of horizontally- versus vertically-based language shift. 
				> s: prestige of the language whose time series is stored in x[:,0]. 

		Returns: 
			<< dF_c_zx_: numpy array with the evaluation of the derivative of the field w.r.t. parameter c_xz. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompressing parameters and variables: 
	(a, c_zx, c_xz, c_zy, c_yz, mu, s) = uncompressParams(params); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating derivative of the differential field: 
	dF_c_xz_x = -(1-mu)*(1-s)*np.multiply(x_, np.power(y_, a)); 
	dF_c_xz_y = np.zeros(y_.shape); 
	dF_c_xz_ = np.array([dF_c_xz_x, dF_c_xz_y]).transpose(); 
	return dF_c_xz_; 

def dF_c_zy(x, params, kwargs={}): 
	"""	dF_c_zy function: 

			This function computes the derivative of the differential field with respect to the parameter c_zy. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> a: exponent that weights the attracting population. 
				> c_zx, c_xz, c_zy, c_yz: scale factors weighting transitions to and from the bilingual group. 
				> mu: mortality rate weighting the influence of horizontally- versus vertically-based language shift. 
				> s: prestige of the language whose time series is stored in x[:,0]. 

		Returns: 
			<< dF_c_zy_: numpy array with the evaluation of the derivative of the field w.r.t. parameter c_zy. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompressing parameters and variables: 
	(a, c_zx, c_xz, c_zy, c_yz, mu, s) = uncompressParams(params); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating derivative of the differential field: 
	dF_c_zy_x = np.zeros(x_.shape); 
	dF_c_zy_y = mu*(1-s)*np.multiply(1-x_-y_, np.power(y_, a)); 
	dF_c_zy_ = np.array([dF_c_zy_x, dF_c_zy_y]).transpose(); 
	return dF_c_zy_; 

def dF_c_yz(x, params, kwargs={}): 
	"""	dF_c_yz function: 

			This function computes the derivative of the differential field with respect to the parameter c_yz. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> a: exponent that weights the attracting population. 
				> c_zx, c_xz, c_zy, c_yz: scale factors weighting transitions to and from the bilingual group. 
				> mu: mortality rate weighting the influence of horizontally- versus vertically-based language shift. 
				> s: prestige of the language whose time series is stored in x[:,0]. 

		Returns: 
			<< dF_c_zx_: numpy array with the evaluation of the derivative of the field w.r.t. parameter c_yz. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompressing parameters and variables: 
	(a, c_zx, c_xz, c_zy, c_yz, mu, s) = uncompressParams(params); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating derivative of the differential field: 
	dF_c_yz_x = np.zeros(x_.shape); 
	dF_c_yz_y = -(1-mu)*s*np.multiply(y_, np.power(x_, a)); 
	dF_c_yz_ = np.array([dF_c_yz_x, dF_c_yz_y]).transpose(); 
	return dF_c_yz_; 

def dF_mu(x, params, kwargs={}): 
	"""	dF_mu function: 

			This function computes the derivative of the differential field with respect to the parameter mu. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> These parameters are either provided or initialized somewhere else. 
				> a: exponent that weights the attracting population. 
				> c_zx, c_xz, c_zy, c_yz: scale factors weighting transitions to and from the bilingual group. 
				> mu: mortality rate weighting the influence of horizontally- versus vertically-based language shift. 
				> s: prestige of the language whose time series is stored in x[:,0]. 

		Returns: 
			<< dF_c_zx_: numpy array with the evaluation of the derivative of the field w.r.t. parameter mu. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompressing parameters and variables: 
	(a, c_zx, c_xz, c_zy, c_yz, mu, s) = uncompressParams(params); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating derivative of the differential field: 
	dF_mu_x = c_zx*s*np.multiply(1-x_-y_, np.power(x_, a)) + c_xz*(1-s)*np.multiply(x_, np.power(y_, a)); 
	dF_mu_y = c_zy*(1-s)*np.multiply(1-x_-y_, np.power(y_, a)) + c_yz*s*np.multiply(y_, np.power(x_, a)); 
	dF_mu_ = np.array([dF_mu_x, dF_mu_y]).transpose(); 
	return dF_mu_; 

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
				> c_zx, c_xz, c_zy, c_yz: scale factors weighting transitions to and from the bilingual group. 
				> mu: mortality rate weighting the influence of horizontally- versus vertically-based language shift. 
				> s: prestige of the language whose time series is stored in x[:,0]. 

		Returns: 
			<< dF_c_zx_: numpy array with the evaluation of the derivative of the field w.r.t. parameter s. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompressing parameters and variables: 
	(a, c_zx, c_xz, c_zy, c_yz, mu, s) = uncompressParams(params); 
	x_ = x[:,0]; 
	y_ = x[:,1]; 

	# Evaluating derivative of the differential field: 
	dF_s_x = mu*c_zx*np.multiply(1-x_-y_, np.power(x_, a)) + (1-mu)*c_xz*np.multiply(x_, np.power(y_, a)); 
	dF_s_y = -mu*c_zy*np.multiply(1-x_-y_, np.power(y_, a)) - (1-mu)*c_yz*np.multiply(y_, np.power(x_, a)); 
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
				> c_zx, c_xz, c_zy, c_yz: scale factors weighting transitions to and from the bilingual group. 
				> mu: mortality rate weighting the influence of horizontally- versus vertically-based language shift. 
				> s: prestige of the language whose time series is stored in x[:,0]. 

		Returns: 
			<< dF_params_: dictionary containing the derivative of the model flux w.r.t. all of the parameters. 

	"""

	dFDict = {}; 
	dFDict["a"] = dF_a; 
	dFDict["c_zx"] = dF_c_zx; 
	dFDict["c_xz"] = dF_c_xz; 
	dFDict["c_zy"] = dF_c_zy; 
	dFDict["c_yz"] = dF_c_yz; 
	dFDict["mu"] = dF_mu; 
	dFDict["s"] = dF_s; 

	if ("toFitParams" in kwargs.keys()): 
		toFitParams = kwargs["toFitParams"]; 
	else: 
		toFitParams = dFDict.keys(); 

	dF_params_ = {}; 
	for key in toFitParams: 
		dF_params_[key] = dFDict[key](x, params, kwargs); 

	return dF_params_; 






