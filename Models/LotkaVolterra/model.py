"""

	model.py: 

		This file contains functions to run the original Lotka-Volterra model. 

"""

import numpy as np; 
	
def uncompressParams(params, kwargs={}): 
	"""	uncompressParams function: 

			This function uncompresses the parameters of the normalized Lotka Volterra model from the dictionary form
			into arrays and matrices as they are more convenient to operate.

		Inputs: 
			>> params: Dictionary containing the parameters needed to run the Lotka-Volterra model. 
				> nVars: number of variables. 
				> r_i: nVar different entries accordingly labeled that represent each specie's unrestricted growth. 
				> k_i: load capacities. 
				> alpha_i_j: interaction terms. 

		Returns: 
			<< nVars: number of variables. 
			<< r: array containing the growth rates. 
			<< k: array containing the load capacities. 
			<< alpha: matrix containing the interaction terms. 
	
	"""

	nVars = params["nVars"]; 
	r = []; 
	k = []; 
	alpha = []; 
	for i in range(nVars): 
		r += [params["r_"+str(i)]]; 
		k += [params["k_"+str(i)]]; 
		alpha += [[]]; 
		for j in range(nVars): 
			alpha[i] += [params["alpha_"+str(i)+"_"+str(j)]]; 

	return (nVars, np.array(r), np.array(k), np.array(alpha)); 

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
					< r: array containing the growth rates. 
					< k: array containing the load capacities. 
					< alpha: matrix containing the interaction terms. 

	"""

	params = {}; 
	nVars = kwargs["nVars"]; 
	params["nVars"] = nVars; 
	for iVar in range(nVars): 
		params["r_"+str(iVar)] = np.random.uniform(0, 1); 
		params["k_"+str(iVar)] = np.random.uniform(100, 500); 
		params["alpha_"+str(iVar)+'_'+str(iVar)] = 1.; 
		for jVar in range(iVar+1, nVars): 
			params["alpha_"+str(iVar)+'_'+str(jVar)] = np.random.uniform(1, 2); 
			params["alpha_"+str(jVar)+'_'+str(iVar)] = np.random.uniform(1, 2); 

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
	"""	F funciton: 

			This function implements the differential step for the Lotka-Volterra model. (ACHTUNG!! Find some
			references.)

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> This is a nVar*nPoints numpy array. 
				> Note that the number of variables can change for this model! 
			>> params: dictionary containing the parameteres to run the normalized Lotka-Volterra model. 
				> r_i: intrinsic growth rates for each species. 
				> k_i: load capacities. 
				> alpha_i_j: interaction terms. 

		Returns: 
			<< F = [Fx_1, ..., Fx_nVar]: differential steps for the nVar species in the model. 
				> This must be composed with dT to integrate numerically. 
				> Since x is a series of numpy arrays, F is evaluated for each of the values provided. 
				> Hence F has the same dimension as x. 

	"""

	# Uncompress variables: 
	(nVars, r, k, alpha) = uncompressParams(params); 
	r_ = np.repeat([r], x.shape[0], axis=0);
	k_ = np.repeat([k], x.shape[0], axis=0);

	# Computing F for each of the variables: 
	F_ =  np.multiply(r_, np.multiply(x, 1 - np.divide(np.dot(alpha, x.transpose()).transpose(), k_))); 

	return F_; 

def dF_r(x, params, kwargs={}): 
	"""	dF_r function:
 
			This function computes the derivative of the differential field with respect to the parameter r. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> r_i: intrinsic growth rates for each species. 
				> k_i: load capacities. 
				> alpha_i_j: interaction terms. 

		Returns: 
			<< dF_r_: numpy array with the evaluation of the derivative of the field w.r.t. parameters in r. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompress parameters: 
	(nVars, r, k, alpha) = uncompressParams(params); 
	r_ = np.repeat([r], x.shape[0], axis=0);
	k_ = np.repeat([k], x.shape[0], axis=0);

	# Computing derivative -- it's just F divided by r: 
	dF_r_merged = np.multiply(x, 1 - np.divide(np.dot(alpha, x.transpose()).transpose(), k_)); 
	dF_r_ = {}; 
	for iVar in range(nVars): 
		dF_r_["r_"+str(iVar)] = np.zeros(x.shape); 
		dF_r_["r_"+str(iVar)][:, iVar] = dF_r_merged[:, iVar]; 

	return dF_r_; 

def dF_k(x, params, kwargs={}): 
	"""	dF_k function: 

			This function computes the derivative of the differential field with respect to the parameter k. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> r_i: intrinsic growth rates for each species. 
				> k_i: load capacities. 
				> alpha_i_j: interaction terms. 

		Returns: 
			<< dF_k_: numpy array with the evaluation of the derivative of the field w.r.t. parameters in k. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompress parameters: 
	(nVars, r, k, alpha) = uncompressParams(params); 
	r_ = np.repeat([r], x.shape[0], axis=0);
	k2_ = np.power(np.repeat([k], x.shape[0], axis=0), 2);

	dF_k_merged = np.multiply(r_, np.multiply(x, np.divide(np.dot(alpha, x.transpose()).transpose(), k2_))); 
	dF_k_ = {}; 
	for iVar in range(nVars): 
		dF_k_["k_"+str(iVar)] = np.zeros(x.shape); 
		dF_k_["k_"+str(iVar)][:, iVar] = dF_k_merged[:, iVar]; 

	return dF_k_; 

def dF_alpha(x, params, kwargs={}): 
	"""	dF_alpha function: 

			This function computes the derivative of the differential field with respect to the parameter alpha. 

		Inputs: 
			>> x: Values at which we wish to evaluate the model. 
				> The caller of this function must make sure to provide the adequate variables. 
				> This is a nVar*nPoints numpy array. 
			>> params: dictionary containing the parameters of the model. 
				> r_i: intrinsic growth rates for each species. 
				> k_i: load capacities. 
				> alpha_i_j: interaction terms. 

		Returns: 
			<< dF_k_: numpy array with the evaluation of the derivative of the field w.r.t. parameters in alpha. 
				> This is evaluated for each of the values of x provided. 

	"""

	# Uncompress parameters: 
	(nVars, r, k, alpha) = uncompressParams(params); 
	r_ = np.repeat([r], x.shape[0], axis=0);
	k_ = np.repeat([k], x.shape[0], axis=0);

	dF_alpha_ = {}; 
	for iVar in range(nVars): 
		dF_alpha_["alpha_"+str(iVar)+'_'+str(iVar)] = np.zeros(x.shape); 
		for jVar in range(iVar+1, nVars): 
			dF_alpha_["alpha_"+str(iVar)+'_'+str(jVar)] = np.zeros(x.shape); 
			dF_alpha_["alpha_"+str(iVar)+'_'+str(jVar)][:, iVar] = -np.divide(np.multiply(r_[:, iVar], np.multiply(x[:, iVar], x[:, jVar])), k_[:, iVar]); 

			dF_alpha_["alpha_"+str(jVar)+'_'+str(iVar)] = np.zeros(x.shape); 
			dF_alpha_["alpha_"+str(jVar)+'_'+str(iVar)][:, jVar] = -np.divide(np.multiply(r_[:, jVar], np.multiply(x[:, jVar], x[:, iVar])), k_[:, jVar]); 

	return dF_alpha_; 


def dF_params(x, params, kwargs={}): 
	"""	dF_params function: 

			This function computes the derivative of the flux w.r.t. all of the parameters, which are returned in a
			dictionary.

			ACHTUNG!! 
				Usually, it should be possible to implement fits with fixed parameters using this function. However,
				this model is singular because the number of parameters depends on the number of variables. This has not
				been accounted for yet.

			Inputs: 
				>> x: Values at which we wish to evaluate the model. 
					> The caller of this function must make sure to provide the adequate variables. 
					> This is a nVar*nPoints numpy array. 
				>> params: dictionary containing the parameters of the model. 
					> r_i: intrinsic growth rates for each species. 
					> k_i: load capacities. 
					> alpha_i_j: normalized interaction terms. 

			Returns: 
				<< dF_params_: dictionary containing the derivative of the model flux w.r.t. all of the parameters. 

	"""

	dF_params_ = dF_r(x, params); 
	dF_params_.update(dF_k(x, params)); 
	dF_params_.update(dF_alpha(x, params)); 

	return dF_params_; 





