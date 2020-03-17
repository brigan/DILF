"""

	helper.py: 

		This file contains all the functions and subroutines to implement MInLF. 

"""

import numpy as np; 
import matplotlib.pyplot as plt; 
from mpl_toolkits.mplot3d import Axes3D; 
import os, sys, imp; 
from copy import copy; 

def loadData(dataWholeName): 
	"""	loadData function: 

			This function loads data for this specific library. 

		Inputs: 
			>> dataWholeName: Single file from which the data is loaded. 

		Returns: 
			<< t, x: time and values of the variables as sampled. 
			<< nVar: number of variables. 
			<< nPoints: number of points along the time trajectories. 

	"""

	dataIn = np.loadtxt(dataWholeName, delimiter=','); 
	t = dataIn[:,0]; 
	x = dataIn[:,1:]; 
	(nPoints, nVars) = x.shape; 

	return (t, x, nPoints, nVars); 


def crossValPolyFitSingleSeries(t, thisX, nPoints, k, maxDeg): 
	"""	crossValFitSingleSeries function: 

			This function takes a single time series and performs k-fold x-validation to find out the polynomial that
			better fits the data without overfitting. Note that there will be at least as many calls to this function as
			variables in the problem.

		Inputs: 
			>> t, thisX: variables to perform the fit. 
			>> nPoints: number of points in the time series. 
			>> k: number of folds in the x-validation. 
			>> maxDeg: maximum degree of the polynomial. 

		Returns: 
			<< pDeg: optimal degree found after x-validation. 
			<< p: polynomial fit with optimal degree found. 

	""" 

	# Generate indexes for cross validation: 
	nIOut = int(nPoints/k); 
	iPerm = list(np.random.permutation(nPoints)); 

	# Loop over degrees: 
	vErrorList = []; # List to store validation error associated to each degree. 
	for pDeg in range(1, maxDeg+1): 
		vError = 0; # Initializing validation error to 0. 
		# Loop over folds: 
		for iFold in range(k): 
			# Picking up points for fitting and testing x-validation: 
			iOut = iPerm[iFold*nIOut:(iFold+1)*nIOut]; 
			iIn = iPerm[0:iFold*nIOut] + iPerm[(iFold+1)*nIOut:]; 
			tFit = t[iIn]; tTest = t[iOut]; 
			xFit = thisX[iIn]; xTest = thisX[iOut]; 

			# Fitting to polynomial: 
			p = np.polyfit(tFit, xFit, pDeg); 
			p_ = np.poly1d(p); 
			# Computing validation error (evaluate polynomial and compare to points left out):
			vError += sum(np.power(xTest - p_(tTest), 2)); 

		# Add to validation error list. 
		vErrorList += [vError]; 

	# Optimal degree has lowest validation error: 
	pDeg = np.argmin(vErrorList) + 1; 
	p = np.polyfit(t, thisX, pDeg)

	return pDeg, p; 


def crossValPolyFit(t, x, nVars, nPoints, k, maxDeg, wPath): 
	"""	crossValPolyFit function: 

			This function computes the optimal polynomial fit for all variables. It further stores these polynomials in
			an output folder where they can be later retrieved. It would be ideal to have an option to plot the results
			of the evaluations of the polynomials (can we outsource this)? 

		Inputs. 
			>> t, x: time and variables for the series. 
			>> nVars: number of variables. 
			>> nPoints: number of points in the time series. 
			>> k: number of folds in the x-validation. 
			>> maxDeg: maximum degree of the polynomial. 
			>> wPath: where the results will be stored. 

	"""

	# Loop over variables: 
	for iVar in range(nVars): 
		thisX = x[:,iVar]; 

		# Maximum degree of polynomial fit is just for guidance. It might change dynamically, so we build a buffer variable
		# to store this value locally. For example, if optimal degree comes out with maximal degree, we want to explore
		# pDeg+1, so we will call the fitter again. Therefore we build this while() and allow to increase thisMaxDeg inside
		# the loop (but individually for each time series). 
		thisMaxDeg = maxDeg; 
		fGo = True; 
		while (fGo): 
			fGo = False; 
			(pDeg, p) = crossValPolyFitSingleSeries(t, thisX, nPoints, k, thisMaxDeg); 
			if (pDeg == thisMaxDeg): 
				thisMaxDeg += 1; 
				fGo = True; 

		# Saving to file for this variable: 
		fOutName = os.path.join(wPath, "p_v"+str(iVar)+".csv"); 
		np.savetxt(fOutName, p, delimiter=','); 

	return; 

def loadPolynomial(iVar, wPath): 
	"""	loadPolynomial function: 

			This function loads a single polynomial associated to a single variable. 

		Inputs: 
			>> iVar: index of the variable to be retrieved (starting at 0). 
			>> wPath: directory where the polynomial is stored. 

		Returns: 
			<< p: exponents associated to the polynomial associated to this variable. 

	"""

	# Reading from file: 
	fInName = os.path.join(wPath, "p_v"+str(iVar)+".csv"); 
	p = np.loadtxt(fInName, delimiter=','); 

	return p; 

def loadPolynomials(nVars, wPath): 
	""" loadPolynomials function: 

			This function loads and returns the polynomials associated to a series of variables. 

		Inputs: 
			>> nVars: number of variables to be loaded. 
			>> wPath: directory where the polynomials are stored. 

		Returns: 
			<< pList: list containing the polynomials associated to each of the variables. 

	"""

	pList = []; 
	for iVar in range(nVars): 
		pList += [loadPolynomial(iVar, wPath)]; 

	return pList; 

def plotPolynomialsWithData(t, x, nVars, wPath): 
	"""plotPolynomialsWithData function: 

			This function loads the polynomials produced after fitting and plots them along with the original data. 

		Inputs: 
			>> t, x: time and variables to be plotted along the polynomials. 
			>> nVars: number of variables. 
			>> wPath: working directory where the polynomials are stored. 

	"""

	pList = loadPolynomials(nVars, wPath); 
	p_List = [np.poly1d(pp) for pp in pList]; 

	fig = plt.figure(); 
	plt.plot(t, x, 'x'); 
	for pp in p_List: 
		plt.plot(t, pp(t)); 

	figName = os.path.join(wPath, "polyFit.eps"); 
	fig.savefig(figName); 

	plt.show(); 

	return; 

def computePSelect(chiList): 
	"""	computePSelect funciton: 

			This function computes the likelihood that a solution will be selected based on its chi2 performance. 

		Inputs: 
			>> chiList: list with the performances upon which selection probability will be based. 

		Returns: 
			<< pSelect: probability that each of the solutions will be selected based on the chi2 performance. 

	"""

	chiMax = chiList[-1]; 
	chiMax += 0.01*chiMax; # Avoid problems if population converges! 
	pSelect = [(chiMax-cc) for cc in chiList]; 
	Z = sum(pSelect); 
	pSelect = [pp/Z for pp in pSelect]; 


	return pSelect; 

def crossover(paramsList): 
	"""	crossover function: 

			This function implements a plain crossover to double the population provided. 

		Inputs: 
			>> paramsList: list with the input population. 

		Returns: 
			<< paramsList: list with the output population. 
				< It has doubled size w.r.t. input paramsList. 
				< Second half is made up of crossed-over elements. 

	"""

	# Prepare keys for swapping: 
	keys = paramsList[0].keys(); 
	nKeys = len(keys); 
	# Prepare list to host new parameter sets: 
	paramsList_ = []; 
	# Loop over 1/2 of current population size (1/4 of actual population size): 
	for iCross in range(len(paramsList)/2): 
		# Select elements to cross over: 
		params1 = np.random.choice(paramsList); 
		params2 = np.random.choice(paramsList); 
		# Select keys to swap: 
		keys1 = np.random.choice(keys, size=nKeys/2); 
		params3 = {}; 
		params4 = {}
		# Allocate each to corresponding new list of parameters: 
		for key in keys: 
			if key in keys1: 
				params3[key] = params1[key]; 
				params4[key] = params2[key]; 
			else: 
				params3[key] = params2[key]; 
				params4[key] = params1[key]; 
		paramsList_ += [copy(params3), copy(params4)]; 

	return paramsList + paramsList_; 

def mutate(paramsList, nProtect, pMutate, protectedParams=[]): 
	"""	mutate function: 

			This function randomly implements a mutation on the existing parameters. Each parameter mutates with
			probability pMutate, for which a binomial random variable is drawn. 

			As of right now, mutation introduces a Gaussian with average whatever the value of the variable and std 20%
			of the variable value. 

			ACHTUNG!! Ideally, another version of this function will be implemented that implements guided mutation as a
			small gradient descent step. 

		Inputs: 
			>> paramsList: list of parameters to be mutated. 
			>> nProtect: number of elite solutions protected from mutation. 
			>> pMutate: probability that each parameter mutates. 
			>> protectedParams=[]: list of parameters protected from mutation. 

		Returns: 
			<< paramsList: list of parameters after mutation. 

	"""

	# Extracting parameters that can be mutated: 
	keys = [key for key in paramsList[0].keys() if key not in protectedParams]; 
	nKeys = len(keys); 
	nCanMutate = len(paramsList)-nProtect; 

	# Computing number of mutations: 
	nMutations = np.random.binomial(nKeys*nCanMutate, pMutate); 
	# Indexes and parameters to be mutated: 
	iMutations = np.random.choice(range(nProtect, len(paramsList)), size=nMutations); 
	keyMutations = np.random.choice(keys, size=nMutations); 
	for (iMutation, keyMutation) in zip(iMutations, keyMutations): 
		# Implement mutation: 
		paramsList[iMutation][keyMutation] = (paramsList[iMutation][keyMutation] 
						+ np.random.normal(paramsList[iMutation][keyMutation], 0.2*abs(paramsList[iMutation][keyMutation]))); 

	return paramsList; 

def genEvolveModel(t, nVars, wD, modelPath, modelName, genEvolParams, initModelArgs): 
	"""	genEvolveModel function: 

			This function applies a genetic algorithm to attempt to evolve good sets of parameters for a model given
			some data. 

		Inputs: 
			>> t, x: time and variables. 
				> t: usually dictates where the polynomials are evaluated. This might change in the future. 
			>> nVars: number of variables involved. 
				> ACH!: Attempting to make this as universal as possible, only involved variables are provided. 
			>> wD: working directory, where the polynomials are stored. 
			>> mPath: where models can be found. 
			>> mName: name of the model. 
			>> genEvolParams: dictionary containing several variables related to the parameters to run the fit. 
				> popSize: size of parameters sets. 
				> nProtect: number of elite parameters that are protected from mutation. 
				> itMax: maximum number of generations. 
				> pMutate: likelihood that a parameter is mutated. 
				> protectedParams=[]: list of parameters protected from mutation. 
			>> initModelArgs: dictionary with parameters that the model might need to get initialized. 

		Returns: 
			<< paramsPopulation: list of dictionaries containing the population after applying the genetic algorithm. 
			<< chi_d2List: list with the chi_d2 values for each set of parameters. 

	"""


	# Load polynomial fits, compute derivatives, evaluate where desired: 
	pList = loadPolynomials(nVars, wD); 
	p_List = [np.poly1d(pp) for pp in pList]; 
	dP_List = [np.polyder(pp_) for pp_ in p_List]; 
	x = np.array([pp(t) for pp in p_List]).transpose(); 
	dX = np.array([pp(t) for pp in dP_List]).transpose(); 

	# Loading the model: 
	mPath = os.path.join(modelPath, modelName); 
	modelFound = imp.find_module("model", [mPath]); 
	modelModuleName = imp.load_module("modelModuleName", *modelFound); 
	import modelModuleName as m; 

	# Unpacking parameters for the genetic algorithm: 
	popSize = genEvolParams["popSize"]; 
	nProtect = genEvolParams["nProtect"]; 
	itMax = genEvolParams["itMax"]; 
	nSelect = popSize/2 - nProtect; 
	pMutate = genEvolParams["pMutate"]; 
	protectedParams = genEvolParams["protectedParams"]; 

	# Initialize population: 
	paramsPopulation = [m.initializeParams(initModelArgs) for ii in range(popSize)]; 

	# Loop over itMax generations: 
	for it in range(itMax): 

		# Evaluate models: 
		iSolStart = 0; 
		chi_d2List_ = []; 
		if (it): 
			iSolStart = nProtect; 
			chi_d2List_ += chi_d2List[:nProtect]; 
		for iSol in range(iSolStart, popSize): 
			F_ = m.F(x, paramsPopulation[iSol]); 
			res = dX-F_; 
			chi_d2List_ += [sum(sum(np.power(res, 2)))/2]; 
		
		# Sort according to chi_d2, compute pSelect biased by this, repopulate non-protected population: 
		(chi_d2List_, paramsPopulation) = (list(t) for t in zip(*sorted(zip(chi_d2List_, paramsPopulation)))); 
		pSelect = computePSelect(chi_d2List_); 
		paramsRePopulation = list(np.random.choice(paramsPopulation, size=nSelect, p=pSelect)); 
		paramsPopulation = paramsPopulation[:nProtect] + paramsRePopulation; 

		# Crossover and mutation: 
		paramsPopulation = crossover(paramsPopulation); 
		paramsPopulation = mutate(paramsPopulation, nProtect, pMutate, protectedParams); 

		# Dummy needed to save elite: 
		chi_d2List = chi_d2List_; 
		
	return (paramsPopulation, chi_d2List); 


def fitModel(t, nVars, wD, modelPath, modelName, fitParams, initModelArgs={}, fitModelArgs={}): 
	"""	fitModel function: 

			This function implements the whole fit procedure to a given model. 

			ACHTUNG!! 

				This function uses the evaluation of the polynomials at t as the values taken by the variables of the
				model -- i.e. the x variable. This is convenient, but not necessarily good. Some models might demand
				*numerically* that physically plausible values are used for x. For example, some models might require
				that some of the variables are always positive. Polynomials fitted to the data might show anomalous
				behavior (e.g. negative values). This might not harm some models, but others might include a function
				that causes numerical problem if unrealistic values are used. For example, some models require the
				evaluation of log(x), which will prompt problems if any(x<0). The function fitModelWDataAsX() bypasses
				this problem by using the original data as x values, which is also a reasonable approach.

		Inputs: 
			>> t, x: time and variables. 
				> t: usually dictates where the polynomials are evaluated. This might change in the future. 
			>> nVars: number of variables involved. 
				> ACH!: Attempting to make this as universal as possible, only involved variables are provided. 
			>> wD: working directory, where the polynomials are stored. 
			>> mPath: where models can be found. 
			>> mName: name of the model. 
			>> fitParams: dictionary containing several variables related to the parameters to run the fit. 
				> eta: scale of the gradient step. 
				> haltCond: stop if improvement in chi_d2 is smaller than this. 
				> itMax: maximum iteration of the fit (even if haltCond is not achieved). 
			>> initModelArgs: dictionary with parameters that the model might need to get initialized. 
			>> fitModelArgs: dictionary with parameters and specifications for the fitting of the model. 

		Returns: 
			<< params: dictionary containing the best parameters resulting from the fit. 
			<< chi_d2: value of the loss function for the best parameters. 
			<< whyStop: reason why the process stopped: 
				< "haltCond": improvement of loss function is less than a fitParams["haltCond"]. 
				< "itMax": maximum number of iterations has been reached. 

	"""

	# Load polynomial fits, compute derivatives, evaluate where desired: 
	pList = loadPolynomials(nVars, wD); 
	p_List = [np.poly1d(pp) for pp in pList]; 
	dP_List = [np.polyder(pp_) for pp_ in p_List]; 
	x = np.array([pp(t) for pp in p_List]).transpose(); 
	dX = np.array([pp(t) for pp in dP_List]).transpose(); 

	# Loading the model: 
	mPath = os.path.join(modelPath, modelName); 
	modelFound = imp.find_module("model", [mPath]); 
	modelModuleName = imp.load_module("modelModuleName", *modelFound); 
	import modelModuleName as m; 

	# Reading fit meta-parameters: 
	eta = fitParams["eta"]; 
	haltCond = fitParams["haltCond"]; 
	itMax = fitParams["itMax"]; 

	# Initialize model parameters: 
	params = m.initializeParams(initModelArgs); 

	# Gradient descent loop: 
	fGo = True; 
	it = 0; 
	chi_d2 = 10E20; 
	whyStop = "None"; 
	while (fGo): 
		it += 1; 

		# Compute field, field derivatives w.r.t. each parameter, current chi_d2, and change in chi_d2: 
		F_ = m.F(x, params, fitModelArgs); 
		res = dX-F_; 
		chi_d2_ = sum(sum(np.power(res, 2)))/2; 
		dChi_d2 = abs(chi_d2_ - chi_d2); 
		chi_d2 = chi_d2_; 
		dF_params_ = m.dF_params(x, params, fitModelArgs); 

		# Update variables: 
		for pp in dF_params_.keys(): 
			params[pp] += eta*sum(sum( np.multiply( res, dF_params_[pp]) )); 

		# Checking for halting conditions: 
		if (it == itMax or dChi_d2<haltCond): 
			fGo = False; 
			if (it == itMax): 
				whyStop = "Iterations"; 
			else: 
				whyStop = "haltCond"; 

	# Saving outcome of fit: 
	saveFitModelOutput(params, chi_d2, whyStop, wD); 

	return (params, chi_d2, whyStop); 


def fitModelWDataAsX(t, x, nVars, wD, modelPath, modelName, fitParams, initModelArgs={}, fitModelArgs={}): 
	"""	fitModelWDataAsX function: 

			This function implements the whole fit procedure to a given model. 


			ACHTUNG!! 

				The function fitModel() above uses the evaluation of the polynomials at t as the values taken by the
				variables of the model -- i.e. the x variable. This is convenient, but not necessarily good. Some models
				might demand *numerically* that physically plausible values are used for x. For example, some models
				might require that some of the variables are always positive. Polynomials fitted to the data might show
				anomalous behavior (e.g. negative values). This might not harm some models, but others might include a
				function that causes numerical problem if unrealistic values are used. For example, some models require
				the evaluation of log(x), which will prompt problems if any(x<0). This function bypasses this problem by
				using the original data as x values, which is also a reasonable approach.

		Inputs: 
			>> t, x: time and variables. 
				> t: usually dictates where the polynomials are evaluated. This might change in the future. 
			>> nVars: number of variables involved. 
				> ACH!: Attempting to make this as universal as possible, only involved variables are provided. 
			>> wD: working directory, where the polynomials are stored. 
			>> mPath: where models can be found. 
			>> mName: name of the model. 
			>> fitParams: dictionary containing several variables related to the parameters to run the fit. 
				> eta: scale of the gradient step. 
				> haltCond: stop if improvement in chi_d2 is smaller than this. 
				> itMax: maximum iteration of the fit (even if haltCond is not achieved). 
			>> initModelArgs: dictionary with parameters that the model might need to get initialized. 
			>> fitModelArgs: dictionary with parameters and specifications for the fitting of the model. 

		Returns: 
			<< params: dictionary containing the best parameters resulting from the fit. 
			<< chi_d2: value of the loss function for the best parameters. 
			<< whyStop: reason why the process stopped: 
				< "haltCond": improvement of loss function is less than a fitParams["haltCond"]. 
				< "itMax": maximum number of iterations has been reached. 

	"""

	# Load polynomial fits, compute derivatives, evaluate where desired: 
	pList = loadPolynomials(nVars, wD); 
	p_List = [np.poly1d(pp) for pp in pList]; 
	dP_List = [np.polyder(pp_) for pp_ in p_List]; 
	dX = np.array([pp(t) for pp in dP_List]).transpose(); 

	# Loading the model: 
	mPath = os.path.join(modelPath, modelName); 
	modelFound = imp.find_module("model", [mPath]); 
	modelModuleName = imp.load_module("modelModuleName", *modelFound); 
	import modelModuleName as m; 

	# Reading fit meta-parameters: 
	eta = fitParams["eta"]; 
	haltCond = fitParams["haltCond"]; 
	itMax = fitParams["itMax"]; 

	# Initialize model parameters: 
	params = m.initializeParams(initModelArgs); 

	# Gradient descent loop: 
	fGo = True; 
	it = 0; 
	chi_d2 = 10E20; 
	whyStop = "None"; 
	while (fGo): 
		it += 1; 

		# Compute field, field derivatives w.r.t. each parameter, current chi_d2, and change in chi_d2: 
		F_ = m.F(x, params, fitModelArgs); 
		res = dX-F_; 
		chi_d2_ = sum(sum(np.power(res, 2)))/2; 
		dChi_d2 = abs(chi_d2_ - chi_d2); 
		chi_d2 = chi_d2_; 
		dF_params_ = m.dF_params(x, params, fitModelArgs); 

		# Update variables: 
		for pp in dF_params_.keys(): 
			params[pp] += eta*sum(sum( np.multiply( res, dF_params_[pp]) )); 

		# Checking for halting conditions: 
		if (it == itMax or dChi_d2<haltCond): 
			fGo = False; 
			if (it == itMax): 
				whyStop = "Iterations"; 
			else: 
				whyStop = "haltCond"; 

	# Saving outcome of fit: 
	saveFitModelOutput(params, chi_d2, whyStop, wD); 

	return (params, chi_d2, whyStop); 


def fitModelManyTimes(t, nVars, wD, modelPath, modelName, fitParams, initModelArgs={}, fitModelArgs={}): 
	"""	fitModelManyTimes function: 

			This function implements many times the whole fit procedure to a given model. 

			ACHTUNG!! 

				Just like the single fit function above, this function uses the evaluation of the polynomials at t as
				the values of x. To circumvent the problems that this might cause because of unrealistic values of x,
				use function fitModelManyTimesWDataAsX() below. 

		Inputs: 
			>> t, x: time and variables. 
				> t: usually dictates where the polynomials are evaluated. This might change in the future. 
			>> nVars: number of variables involved. 
				> ACH! This # might differ from the # of variables used by the model (e.g. if normalization to 1). 
				> ACH! Each model has to take care of these details on its own (e.g. MP uses first and last var). 
			>> wD: working directory, where the polynomials are stored. 
			>> mPath: where models can be found. 
			>> mName: name of the model. 
			>> fitParams: dictionary containing several variables related to the parameters to run the fit. 
				> eta: scale of the gradient step. 
				> haltCond: stop if improvement in chi_d2 is smaller than this. 
				> itMax: maximum iteration of the fit (even if haltCond is not achieved). 
				> nFit: number of times that the model is fitted. 
			>> initModelArgs: dictionary with parameters that the model might need to get initialized. 
			>> fitModelArgs: dictionary with parameters and specifications for the fitting of the model. 

		Returns: 
			<< paramsList: list containing the dictionaries with the best parameters resulting from each fit. 
			<< chi_d2List: list containing the values of the loss function for the best parameters. 
			<< whyStopList: list containing the reasons why each process stopped: 
				< "haltCond": improvement of loss function is less than a fitParams["haltCond"]. 
				< "itMax": maximum number of iterations has been reached. 
	"""

	# Load polynomial fits, compute derivatives. 
	# In this version, we evaluate within the fit loop so that x, dX are properly initialized. 
	pList = loadPolynomials(nVars, wD); 
	p_List = [np.poly1d(pp) for pp in pList]; 
	dP_List = [np.polyder(pp_) for pp_ in p_List]; 

	# Loading the model: 
	mPath = os.path.join(modelPath, modelName); 
	modelFound = imp.find_module("model", [mPath]); 
	modelModuleName = imp.load_module("modelModuleName", *modelFound); 
	import modelModuleName as m; 

	# Reading fit meta-parameters: 
	eta = fitParams["eta"]; 
	haltCond = fitParams["haltCond"]; 
	itMax = fitParams["itMax"]; 
	nFit = fitParams["nFit"]; 

	# Loop over number of fits: 
	paramsList = []; 
	chi_d2List = []; 
	whyStopList = []; 
	for iFit in range(nFit): 

		# Initialize x, dX: 
		x = np.array([pp(t) for pp in p_List]).transpose(); 
		dX = np.array([pp(t) for pp in dP_List]).transpose(); 

		# Initialize model parameters: 
		params = m.initializeParams(initModelArgs); 

		# Gradient descent loop: 
		fGo = True; 
		it = 0; 
		chi_d2 = 10E20; 
		whyStop = "None"; 
		while (fGo): 
			it += 1; 

			# Compute field, field derivatives w.r.t. each parameter, current chi_d2, and change in chi_d2: 
			F_ = m.F(x, params, fitModelArgs); 
			res = dX-F_; 
			chi_d2_ = sum(sum(np.power(res, 2)))/2; 
			dChi_d2 = abs(chi_d2_ - chi_d2); 
			chi_d2 = chi_d2_; 
			dF_params_ = m.dF_params(x, params, fitModelArgs); 

			# Update variables: 
			for pp in dF_params_.keys(): 
				params[pp] += eta*sum(sum( np.multiply( res, dF_params_[pp]) )); 

			# Checking for halting conditions: 
			if (it == itMax or dChi_d2<haltCond): 
				fGo = False; 
				if (it == itMax): 
					whyStop = "Iterations"; 
				else: 
					whyStop = "haltCond"; 

		# Saving outcome of fit: 
		saveFitModelOutput(params, chi_d2, whyStop, wD, "fit_"+modelName+'_'+str(iFit)+".txt"); 
		paramsList += [params]; 
		chi_d2List += [chi_d2]; 
		whyStopList += [whyStop]; 

	return (paramsList, chi_d2List, whyStopList); 


def fitModelManyTimesWDataAsX(t, xSafe, nVars, wD, modelPath, modelName, fitParams, initModelArgs={}, fitModelArgs={}): 
	"""	fitModelManyTimesWDataAsX function: 

			This function implements many times the whole fit procedure to a given model. 

			ACHTUNG!! 

				Just like the single fit function above, this function uses the original data as values of the variable
				x. This solves the problem of unrealistic evaluations of x. 

		Inputs: 
			>> t, x: time and variables. 
				> t: usually dictates where the polynomials are evaluated. This might change in the future. 
			>> xSafe: safe copy of the original data, which must be loaded several times without being modified. 
			>> nVars: number of variables involved. 
				> ACH! This # might differ from the # of variables used by the model (e.g. if normalization to 1). 
				> ACH! Each model has to take care of these details on its own (e.g. MP uses first and last var). 
			>> wD: working directory, where the polynomials are stored. 
			>> mPath: where models can be found. 
			>> mName: name of the model. 
			>> fitParams: dictionary containing several variables related to the parameters to run the fit. 
				> eta: scale of the gradient step. 
				> haltCond: stop if improvement in chi_d2 is smaller than this. 
				> itMax: maximum iteration of the fit (even if haltCond is not achieved). 
				> nFit: number of times that the model is fitted. 
			>> initModelArgs: dictionary with parameters that the model might need to get initialized. 

		Returns: 
			<< paramsList: list containing the dictionaries with the best parameters resulting from each fit. 
			<< chi_d2List: list containing the values of the loss function for the best parameters. 
			<< whyStopList: list containing the reasons why each process stopped: 
				< "haltCond": improvement of loss function is less than a fitParams["haltCond"]. 
				< "itMax": maximum number of iterations has been reached. 
	"""

	# Load polynomial fits, compute derivatives. 
	# In this version, we evaluate within the fit loop so that x, dX are properly initialized. 
	pList = loadPolynomials(nVars, wD); 
	p_List = [np.poly1d(pp) for pp in pList]; 
	dP_List = [np.polyder(pp_) for pp_ in p_List]; 

	# Loading the model: 
	mPath = os.path.join(modelPath, modelName); 
	modelFound = imp.find_module("model", [mPath]); 
	modelModuleName = imp.load_module("modelModuleName", *modelFound); 
	import modelModuleName as m; 

	# Reading fit meta-parameters: 
	eta = fitParams["eta"]; 
	haltCond = fitParams["haltCond"]; 
	itMax = fitParams["itMax"]; 
	nFit = fitParams["nFit"]; 

	# Loop over number of fits: 
	paramsList = []; 
	chi_d2List = []; 
	whyStopList = []; 
	for iFit in range(nFit): 

		# Initialize x, dX: 
		x = copy(xSafe); 
		dX = np.array([pp(t) for pp in dP_List]).transpose(); 

		# Initialize model parameters: 
		params = m.initializeParams(initModelArgs); 

		# Gradient descent loop: 
		fGo = True; 
		it = 0; 
		chi_d2 = 10E20; 
		whyStop = "None"; 
		while (fGo): 
			it += 1; 

			# Compute field, field derivatives w.r.t. each parameter, current chi_d2, and change in chi_d2: 
			F_ = m.F(x, params, fitModelArgs); 
			res = dX-F_; 
			chi_d2_ = sum(sum(np.power(res, 2)))/2; 
			dChi_d2 = abs(chi_d2_ - chi_d2); 
			chi_d2 = chi_d2_; 
			dF_params_ = m.dF_params(x, params, fitModelArgs); 

			# Update variables: 
			for pp in dF_params_.keys(): 
				params[pp] += eta*sum(sum( np.multiply( res, dF_params_[pp]) )); 

			# Checking for halting conditions: 
			if (it == itMax or dChi_d2<haltCond): 
				fGo = False; 
				if (it == itMax): 
					whyStop = "Iterations"; 
				else: 
					whyStop = "haltCond"; 

		# Saving outcome of fit: 
		saveFitModelOutput(params, chi_d2, whyStop, wD, "fit_"+modelName+'_'+str(iFit)+".txt"); 
		paramsList += [params]; 
		chi_d2List += [chi_d2]; 
		whyStopList += [whyStop]; 

	return (paramsList, chi_d2List, whyStopList); 


def saveFitModelOutput(params, chi_d2, whyStop, wD, fitName="fit.txt"): 
	"""	saveFitModelOutput function: 

			This function saves the output of a fit to a model in a format that can be later retrieved. Because the
			number of parameters changes from one model to the next, this function will just sort the parameters in
			params by alphabetical order. Each model must provide a specific function later that takes care of how to
			associate these to the corresponding parameters. 

		Inputs: 
			>> params: result of the fit. 
			>> chi_d2: error at the flux level for these parameters. 
			>> whyStop: stop condition reached for this run. 

	"""

	fOut = open(os.path.join(wD, fitName), 'w'); 
	for key in params.keys(): 
		fOut.write(key + ':' + str(params[key])+'\n'); 
	fOut.write(str(chi_d2)+'\n'); 
	fOut.write(whyStop); 
	fOut.close(); 

	return; 


def writeFitModelOutput(params, chi_d2, whyStop, wD, fitName="fit.txt"): 
	saveFitModelOutput(params, chi_d2, whyStop, wD, fitName); 
	return; 


def readFitModelOutput(wD, fitName): 
	"""	readFitModelOutput function: 

			This function loads the outcome of a former fit.  

		Inputs: 
			>> wD: where the results have been stored. 
			>> fitName: of the actual fit that we wish to read. 

		Returns: 
			<< params: dictionary with the parameters of the model. 
			<< chi_d2: value of the error function. 
			<< whyStop: reason why the fit stopped. 

	"""

	fIn = open(os.path.join(wD, fitName)); 
	dR = fIn.read(); 
	fIn.close(); 
	dL = dR.split('\n'); 
	params = {}; 
	fParams = True; 
	for ll in dL: 
		if ':' in ll: 
			newParam = ll.split(':'); 
			params[newParam[0]] = float(newParam[1]); 
		else: 
			try: 
				chi_d2 = float(ll); 
			except ValueError: 
				whyStop = ll; 

	return (params, chi_d2, whyStop); 


def loadFitModelOutput(wD, fitName):
	(params, chi_d2, whyStop) = readFitModelOutput(wD, fitName); 
	return (params, chi_d2, whyStop); 


def integrateModel(t, x, integrationParams, params, modelPath, modelName, kwArgs={}): 
	"""	integrateModel function: 

			This function integrates the model provided. It is possible to provide an intermediate point in time as
			initial condition and integrate backwards until a minimum time and then forwards until a maximum time step.

		Inputs: 
			>> t, x: time step at which the integration starts. 
			>> integrationParams: specifications for the integration. 
				> tMin: until which to integrate backwards. 
				> tMax: until which to integrate forwards. 
				> dT: time step. 
			>> params: of the model to integrate -- specific for each model. 
			>> modelPath, modelName: to locate the model. 

		Returns: 
			<< t_, x_: time and evolution variables. 

	"""

	# Loading the model: 
	mPath = os.path.join(modelPath, modelName); 
	modelFound = imp.find_module("model", [mPath]); 
	modelModuleName = imp.load_module("modelModuleName", *modelFound); 
	import modelModuleName as m; 

	# Unpacking integration parameters: 
	tMin = integrationParams["tMin"]; 
	tMax = integrationParams["tMax"]; 
	dT = integrationParams["dT"]; 

	# Load initial condition: 
	t_ = [t]; 
	x_ = x; 
	safeX = copy(x); 
	# Integrate backwards: 
	while(t > tMin): 
		t -= dT; 
		x -= m.F(x, params, kwArgs)*dT; 
		t_ += [t]; 
		x_ = np.append(x_, x, axis=0); 
	# Flip time and evolution: 
	t_ = list(np.flip(t_, 0)); 
	t = t_[-1]; 
	x_ = np.flip(x_, 0); 
	x = safeX; 
	# Integrate forward: 
	while(t < tMax): 
		t += dT; 
		x += m.F(x, params, kwArgs)*dT; 
		t_ += [t]; 
		x_ = np.append(x_, x, axis=0); 

	return (t_, x_); 


def integrateNoisyModel(t, x, integrationParams, params, modelPath, modelName, noiseArgs, kwArgs={}): 
	"""	integrateNoisyModel function: 

			This function integrates the model provided introducing noise in each iteration. It is possible to provide
			an intermediate point in time as initial condition and integrate backwards until a minimum time and then
			forwards until a maximum time step.

		Inputs: 
			>> t, x: time step at which the integration starts. 
			>> integrationParams: specifications for the integration. 
				> tMin: until which to integrate backwards. 
				> tMax: until which to integrate forwards. 
				> dT: time step. 
			>> params: of the model to integrate -- specific for each model. 
			>> modelPath, modelName: to locate the model. 
			>> noiseArgs: arguments specifying how to implement the noise (by now just std of normal distro). 
				> "sigma": standard deviation of a Gaussian distro that will introduce noise in the model derivative. 

		Returns: 
			<< t_, x_: time and evolution variables. 

	"""

	# Loading the model: 
	mPath = os.path.join(modelPath, modelName); 
	modelFound = imp.find_module("model", [mPath]); 
	modelModuleName = imp.load_module("modelModuleName", *modelFound); 
	import modelModuleName as m; 

	# Unpacking integration parameters: 
	tMin = integrationParams["tMin"]; 
	tMax = integrationParams["tMax"]; 
	dT = integrationParams["dT"]; 

	# Unpacking noise arguments: 
	sigma = noiseArgs["sigma"]; 

	# Load initial condition: 
	t_ = [t]; 
	x_ = x; 
	safeX = copy(x); 
	# Integrate backwards: 
	while(t > tMin): 
		t -= dT; 
		x -= (m.F(x, params, kwArgs) + np.random.normal(0, sigma, size=x.size))*dT; 
		t_ += [t]; 
		x_ = np.append(x_, x, axis=0); 
	# Flip time and evolution: 
	t_ = list(np.flip(t_, 0)); 
	t = t_[-1]; 
	x_ = np.flip(x_, 0); 
	x = safeX; 
	# Integrate forward: 
	while(t < tMax): 
		t += dT; 
		x += (m.F(x, params, kwArgs) + np.random.normal(0, sigma))*dT; 
		t_ += [t]; 
		x_ = np.append(x_, x, axis=0); 

	return (t_, x_); 


def integrateNoisyPositiveModel(t, x, integrationParams, params, modelPath, modelName, noiseArgs): 
	"""	integrateNoisyPositiveModel function: 

			This function integrates the model provided. It is possible to provide an intermediate point in time as
			initial condition and integrate backwards until a minimum time and then forwards until a maximum time step.
			This model assumes that all variables must be positive at all time. If one of them hits zero, it remains so
			for the rest of the integration.

		Inputs: 
			>> t, x: time step at which the integration starts. 
			>> integrationParams: specifications for the integration. 
				> tMin: until which to integrate backwards. 
				> tMax: until which to integrate forwards. 
				> dT: time step. 
			>> params: of the model to integrate -- specific for each model. 
			>> modelPath, modelName: to locate the model. 
			>> noiseArgs: arguments specifying how to implement the noise (by now just std of normal distro). 
				> "sigma": standard deviation of a Gaussian distro that will introduce noise in the model derivative. 

		Returns: 
			<< t_, x_: time and evolution variables. 

	"""

	# Loading the model: 
	mPath = os.path.join(modelPath, modelName); 
	modelFound = imp.find_module("model", [mPath]); 
	modelModuleName = imp.load_module("modelModuleName", *modelFound); 
	import modelModuleName as m; 

	# Unpacking integration parameters: 
	tMin = integrationParams["tMin"]; 
	tMax = integrationParams["tMax"]; 
	dT = integrationParams["dT"]; 

	# Unpacking noise arguments: 
	sigma = noiseArgs["sigma"]; 

	# Load initial condition: 
	t_ = [t]; 
	x_ = x; 
	safeX = copy(x); 
	nVars = x.shape[1]; 
	# Integrate backwards: 
	while(t > tMin): 
		t -= dT; 
		x -= m.F(x, params)*dT; 
		for iVar in range(nVars): 
			if (x[0, iVar]<=0.): 
				x[0, iVar] = 0; 
			else: 
				x[0, iVar] += np.random.normal(0, sigma)*dT; 
		t_ += [t]; 
		x_ = np.append(x_, x, axis=0); 
	# Flip time and evolution: 
	t_ = list(np.flip(t_, 0)); 
	t = t_[-1]; 
	x_ = np.flip(x_, 0); 
	x = safeX; 
	# Integrate forward: 
	while(t < tMax): 
		t += dT; 
		x += m.F(x, params)*dT; 
		for iVar in range(nVars): 
			if (x[0, iVar]<=0.): 
				x[0, iVar] = 0; 
			else: 
				x[0, iVar] += np.random.normal(0, sigma)*dT; 
		t_ += [t]; 
		x_ = np.append(x_, x, axis=0); 

	return (t_, x_); 


def extractPointsFromTimeSeries(t, t_, x_, dT): 
	""" 
		extractPointsFromTimeSeries function: 

			This function extracts a series of data points from a larger time series. The idea of this function is to
			pick up the time points that will be compared to those of the original data, and to do so independently of
			other calculations (e.g. chi^2, p, r) so we don't need to retrieve these points each time. 

		Inputs: 
			>> t: times from which we wish to return data. 
			>> t_, x_: time and points from the simulated evolution. 
			>> dT: time step of the model integration. 

		Returns: 
			<< x__: points of x_ that take places the closest to t possible. 

	"""

	x__ = []; 
	iData = 0; 
	maxData = len(t); 
	iT = 0; 
	fGo = True; 
	while(fGo): 
		nextT = t[iData]; 
		while(abs(t_[iT] - nextT) > dT): 
			iT += 1; 
		x__ += [x_[iT, :]]; 
		iData += 1; 
		if (iData==maxData or iData>20): 
			fGo = False; 
		else: 
			dataGap = t[iData] - nextT; 
			tGap = int(dataGap/dT); 
			iT += tGap; 

	return x__; 


def evaluateChi2(x, x__, fNormalizeNPoints = False, fNormalizeNVars = False): 
	"""	evaluateChi2 function: 

			This function computes the loss function chi^2 given the original data and corresponding data points from
			the simulated model.

		Inputs: 
			>> x: Original data to which we wish to compare the model. 
			>> x__: Data from the simulated model (x__ has been chose for notation consistency with other functions). 
			>> fNormalizeNPoints=False: Flag indicating whether we want to divide chi^2 by the number of data points. 
			>> fNormalizeNVars=False: Flag indicating whether we want to divide chi^2 by the number of variables. 

		Returns: 
			<< chi2: sum of the absolute value of the differences between each data point and its counterpart 
						in the simulated model. 

	"""

	chi2 = sum(sum(abs(x - x__))); 
	if fNormalizeNPoints: 
		chi2 /= x.shape[0]; 
	if fNormalizeNVars: 
		chi2 /= x.shape[1]; 

	return chi2; 


def visualizeParamPCs(paramsList, kwArgs): 
	"""	visualizeParamPCs function: 

			This function visualizes in different ways a list of different parameter sets from a model and their
			performance. Such a list of parameters should have been obtained from fits of the model to data. These are
			transformed into principal components to be able to visualize them. 

			By now: 
				- Plot eigenvalues. 
				- Plot 2D or 3D with the first two or three principal components. 

		Inputs: 
			>> paramsList: List containing the results of a lot of fits to the desired model. 
			>> kwArgs: key word arguments with further specifications. 
				> "plotEig": flag indicating whether to make a linear plot with the eigenvalues. 
				> "plot2PC": flag indicating whether to plot in 2D the first two principal components. 
				> "plot3PC": flag indicating whether to plot in 3D the first three principal components. 
				> "saveEig": flag indicating whether to save the plot of the eigenvalues. 
				> "save2PC": flag indicating whether to save the plot of the eigenvalues. 
				> "save3PC": flag indicating whether to save the plot of the eigenvalues. 
				> "fSave": base name of the file to save the plots. 

	"""

	# Unpacking number of fits, parameter names, number of parameters, parameters. 
	nFit = len(paramsList); 
	paramKeys = paramsList[0].keys(); 
	nKeys = len(paramKeys); 
	allParams = np.zeros([nFit, nKeys]); 
	for iFit in range(nFit): 
		for (iKey, key) in enumerate(paramKeys): 
			allParams[iFit, iKey] = paramsList[iFit][key]; 
	allParams = allParams.transpose(); 

	# Computing eigenvalues, eigenvectors: 
	paramsCov = np.cov(allParams); 
	(w, v) = np.linalg.eig(paramsCov); 
	if (kwArgs["plotEig"]): 
		fig = plt.figure(); 
		plt.plot(w); 

		if (kwArgs["saveEig"]): 
			fig.savefig(kwArgs["fSave"]+"eig.eps"); 

	# Projecting into eigenspace and plotting if desired: 
	allParams_ = np.dot(v, allParams); 

	# Plotting 2PCs: 
	if (kwArgs["plot2PC"]): 
		fig = plt.figure(); 
		plt.plot(allParams_[0,:], allParams_[1,:], 'x'); 
		# Saving to file? 
		if (kwArgs["save2PC"]): 
			fig.savefig(kwArgs["fSave"]+"2PC.eps"); 

	# Plotting 3PCs: 
	if (kwArgs["plot3PC"]): 
		fig = plt.figure(); 
		ax = fig.add_subplot(111, projection='3d'); 
		ax.scatter(allParams_[0,:], allParams_[1,:], allParams_[2,:]); 
		# Saving to file? 
		if (kwArgs["save3PC"]): 
			fig.savefig(kwArgs["fSave"]+"3PC.eps"); 

	plt.show(); 
	return; 


def visualizeLandscapePCs(paramsList, potentialList, kwArgs): 
	"""	visualizeLandscapePCs function: 

			This function visualizes a potential landscape (based on some error function -- e.g. chi^2 or chi_d^2) in
			the PC-space of the parameters resulting from a lot of fits. It might offer several options depending on the
			implementation. 

			By now:
				- 1PC: plot potential associated to just the first PC. 
					-- Line + potential in the vertical axis. 
				- 2PC: plot landscape potential associated to two PCs.  
					-- Probably the closest to standard landscape potential. 
					-- 2PC + potential in a scatter plot. 
				- 3PC: attempt at landscape potential associated to three PCs. 
					-- 3PC scatter plot + color-coded potential. 

		Inputs: 
			>> paramsList: List containing the results of a lot of fits to the desired model. 
			>> kwArgs: key word arguments with further specifications. 
				> "plot1PC": flag indicating whether to make a linear plot with the eigenvalues. 
				> "plot2PC": flag indicating whether to plot in 2D the first two principal components. 
				> "plot3PC": flag indicating whether to plot in 3D the first three principal components. 
				> "save1PC": flag indicating whether to save the plot of the eigenvalues. 
				> "save2PC": flag indicating whether to save the plot of the eigenvalues. 
				> "save3PC": flag indicating whether to save the plot of the eigenvalues. 
				> "fSave": base name of the file to save the plots. 

	"""
	# Unpacking number of fits, parameter names, number of parameters, parameters. 
	nFit = len(paramsList); 
	paramKeys = paramsList[0].keys(); 
	nKeys = len(paramKeys); 
	allParams = np.zeros([nFit, nKeys]); 
	for iFit in range(nFit): 
		for (iKey, key) in enumerate(paramKeys): 
			allParams[iFit, iKey] = paramsList[iFit][key]; 
	allParams = allParams.transpose(); 

	# Computing eigenvalues, eigenvectors: 
	paramsCov = np.cov(allParams); 
	(w, v) = np.linalg.eig(paramsCov); 

	# Projecting into eigenspace and plotting if desired: 
	allParams_ = np.dot(v, allParams); 

	# Plotting 1PC + potential: 
	if (kwArgs["plot1PC"]): 
		fig = plt.figure(); 
		plt.plot(allParams_[0,:], potentialList, 'x'); 
		# Saving to file? 
		if (kwArgs["save1PC"]): 
			fig.savefig(kwArgs["fSave"]+"1PC_potential.eps"); 

	if (kwArgs["plot2PC"]): 
		fig = plt.figure(); 
		ax = fig.add_subplot(111, projection='3d'); 
		ax.scatter(allParams_[0,:], allParams_[1,:], potentialList); 
		# Saving to file? 
		if (kwArgs["save2PC"]): 
			fig.savefig(kwArgs["fSave"]+"2PC_potential.eps"); 

	if (kwArgs["plot3PC"]): 
		fig = plt.figure(); 
		ax = fig.add_subplot(111, projection='3d'); 
		ax.scatter(allParams_[0,:], allParams_[1,:], allParams_[2,:], c = potentialList); 
		# Saving to file? 
		if (kwArgs["save3PC"]): 
			fig.savefig(kwArgs["fSave"]+"3PC_potential.eps"); 

	plt.show(); 
	return; 

