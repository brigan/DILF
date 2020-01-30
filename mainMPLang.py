"""

	main.py: 

		This file is a script to test different parts of the MInLF. 

"""

import numpy as np; 
import matplotlib.pyplot as plt
import os; 
import helper as h; 
import sys


# Loading setup file: 
fIn = open("./setupLang.txt", 'r'); 
sRaw = fIn.read(); 
fIn.close(); 
sLines = sRaw.split('\n'); 
fDataPath = sLines[0]; 
fDataFile = sLines[1]; 
nVars = int(sLines[2]); 
nPoints = int(sLines[3]); 
k = int(sLines[4]); 			# For k-fold x-validation. 
maxDeg = int(sLines[5]); 
wD = "./WD/"; 
modelPath = "/home/luis/Desktop/Research_comeBack/DILF/Code/Models/"; 
modelName = "MPLang"; 

# Loading data: 
dataWholeName = os.path.join(fDataPath, fDataFile); 
(t, x, nPoints_, nVars_) = h.loadData(dataWholeName); 
x_ = np.zeros([nPoints, nVars]); 
x_[:,0] = x[:,0]; x_[:,1] = x[:,2]; 

# Fitting variables to polynomials: 
h.crossValPolyFit(t, x_, nVars, nPoints, k, maxDeg, wD); 

fitParams = {}; 
fitParams["eta"] = 0.01; 
fitParams["haltCond"] = 10E-10; 
fitParams["itMax"] = 10000; 
fitParams["nFit"] = 10; 
initModelArgs = {}; 
fitModelArgs = {}; 
(params, chi_d2, whyStop) = h.fitModel(t, nVars, wD, modelPath, modelName, fitParams, initModelArgs, fitModelArgs); 
print chi_d2; 
print params; 
print whyStop; 



# Import model: 
mPath = os.path.join(modelPath, modelName); 
sys.path.insert(0, mPath);
import model as m; 

# Integrating forward in time: 
t_ = t[0]; 
x_ = np.array([x[0,0], x[0,2]]); 
x_ = np.expand_dims(x_, axis=0); 
integrationParams = {}; 
integrationParams["tMin"] = t_; 
integrationParams["tMax"] = t[-1]; 
integrationParams["dT"] = 0.1; 
integrationKwargs = {}; 
(tEvol, xEvol) = h.integrateModel(t_, x_, integrationParams, params, modelPath, modelName, integrationKwargs); 

# Plotting: 
plt.figure(); 
plt.plot(tEvol, xEvol); 
plt.plot(t, x); 
plt.show(); 





