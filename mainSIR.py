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
fIn = open("./setupSIR.txt", 'r'); 
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
modelName = "SIR"; 

# Loading data: 
dataWholeName = os.path.join(fDataPath, fDataFile); 
(t, x, nPoints_, nVars_) = h.loadData(dataWholeName); 
x_ = np.zeros([nPoints, nVars]); 
x_[:,0] = x[:,0]; x_[:,1] = x[:,1]; 



# Fitting variables to polynomials: 
h.crossValPolyFit(t, x_, nVars, nPoints, k, maxDeg, wD); 
# h.plotPolynomialsWithData(t, x_, nVars, wD); 				# Uncomment to plot polynomials. 


# Initializing fit parameters: 
fitParams = {}; 
fitParams["eta"] = 0.01; 
fitParams["haltCond"] = 10E-10; 
fitParams["itMax"] = 10000; 
fitParams["nFit"] = 10; 
initModelArgs = {}; 
fitModelArgs = {}; 
(paramsList_MW, chi_d2List_MW, whyStop_MW) = h.fitModelManyTimes(t, nVars, wD, modelPath, modelName, fitParams, initModelArgs); 



# Initializing integration parameters: 
integrationParams = {}; 
integrationParams["dT"] = 0.1; 
integrationParams["tMin"] = t[0]; 
integrationParams["tMax"] = t[-1]; 




