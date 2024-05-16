# FRAPPE (version 0.1)

## Installation:

FRAPPE is currently not set up as a python package.
Currently only the functionality to interpolate between a set of X-Shooter templates and access the interpolated spectra is provided.
To make use of the provided functions simply add the PhotFeatures_Ray.py file to you python path.


## Usage:



```
Directory pointing towards and interpolated grid
dirInterp = '/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeVIS.pnz'

# First the class III object needs to be initialized
classIIIreadIn = pf.classIII(dirInterp)

usedFeatures = classIIIreadIn.getUsedInterpFeat()
wlObs = (usedFeatures[:,0]+usedFeatures[:,1])/2
normWLandWidth = classIIIreadIn.getUsedNormWl()

features, outSPTcode, ValuesOut, lower, upper = classIIIreadIn.getInterpFeat()
```

## Coming later:
FRAPPE version 1.0 will include 
