# FRAPPE (version 0.1)

## Installation:

FRAPPE is currently not set up as a python package.
Currently only the functionality to interpolate between a set of X-Shooter templates and access the interpolated spectra is provided.
To make use of the provided functions simply add the PhotFeatures_Ray.py file to you python path.
The required packages are: 
  - numpy
  - matplotlib
  - localreg (https://github.com/sigvaldm/localreg)
  - scipy
  - ray


## Usage:
### loading and plotting an interpolates spectrum at a given SpT
The below eaxmple shows how to load an interpolated spectrum at a given SpT and plot it.

```
Directory pointing towards and interpolated grid
dirInterp = '/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeVIS.pnz'

import matplotlib.pyplot as plt
import PhotFeatures_Ray as pf

# First the class III object needs to be initialized
classIIIreadIn = pf.classIII(dirInterp)

# then load in the wavelength ranges used in the interterpolated grid
usedFeatures = classIIIreadIn.getUsedInterpFeat()
# compute the central wavelengths of the ranges:
wl = (usedFeatures[:,0]+usedFeatures[:,1])/2
# get the wl range used to normalize the spectra
normWLandWidth = classIIIreadIn.getUsedNormWl()

# we convert a SpT to a SpT code
spt_coding
# then can sample the interpolated grid at a given SpT
featuresUVB,errorsUVB = classIIIreadInUVB.getFeatsAtSpt_symetricErr(sptCode)
#The spectrum can then be plotted using
plt.figure()
plt.plot(wl,np.log10(features)+i,'tab:blue',linewidth=.5)
plt.fill_between(wl,np.log10(features-errors) +i, np.log10(features +errors)+i,color='tab:blue',alpha = 0.4)


```
this results in the following plot:

![plot](https://github.com/RikClaes/FRAPPE/blob/main/Figures/InterpVIS_m3.png)


PhotFeatures_Ray.py also includes the neccecary tools to generate your own interpolated spectra based on X-Shooter.
A description of how these methods work can be found in Claes et al. in prep.
Below an example is given of how this can be done. 
A future update will allow for other spectra to be included.


```
```



## Coming later:
FRAPPE version 1.0 will include 
