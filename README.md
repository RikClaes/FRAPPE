# FRAPPE (v0.1)

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
sptCode = spt_coding('M3')
# then can sample the interpolated grid at a given SpT
features,errors = classIIIreadIn.getFeatsAtSpt_symetricErr(sptCode)
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
import numpy as np
import MyFitter.PhotFeatures_Ray as pf
import matplotlib.pyplot as plt

SptFile = '~/.../ClassIII_info.txt'
SptInfo = np.genfromtxt(SptFile,usecols=(0,2,4),skip_header=1,dtype=[('Name','U64'),('Spt','U4'),('SptErr','f8')])


nameList = SptInfo['Name']
Spts = SptInfo['Spt']

features = np.array([[335-5,335+5],
                     [340-5,340+5],
                     [357.5-5,357.5+5],
                     [355-6,355+6],#
                     [400-4,400+4], 
                     [450-4,450+4],
                     [475-2,475+2],
                     [461-3,461+3],
                     [703-1,703+1],
                     [707-1,707+1],
                     [710-1,710+1],
                     [715-1,715+1]]
                   )

# compute normalized fluxes and uncertainties withing these ranges
classIIIFeat = pf.classIII()
classIIIFeat.extractFeaturesXS_ray(dirSpec,nameList,Spts,features,WLnorm= 731,SpTErr = SptInfo['SptErr'])

output = '/someFileName' #the output file you want to produce .npz should not be included in the name
#run the non parametric fits
classIIICarlosFeat.nonParamFit_ray(200,1000,rad =2.5,deg =1,outFile = output)

# the following plots can be produced to inspect the resulting interpolated spectrum
# first the wavelength ranges are plotted individually
classIIICarlosFeat.plotAllInterpIndividualy_FitResiduals(outDir+'figsGoodInterp_NoBTsettl/')
# plot several different interpolations together
classIIICarlosFeat.plotInterpTogetherWithErr(outDir+'figsGoodInterp_NoBTsettl/')
```
Below are two example of the produced figures
![plot](https://github.com/RikClaes/FRAPPE/blob/main/Figures/wl_range%3A709.5-710.5nm.png)
![plot](https://github.com/RikClaes/FRAPPE/blob/main/Figures/allInterpWithErr3.0.png)

## Coming later:
FRAPPE version 1.0 will include the code used to fit th UV continuum excess of accreting T Tauri Stars
