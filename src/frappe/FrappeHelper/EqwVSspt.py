import numpy as np
from spec_readspec import spec_readspec
import matplotlib.pyplot as plt
#import pylab as pl
from cardelli_extinction import cardelli_extinction
import spt_coding as scod
import glob
import numpy as np
from localreg import *
from scipy import interpolate as intp
from readcol_py3 import *
import os
import eqw_auto as eqw
from scipy.optimize import leastsq
from int_tabulated import *
import ray
#import MyFitter.em_lineFlux_func_rc as em


@ray.remote
def InerLoopFit(i,featureMatrix,errorMatrix,sptCode,SpTErr,mask,mcSamples,outSPTcode,rad,deg):
	feats = featureMatrix[i][mask[i]]
	errs = errorMatrix[i][mask[i]]
	sptCodeCut = sptCode[mask[i]]
	SpTErrCut = SpTErr[mask[i]]
	#print(mask[i])
	#ValuesOut[i,:] = localreg(sptCodeCut, feats, outSPTcode , degree=deg, kernel=rbf.epanechnikov, radius=rad)
	mcResults = np.zeros((mcSamples, len(outSPTcode)))
	for N in range(mcSamples):
		maxInt = len(feats)
		randSamp = np.random.randint(0,maxInt,maxInt)
		featSample = feats[randSamp]
		sptCodeCutSample = sptCodeCut[randSamp]
		#featSample = feats + (np.random.normal(0,1,len(feats))*errs)
		#sptCodeCutSample = sptCodeCut + (np.random.normal(0,1,len(feats))*SpTErrCut)#print(feats)#print(featSample)
		mcResults[N,:] = localreg(sptCodeCutSample, featSample, outSPTcode , degree=deg, kernel=rbf.gaussian, radius=rad)
	lowerOut = np.percentile(mcResults,  15.9, axis=0)
	upperOut = np.percentile(mcResults, 100-15.9, axis=0)
	medOut = np.nanmedian(mcResults,axis = 0)
	lowerOut[outSPTcode > np.max(sptCodeCut)] = np.nan
	lowerOut[outSPTcode < np.min(sptCodeCut)] = np.nan
	upperOut[outSPTcode > np.max(sptCodeCut)] = np.nan
	upperOut[outSPTcode < np.min(sptCodeCut)] = np.nan
	medOut[outSPTcode > np.max(sptCodeCut)] = np.nan
	medOut[outSPTcode < np.min(sptCodeCut)] = np.nan
	return medOut,lowerOut,upperOut

def poly_fit(wl_fit,fl_fit,plot='YES'):
	"""
	Fit the given points with a second degree polynomial
	"""
	Pol = lambda p, x: p[0]*x**2 + p[1]*x + p[2]
	res_fit = lambda p, x, y: (Pol(p,x) - y)

	out = leastsq(res_fit, [1.,1.,1.], args=(wl_fit,fl_fit), full_output=1)

	fit_par = out[0] #fit parameters out
	covar = out[1] #covariance matrix output

	if plot == 'YES' or plot == 'save':
		plt.plot(wl_fit,Pol(fit_par,wl_fit),'g--',lw=2)
		plt.show()

	return fit_par, res_fit(fit_par,wl_fit,fl_fit)


def continuum_estimate(wl,fl,wave_line,size_cont=1.,plot='YES',verbose='NO',namefile=None,pixel_buff=5,niter_max=20):#,cont_out='NO',err_cont_out='NO'):
	### I ASSUME INPUT WAVELENGTHS ARE IN NM!!! ###
	# PLOT THE LINE REGION
	flmin = 0.9*np.min(fl[ (wl>wave_line-size_cont) & (wl<wave_line+size_cont)])
	flmax = 1.1 * np.max(fl[ (wl>wave_line-size_cont) & (wl<wave_line+size_cont)])
	if plot == 'YES' or plot == 'stop':
		plt.figure(1)
		plt.subplot(211)
		plt.plot(wl,fl, 'k',lw=2)#,drawstyle='steps-mid',lw=2) # spectrum
		plt.axis([wave_line-1.0,wave_line+1.0, flmin,flmax]) # limits of the plot
	"""
	Estimate continuum:
		do it in a region of 2 nm around the theoretical line position
		by iteratively exclude the points which are outside a given sigma
	"""
	# select wl and fl around the theoretical line position
	wl_cont = wl[(wl >= wave_line-size_cont) & (wl <= wave_line+size_cont)]
	fl_cont = fl[(wl >= wave_line-size_cont) & (wl <= wave_line+size_cont)]
	wl_WithinLine = np.array([])
	# fit the continuum in this region using all points
	if plot == 'YES' or plot == 'stop':
		par,res = poly_fit(wl_cont,fl_cont)
	else:
		par,res = poly_fit(wl_cont,fl_cont,plot='NO')
	std_res = np.std(res)
	sigma_cut = 2.
	if plot == 'YES' or plot == 'stop':
		plt.subplot(212)
		plt.plot(wl_cont,res,'gd')
		plt.plot(wl_cont,np.repeat(sigma_cut*std_res,len(wl_cont)),'r',lw=2)	# plot the 2 sigma of the residuals
		plt.plot(wl_cont,np.repeat(-sigma_cut*std_res,len(wl_cont)),'r',lw=2)	# plot the 2 sigma of the residuals
		plt.xlim(wave_line-size_cont,wave_line+size_cont)
		plt.show()
		if plot =='stop':
			plt.draw()
			input('First guess')
		plt.close()
	# exclude points with res larger than 2sigma	- ITERATIONS
	niter = 0
	npoints = len(wl_cont)
	npoints_old = 1e15
	while npoints_old-npoints > 0 and niter <= niter_max:
		niter+=1
		wl_WithinLine = np.concatenate((wl_WithinLine,wl_cont[np.abs(res) > sigma_cut*std_res]))
		wl_cont = wl_cont[np.abs(res) <= sigma_cut*std_res]
		fl_cont = fl_cont[np.abs(res) <= sigma_cut*std_res]

		minwlInLine = np.min(wl_WithinLine)
		maxwlInLine = np.max(wl_WithinLine)

		npoints_old = npoints
		npoints = len(wl_cont)

		if plot == 'YES' or plot == 'stop' or plot == 'save':
			plt.figure(2)
			plt.subplot(211)
			plt.plot(wl,fl, 'k',lw=2)#,drawstyle='steps-mid',lw=2) # spectrum
			plt.axis([wave_line-size_cont,wave_line+size_cont, flmin,flmax]) # limits of the plot
			plt.plot([wl_cont[np.max(np.where(wl_cont<minwlInLine))-pixel_buff],wl_cont[np.max(np.where(wl_cont<minwlInLine))-pixel_buff]],[flmin,flmax],'r-')
			plt.plot([wl_cont[np.min(np.where(wl_cont>maxwlInLine))+pixel_buff],wl_cont[np.min(np.where(wl_cont>maxwlInLine))+pixel_buff]],[flmin,flmax],'r-')
			plt.xlabel(r'flux[$erg/s/cm^2/nm$]')
		if verbose != 'NO':
			print('POINTS %i' % len(wl_cont))

		# fit the continuum in this region using all points
		if plot == 'YES' or plot == 'stop' or plot=='save':
			par,res = poly_fit(wl_cont,fl_cont)
		else:
			par,res = poly_fit(wl_cont,fl_cont,plot='NO')

		# old_std_res = std_res
		std_res = np.std(res)
		if plot == 'YES' or plot == 'stop' or plot == 'save':
			plt.subplot(212)
			plt.plot(wl_cont,res,'gd')
			plt.plot(wl_cont,np.repeat(sigma_cut*std_res,len(wl_cont)),'r',lw=2)	# plot the 2 sigma of the residuals
			plt.plot(wl_cont,np.repeat(-sigma_cut*std_res,len(wl_cont)),'r',lw=2)	# plot the 2 sigma of the residuals
			plt.xlim(wave_line-size_cont,wave_line+size_cont)
			plt.ylabel('residuals')
			plt.xlabel('wavelength [nm]')
			if plot == 'save':
				plt.savefig('%s.png' % namefile)
			if plot =='stop':
				plt.show()
				plt.draw()
				input('Iterating')
			plt.close()


	#wl_cont_lims = np.array([wl_cont[np.max(np.where(wl_cont<wave_line))-pixel_buff],wl_cont[np.min(np.where(wl_cont>wave_line))+pixel_buff]])
	# My version for double line!!
	print(wl_WithinLine)
	minwlInLine = np.min(wl_WithinLine)
	maxwlInLine = np.max(wl_WithinLine)
	wl_cont_lims = np.array([wl_cont[np.max(np.where(wl_cont<minwlInLine))-pixel_buff],wl_cont[np.min(np.where(wl_cont>maxwlInLine))+pixel_buff]])
	print(wl_cont_lims)
	Pol = lambda p, x: p[0]*x**2 + p[1]*x + p[2]
	# this is different! I compute the continuum values underneat the line!!!! RC!!!
	#wl_contUnderLine = wl[(wl >= wave_line-size_cont) & (wl <= wave_line+size_cont)]

	return Pol(par,wave_line),std_res, wl_cont_lims,wl_cont	# this is the value of the continuum at the central wavelength of the line

# ------------------------------------------------------------


def continuum_estimate_carlo(wl,fl,wave_line,size_cont=1.,plot='YES',verbose='NO',namefile=None,pixel_buff=5,niter_max=20):#,cont_out='NO',err_cont_out='NO'):
	### I ASSUME INPUT WAVELENGTHS ARE IN NM!!! ###
	# PLOT THE LINE REGION
	flmin = 0.9*np.min(fl[ (wl>wave_line-size_cont) & (wl<wave_line+size_cont)])
	flmax = 1.1 * np.max(fl[ (wl>wave_line-size_cont) & (wl<wave_line+size_cont)])
	if plot == 'YES' or plot == 'stop':
		plt.figure(1)
		plt.subplot(211)
		plt.plot(wl,fl, 'k',lw=2)#,drawstyle='steps-mid',lw=2) # spectrum
		plt.axis([wave_line-1.0,wave_line+1.0, flmin,flmax]) # limits of the plot
	"""
	Estimate continuum:
		do it in a region of 2 nm around the theoretical line position
		by iteratively exclude the points which are outside a given sigma
	"""
	# select wl and fl around the theoretical line position
	wl_cont = wl[(wl >= wave_line-size_cont) & (wl <= wave_line+size_cont)]
	fl_cont = fl[(wl >= wave_line-size_cont) & (wl <= wave_line+size_cont)]
	wl_WithinLine = np.array([])
	# fit the continuum in this region using all points
	if plot == 'YES' or plot == 'stop':
		par,res = poly_fit(wl_cont,fl_cont)
	else:
		par,res = poly_fit(wl_cont,fl_cont,plot='NO')
	std_res = np.std(res)
	sigma_cut = 2.
	if plot == 'YES' or plot == 'stop':
		plt.subplot(212)
		plt.plot(wl_cont,res,'gd')
		plt.plot(wl_cont,np.repeat(sigma_cut*std_res,len(wl_cont)),'r',lw=2)	# plot the 2 sigma of the residuals
		plt.plot(wl_cont,np.repeat(-sigma_cut*std_res,len(wl_cont)),'r',lw=2)	# plot the 2 sigma of the residuals
		plt.xlim(wave_line-size_cont,wave_line+size_cont)
		plt.show()
		if plot =='stop':
			plt.draw()
			input('First guess')
		plt.close()
	# exclude points with res larger than 2sigma	- ITERATIONS
	niter = 0
	npoints = len(wl_cont)
	npoints_old = 1e15
	while npoints_old-npoints > 0 and niter <= niter_max:
		niter+=1
		#wl_WithinLine = np.concatenate((wl_WithinLine,wl_cont[np.abs(res) > sigma_cut*std_res]))
		wl_cont = wl_cont[np.abs(res) <= sigma_cut*std_res]
		fl_cont = fl_cont[np.abs(res) <= sigma_cut*std_res]
		npoints_old = npoints
		npoints = len(wl_cont)


		if verbose != 'NO':
			print('POINTS %i' % len(wl_cont))

		# fit the continuum in this region using all points
		if plot == 'YES' or plot == 'stop' or plot=='save':
			par,res = poly_fit(wl_cont,fl_cont,plot = 'NO')
		else:
			par,res = poly_fit(wl_cont,fl_cont,plot='NO')

		# old_std_res = std_res
		std_res = np.std(res)

	if plot == 'YES' or plot == 'stop' or plot == 'save':
			plt.figure(2)
			Pol = lambda p, x: p[0]*x**2 + p[1]*x + p[2]

			plt.subplot(211)
			plt.plot(wl,Pol(par,wl),'g--',lw=2)
			plt.plot(wl,fl, 'k',lw=2)#,drawstyle='steps-mid',lw=2) # spectrum
			plt.axis([wave_line-size_cont,wave_line+size_cont, flmin,flmax]) # limits of the plot
			plt.plot([wl_cont[np.max(np.where(wl_cont<wave_line))-pixel_buff],wl_cont[np.max(np.where(wl_cont<wave_line))-pixel_buff]],[flmin,flmax],'r-')
			plt.plot([wl_cont[np.min(np.where(wl_cont>wave_line))+pixel_buff],wl_cont[np.min(np.where(wl_cont>wave_line))+pixel_buff]],[flmin,flmax],'r-')
			plt.xlabel(r'flux[$erg/s/cm^2/nm$]')

	if plot == 'YES' or plot == 'stop' or plot == 'save':

		plt.subplot(212)
		plt.plot(wl_cont,res,'gd')
		plt.plot(wl_cont,np.repeat(sigma_cut*std_res,len(wl_cont)),'r',lw=2)	# plot the 2 sigma of the residuals
		plt.plot(wl_cont,np.repeat(-sigma_cut*std_res,len(wl_cont)),'r',lw=2)	# plot the 2 sigma of the residuals
		plt.xlim(wave_line-size_cont,wave_line+size_cont)
		plt.ylabel('residuals')
		plt.xlabel('wavelength [nm]')
		if plot == 'save':
			plt.savefig('%s.png' % namefile)
		if plot =='stop':
			plt.show()
			plt.draw()
			input('Iterating')
		plt.close()


	wl_cont_lims = np.array([wl_cont[np.max(np.where(wl_cont<wave_line))-pixel_buff],wl_cont[np.min(np.where(wl_cont>wave_line))+pixel_buff]])
	# My version for double line!!
	#print(wl_WithinLine)
	#minwlInLine = np.min(wl_WithinLine)
	#maxwlInLine = np.max(wl_WithinLine)
	#wl_cont_lims = np.array([wl_cont[np.max(np.where(wl_cont<minwlInLine))-pixel_buff],wl_cont[np.min(np.where(wl_cont>maxwlInLine))+pixel_buff]])
	#print(wl_cont_lims)
	Pol = lambda p, x: p[0]*x**2 + p[1]*x + p[2]
	# this is different! I compute the continuum values underneat the line!!!! RC!!!
	#wl_contUnderLine = wl[(wl >= wave_line-size_cont) & (wl <= wave_line+size_cont)]

	return Pol(par,wave_line),std_res, wl_cont_lims,wl_cont	# this is the value of the continuum at the central wavelength of the line

# ------------------------------------------------------------

# FUNCTION to compute the flux at a given wavelength of a spectrum (wl in nm!!!)
# if not provided, the wl interval on which the mean should be computed is assumed to be 8 nm
# centered on the nominal wavelength
def compute_flux_at_wl_std(wl_in,fl_in,wl0,interval=8):
	ind = (wl_in >= (wl0-interval*0.5)) & (wl_in <= (wl0+interval*0.5))
	flux_at = np.mean(fl_in[ind], dtype=np.float64)
	stddev_at = np.std(fl_in[ind], dtype=np.float64)
	return flux_at,stddev_at
# ------------------------------------------------------------


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def computeEW(wl,fl,wlline,size_cont=1.,dire =None):
	cut = interactiveLineCuts(wl,fl,wlline,size_cont)
	wl,fl = cut.wl_cut , cut.fl_cut
	if wl is None:
		return None, None
	cont,err_cont,wl_line_ext,wl_cont = continuum_estimate_carlo(wl,fl,wlline,size_cont=size_cont,plot='save',namefile=dire,pixel_buff=5)#,plot='stop')
	#print(len(cont))
	ind = np.where((wl>=np.min(wl_line_ext))&(wl<=np.max(wl_line_ext)))
	flux = int_tabulated(wl[ind],fl[ind]-cont)
	ew_auto = -1.*flux/cont
	err_flux = err_cont*np.diff(wl_line_ext)  # propagated for the size of the integration interval
	err_ew_auto = ((np.abs(flux)+err_flux)/cont)-np.abs(ew_auto)
	return np.squeeze(ew_auto), np.squeeze(err_ew_auto)

class EQWvsSpT:
    def __init__(self, dir = None):
        self.extractedFeatureValues = None
        self.extractedFeatureErrors = None
        self.extractedFeatureSptCodes = None
        self.Mask=None
        self.usedFeatures =None
        self.SpTErr = None
                # features obtained from non parametric fit
        self.sptCode = None
        self.medInterp = None
        self.lowerErrInterp = None
        self.upperErrInterp =  None
        #self.normWL = None
        if dir == None:
            print('You have to either set a grid of interpoleted features from a table using the readInterpFeat() method or create a new grid by running extractFeaturesXS() and nonParamFit')

        else:
            self.readInterpFeat(dir)
        # discrete points taken from observations and their errors




    def extractFeaturesXS(self,DirSpec,nameList,SpTlist,usedFeatures,SpTErr = [],dirOut = None):
        #values = np.array([0 for x in range(len(SptInfo['Name']))],dtype = float)
        #errors = np.array([0 for x in range(len(SptInfo['Name']))],dtype = float)
        Values = np.zeros((len(usedFeatures),len(SpTlist)))
        Errors = np.zeros((len(usedFeatures),len(SpTlist)))
        mask = np.zeros((len(usedFeatures),len(SpTlist)))
        #self.normWL = np.array([WLnorm,wlNormHalfWidth])
        #dirOut ='/Users/rclaes/python/functions/MyFitter/PlotsEQWClassIII/'
        if len(SpTErr) == 0:
            self.SpTErr = np.zeros(len(SpTlist))
        else:
            self.SpTErr = SpTErr
        fig =plt.figure(figsize=(10,10))
        #print(SpTlist)
        SptCodes = scod.spt_coding(SpTlist)
        ticklabels = np.array(['G4','','G6','','','G9','K0','','','K3','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
        ticks =  np.arange(-14,11,1)
        for i in range(len(nameList)):
            name  = nameList[i]
            #print(name)
            #flux_correction(uvbFile,1.,fileout=uvbFile,flux_un='erg/s/cm2/nm')
            print(name)
            visFile = glob.glob(DirSpec+'VIS/*%s*' %name)[0]
            Wvis,Fvis = spec_readspec(visFile)
            uvbFile = glob.glob(DirSpec+'UVB/*%s*' %name)[0]
            Wuvb,Fuvb = spec_readspec(uvbFile)
            nirFile = glob.glob(DirSpec+'NIR/*%s*' %name)[0]
            Wnir,Fnir = spec_readspec(nirFile)


            wl = np.concatenate([Wuvb[Wuvb<550],Wvis[Wvis>550],Wnir[Wnir>1020]])
            fl = np.concatenate([Fuvb[Wuvb<550],Fvis[Wvis>550],Fnir[Wnir>1020]])
            #fl[fl<0] = 0#np.nan
            # this creates a problem: J1111 and DENIS will have too high of a flux at wl ~3500!!!!!


            #fl = (1/cardelli_extinction(wl*10,av))*fl

            #fl_red = (cardelli_extinction(wl*10,AvErr))*fl
            #fl_derred = fl/(cardelli_extinction(wl*10,np.abs(AvErr)))


            #halfWidth = 0.5
            #fwlnorm = np.nanmedian(fl[(wl<WLnorm+halfWidth)&(wl>WLnorm-halfWidth)])
            #fwlnormErr = np.nanstd(fl[(wl<WLnorm+halfWidth)&(wl>WLnorm-halfWidth)])#np.nanstd(normSpectrum[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])])
            #normSpectrum = fl/fwlnorm

            for j in range(len(usedFeatures)):
                # this fits gaussian/Lorentzian profiles!
                #eqwVal, eqw_err, mode = eqw.eqw_auto(wl,fl,usedFeatures[j,0],size_cont=usedFeatures[j,1],plot='save',mode='best',name_out=dirOut+str(name)+'.png')
                #print(eqwVal)
                #this does a direct integration!
                eqwVal, eqw_err = computeEW(wl,fl,usedFeatures[j,0],size_cont=usedFeatures[j,1],dire = dirOut+str(name)+'_'+str(usedFeatures[j,0])+'nm.png')

                Values[j,i] = eqwVal
                Errors[j,i] =  eqw_err
                mask[j,i] =  Values[j,i] > 2*Errors[j,i]
                #print(Values[j])

        self.setExtarctedFeatures(usedFeatures, Values, Errors, SptCodes,mask.astype(bool))


    def setExtarctedFeatures(self,usedFeatures, Values, Errors, SptCodes,mask):
        self.usedFeatures = usedFeatures
        self.extractedFeatureValues = Values
        self.extractedFeatureErrors = Errors
        self.extractedFeatureSptCodes = SptCodes
        self.Mask=mask

    def getExtractedFeatures(self):
        usedF = self.usedFeatures
        return usedF, self.extractedFeatureSptCodes, self.extractedFeatureValues,self.extractedFeatureErrors, self.Mask


    def nonParamFit(self,nrOfPoints = 200,mcSamples =1000,rad =3.5,deg = 2,outFile =None):

        features,sptCode,featureMatrix,errorMatrix,mask = self.getExtractedFeatures()

        SpTErr = self.SpTErr
        #features = self.usedFeatures
        #featureMatrix = self.extractedFeatureValues
        #errorMatrix = self.extractedFeatureErrors
        if len(features) != len(featureMatrix):
            print('features and featureMatrix must have the same dimesion allong axis 0')
            sys.exit(1)
        if len(features) != len(featureMatrix):
            print('features and errorMatrix must have the same dimesion allong axis 0')
            sys.exit(1)
        outSPTcode = np.linspace(np.min(sptCode),np.max(sptCode),nrOfPoints)
        ValuesOut = np.empty((len(features),len(outSPTcode)))
        medOut = np.empty((len(features),len(outSPTcode)))
        lowerOut = np.empty((len(features),len(outSPTcode)))
        upperOut = np.empty((len(features),len(outSPTcode)))
        for i in range(len(features)):
            # remove the values that are 0 and have inf error!

            feats = featureMatrix[i][self.Mask[i]]
            errs = errorMatrix[i][self.Mask[i]]
            sptCodeCut = sptCode[self.Mask[i]]
            SpTErrCut = SpTErr[self.Mask[i]]
            print(self.Mask[i])


            #ValuesOut[i,:] = localreg(sptCodeCut, feats, outSPTcode , degree=deg, kernel=rbf.epanechnikov, radius=rad)

            mcResults = np.zeros((mcSamples, len(outSPTcode)))
            for N in range(mcSamples):
                #print(N)
                featSample = feats + (np.random.normal(0,1,len(feats))*errs)
                sptCodeCutSample = sptCodeCut + (np.random.normal(0,1,len(feats))*SpTErrCut)
                #print(feats)
                #print(featSample)
                mcResults[N,:] = localreg(sptCodeCutSample, featSample, outSPTcode , degree=deg, kernel=rbf.gaussian, radius=rad)

            lowerOut[i,:] = np.percentile(mcResults,  15.9, axis=0)
            upperOut[i,:] = np.percentile(mcResults, 100-15.9, axis=0)
            medOut[i,:] = np.nanmedian(mcResults,axis = 0)

            #self

            lowerOut[i,:][outSPTcode > np.max(sptCodeCut)] = np.nan
            lowerOut[i,:][outSPTcode < np.min(sptCodeCut)] = np.nan
            upperOut[i,:][outSPTcode > np.max(sptCodeCut)] = np.nan
            upperOut[i,:][outSPTcode < np.min(sptCodeCut)] = np.nan
            medOut[i,:][outSPTcode > np.max(sptCodeCut)] = np.nan
            medOut[i,:][outSPTcode < np.min(sptCodeCut)] = np.nan
            medAndErr = np.array([medOut,lowerOut,upperOut])
        if outFile != None:
            np.savez(outFile, medAndErr =medAndErr,sptCode = outSPTcode,usedFeatures = features)
        self.setInterpFeat(outSPTcode,medOut, lowerOut, upperOut)

    def nonParamFit_ray(self,nrOfPoints = 200,mcSamples =1000,rad =3.5,deg = 2,outFile =None):
    	features,sptCode,featureMatrix,errorMatrix,m = self.getExtractedFeatures()
    	SpTErr = self.SpTErr
    	if len(features) != len(featureMatrix):
    		print('features and featureMatrix must have the same dimesion allong axis 0')
    		sys.exit(1)
    	if len(features) != len(featureMatrix):
    		print('features and errorMatrix must have the same dimesion allong axis 0')
    		sys.exit(1)
    	outSPTcode = np.linspace(np.min(sptCode),np.max(sptCode),nrOfPoints)
    	ValuesOut = np.empty((len(features),len(outSPTcode)))
    	medOut = np.empty((len(features),len(outSPTcode)))
    	lowerOut = np.empty((len(features),len(outSPTcode)))
    	upperOut = np.empty((len(features),len(outSPTcode)))
    	mask = self.Mask
    	ray.shutdown()
    	ray.init()
    	pool = ray.get([InerLoopFit.remote(i,featureMatrix,errorMatrix,sptCode,SpTErr,mask,mcSamples,outSPTcode,rad,deg)for i in range(len(features))])
    	pool1 = np.array(pool)
    	print("HERE")
    	print(pool1.shape)
    	medOut,lowerOut,upperOut  = pool1[:,0,:], pool1[:,1,:], pool1[:,2,:]
    	medAndErr = np.array([medOut,lowerOut,upperOut])
    	if outFile != None:
    		np.savez(outFile, medAndErr =medAndErr,sptCode = outSPTcode,usedFeatures = features)
    	self.setInterpFeat(outSPTcode,medOut,lowerOut,upperOut)
    	ray.shutdown()

        #uses Monte carlo as described in Anas paper!!
    def nonParamFit_OnBestFit(self,nrOfPoints = 200,mcSamples =1000,rad =3.5,deg = 2,outFile =None):

        features,sptCode,featureMatrix,errorMatrix = self.getExtractedFeatures()

        SpTErr = self.SpTErr
        #features = self.usedFeatures
        #featureMatrix = self.extractedFeatureValues
        #errorMatrix = self.extractedFeatureErrors
        if len(features) != len(featureMatrix):
            print('features and featureMatrix must have the same dimesion allong axis 0')
            sys.exit(1)
        if len(features) != len(featureMatrix):
            print('features and errorMatrix must have the same dimesion allong axis 0')
            sys.exit(1)
        outSPTcode = np.linspace(np.min(sptCode),np.max(sptCode),nrOfPoints)
        ValuesBestFitOut = np.empty((len(features),len(outSPTcode)))

        medOut = np.empty((len(features),len(outSPTcode)))
        lowerOut = np.empty((len(features),len(outSPTcode)))
        upperOut = np.empty((len(features),len(outSPTcode)))
        for i in range(len(features)):
            # remove the values that are 0 and have inf error!

            feats = featureMatrix[i][self.Mask[i]]
            errs = errorMatrix[i][self.Mask[i]]
            sptCodeCut = sptCode[self.Mask[i]]
            SpTErrCut = SpTErr[self.Mask[i]]

            print(self.Mask[i])


            ValuesBestFitOut[i,:] = localreg(sptCodeCut, feats, outSPTcode , degree=deg, kernel=rbf.epanechnikov, radius=rad)
            ValuesBestFitLoop = localreg(sptCodeCut, feats , degree=deg, kernel=rbf.epanechnikov, radius=rad)


            mcResults = np.zeros((mcSamples, len(outSPTcode)))
            for N in range(mcSamples):
                #print(N)
                featSample = ValuesBestFitLoop + (np.random.normal(0,1,len(feats))*errs)
                sptCodeCutSample = sptCodeCut + (np.random.normal(0,1,len(feats))*SpTErrCut)
                #print(feats)
                #print(featSample)
                mcResults[N,:] = localreg(sptCodeCutSample, featSample, outSPTcode , degree=deg, kernel=rbf.gaussian, radius=rad)

            lowerOut[i,:] = np.percentile(mcResults,  15.9, axis=0)
            upperOut[i,:] = np.percentile(mcResults, 100-15.9, axis=0)
            medOut[i,:] = np.nanmedian(mcResults,axis = 0)

            #self
            medAndErr = np.array([ValuesBestFitOut,lowerOut,upperOut])
        if outFile != None:
            np.savez(outFile, medAndErr =medAndErr,sptCode = outSPTcode,usedFeatures = features)
        self.setInterpFeat(outSPTcode,medOut, lowerOut, upperOut)

        #return ValuesOut,outSPTcode, medOut, lowerOut, upperOut
            #preform Monte carlo simulation to compute the errors


    ######
    #TODO: hightlight points not used!
    ######
    def plotAllInterpIndividualy(self, outdir ='./'):
        for i in range(len(self.medInterp)):
            fig, axs = plt.subplots(2, 1,figsize=(16,9),sharex='col', gridspec_kw={'height_ratios': [3, 1]})
            axs[0].set_title('wl range: '+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+' nm')
            axs[0].plot(self.sptCode,self.medInterp[i])
            mask= self.Mask
            axs[0].errorbar(self.extractedFeatureSptCodes[mask[i]],self.extractedFeatureValues[i][mask[i]],self.extractedFeatureErrors[i][mask[i]],xerr = self.SpTErr[mask[i]],c='k',linestyle='',marker ='o', label = str(self.usedFeatures[i]))
            axs[0].errorbar(self.extractedFeatureSptCodes[~mask[i]],self.extractedFeatureValues[i][~mask[i]],self.extractedFeatureErrors[i][~mask[i]],xerr = self.SpTErr[~mask[i]],c= 'r',alpha=0.5,linestyle='',marker ='o', label = str(self.usedFeatures[i]))

            #axs[0].plot(outSPTcode,med[i])
            axs[0].fill_between(self.sptCode,self.lowerErrInterp[i], self.upperErrInterp[i],alpha = 0.4)
            axs[0].set_xlabel('SpT code')
            axs[0].set_ylabel('EQW[nm]')
            #axs[0].set_yscale('log')
            #residuals
            interp = np.interp(self.extractedFeatureSptCodes[mask[i]],self.sptCode,self.medInterp[i])
            axs[1].scatter(self.extractedFeatureSptCodes[mask[i]],(self.extractedFeatureValues[i][mask[i]]-interp)/self.extractedFeatureErrors[i][mask[i]],c='k')
            interp = np.interp(self.extractedFeatureSptCodes[~mask[i]],self.sptCode,self.medInterp[i])
            axs[1].scatter(self.extractedFeatureSptCodes[~mask[i]],(self.extractedFeatureValues[i][~mask[i]]-interp)/self.extractedFeatureErrors[i][~mask[i]],c='r')
            axs[1].plot(self.sptCode,np.zeros(len(self.sptCode)))
            axs[1].set_xlabel('SpT code')
            axs[1].set_ylabel(r'$(f_{spec.} - f_{fit.})/\sigma$')
            #axs[1].set_ylim(-0.6,0.6)
            ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
            ticks =  np.arange(-10,11,1)
            plt.xticks(ticks,ticklabels)
            plt.savefig(outdir+'wl_range:'+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+'nm.png')

    def plotAllInterpIndividualy_moreResiduals_log(self, outdir ='./'):
        for i in range(len(self.medInterp)):
            fig, axs = plt.subplots(2, 1,figsize=(16,9),sharex='col', gridspec_kw={'height_ratios': [3, 1]})
            axs[0].set_title('wl range: '+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+' nm')
            axs[0].plot(self.sptCode,self.medInterp[i])
            mask= self.Mask
            axs[0].errorbar(self.extractedFeatureSptCodes[mask[i]],self.extractedFeatureValues[i][mask[i]],self.extractedFeatureErrors[i][mask[i]],xerr = self.SpTErr[mask[i]],c='k',linestyle='',marker ='o', label = str(self.usedFeatures[i]))
            axs[0].errorbar(self.extractedFeatureSptCodes[~mask[i]],self.extractedFeatureValues[i][~mask[i]],self.extractedFeatureErrors[i][~mask[i]],xerr = self.SpTErr[~mask[i]],c= 'r',alpha=0.5,linestyle='',marker ='o', label = str(self.usedFeatures[i]))

            #axs[0].plot(outSPTcode,med[i])
            axs[0].fill_between(self.sptCode,self.lowerErrInterp[i], self.upperErrInterp[i],alpha = 0.4)
            axs[0].set_xlabel('SpT code')
            axs[0].set_ylabel('EQW[nm]')
            axs[0].set_yscale('log')
            #residuals
            interp = np.interp(self.extractedFeatureSptCodes[mask[i]],self.sptCode,self.medInterp[i])
            axs[1].scatter(self.extractedFeatureSptCodes[mask[i]],(self.extractedFeatureValues[i][mask[i]]-interp)/self.extractedFeatureErrors[i][mask[i]],c='k')
            interp = np.interp(self.extractedFeatureSptCodes[~mask[i]],self.sptCode,self.medInterp[i])
            axs[1].scatter(self.extractedFeatureSptCodes[~mask[i]],(self.extractedFeatureValues[i][~mask[i]]-interp)/self.extractedFeatureErrors[i][~mask[i]],c='r')
            axs[1].plot(self.sptCode,np.zeros(len(self.sptCode)))
            axs[1].set_xlabel('SpT code')
            axs[1].set_ylabel(r'$(f_{spec.} - f_{fit.})/\sigma$')
            #axs[1].set_ylim(-0.6,0.6)
            ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
            ticks =  np.arange(-10,11,1)
            plt.xticks(ticks,ticklabels)
            plt.savefig(outdir+'wl_range:'+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+'nm_LOG.png')



    def plotInterpTogetherNoErr(self, outdir ='./'):
        plt.figure()
        for i in range(len(self.medInterp)):
            plt.plot(self.sptCode,self.medInterp[i],label = str(self.usedFeatures[i]))
            #plt.fill_between(self.sptCode,self.lowerErrInterp[i], self.upperErrInterp[i],alpha = 0.4)
            legend = plt.legend()
            plt.xlabel('SpT code')

            plt.ylabel('EQW[nm]')
            ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
            ticks =  np.arange(-10,11,1)
            plt.xticks(ticks,ticklabels)
            legend.set_title("wavelength range: [nm]")
            plt.savefig(outdir+'allInterpNoErr.png')


    def plotInterpTogetherWithErr(self, outdir ='./'):
        plt.figure()
        for i in range(len(self.medInterp)):
            plt.plot(self.sptCode,self.medInterp[i],label = str(self.usedFeatures[i]))
            plt.fill_between(self.sptCode,self.lowerErrInterp[i], self.upperErrInterp[i],alpha = 0.4)
            legend = plt.legend()
            plt.xlabel('SpT code')

            plt.ylabel('EQW[nm]')
            ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
            ticks =  np.arange(-10,11,1)
            plt.xticks(ticks,ticklabels)
            legend.set_title("wavelength range: [nm]")
            if i%3 == 0:
                plt.savefig(outdir+'allInterpWithErr'+str(i/3)+'.png')
                plt.figure()
                #plt.ylim(-0.1,2)

    def plotAllInterpIndividualy_FitResiduals(self, outdir ='./',logScale = False):
    	for i in range(len(self.medInterp)):
    		fig, axs = plt.subplots(2, 1,figsize=(5,6.5),sharex='col', gridspec_kw={'height_ratios': [4, 1]})
    		axs[0].set_title('wl range: '+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+' nm')
    		axs[0].plot(self.sptCode,self.medInterp[i])
    		mask= self.Mask
    		axs[0].errorbar(self.extractedFeatureSptCodes[mask[i]],self.extractedFeatureValues[i][mask[i]],self.extractedFeatureErrors[i][mask[i]],xerr = self.SpTErr[mask[i]],c='k',linestyle='',marker ='o', label = str(self.usedFeatures[i]))
    		axs[0].errorbar(self.extractedFeatureSptCodes[~mask[i]],self.extractedFeatureValues[i][~mask[i]],self.extractedFeatureErrors[i][~mask[i]],xerr = self.SpTErr[~mask[i]],c= 'r',alpha=0.5,linestyle='',marker ='o', label = str(self.usedFeatures[i]))
    		axs[0].fill_between(self.sptCode,self.lowerErrInterp[i], self.upperErrInterp[i],alpha = 0.4)
    		axs[0].set_xlabel('SpT code')
    		axs[0].set_ylabel('EQW[nm]')
    		if logScale:
    			axs[0].set_yscale('log')
    		#residuals
    		interp = np.interp(self.extractedFeatureSptCodes[mask[i]],self.sptCode,self.medInterp[i])
    		interpErr = np.interp(self.extractedFeatureSptCodes[mask[i]],self.sptCode,( self.upperErrInterp[i] - self.lowerErrInterp[i])/2)
    		axs[1].scatter(self.extractedFeatureSptCodes[mask[i]],(self.extractedFeatureValues[i][mask[i]]-interp)/interpErr,c='k')
    		interp = np.interp(self.extractedFeatureSptCodes[~mask[i]],self.sptCode,self.medInterp[i])
    		interpErr = np.interp(self.extractedFeatureSptCodes[~mask[i]],self.sptCode,( self.upperErrInterp[i] - self.lowerErrInterp[i])/2)
    		axs[1].scatter(self.extractedFeatureSptCodes[~mask[i]],(self.extractedFeatureValues[i][~mask[i]]-interp)/interpErr,c='r')
    		axs[1].plot(self.sptCode,np.zeros(len(self.sptCode)))
    		axs[1].set_xlabel('SpT code')
    		axs[1].set_ylabel(r'$(f_{spec.} - f_{fit.})/\sigma$')
    		ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
    		ticks =  np.arange(-10,11,1)#plt.xlim(-10,6)
    		if logScale:
    			plt.xticks(ticks,ticklabels)
    			plt.tight_layout()
    			plt.savefig(outdir+'wl_range:'+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+'nm_LOG.png')
    		else:
    			plt.xticks(ticks,ticklabels)
    			plt.tight_layout()
    			plt.savefig(outdir+'wl_range:'+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+'nm.png')
    		plt.close()

    def getFeatsAtSpt_symetricErr(self, SpT):
        #move the next line outside, set as class atribute!
        medInterpolant = intp.interp1d(self.sptCode,self.medInterp)
        meds = medInterpolant(SpT)
        lowErrsInterpolant = intp.interp1d(self.sptCode,self.lowerErrInterp)
        lowErrs = lowErrsInterpolant(SpT)
        upErrsInterpolant = intp.interp1d(self.sptCode,self.upperErrInterp)
        upErrs = upErrsInterpolant(SpT)
        error = (meds - lowErrs) + (upErrs - meds)
        error *= 0.5
        if float(SpT) < -8.5:
            medsMin8,error = self.getFeatsAtSpt_symetricErr(-8.5)
        #if float(SpT) > 9:
        #    medsMin8,error = self.getFeatsAtSpt_symetricErr(8)
        return meds,error


    def setInterpFeat(self,outSPTcode,med,lower,upper):
        self.sptCode = outSPTcode
        self.medInterp = med
        self.lowerErrInterp = lower
        self.upperErrInterp = upper

    def getInterpFeat(self):
        return self.usedFeatures,self.sptCode, self.medInterp, self.lowerErrInterp, self.upperErrInterp

    def getUsedInterpFeat(self):
        return self.usedFeatures


    def readInterpFeat(self,dirInterpfeat):
        npz = np.load(dirInterpfeat)
        featuresandErrs =npz['medAndErr']
        self.sptCode = npz['sptCode']
        self.usedFeatures = npz['usedFeatures']
        self.medInterp = featuresandErrs[0]
        self.lowerErrInterp = featuresandErrs[1]
        self.upperErrInterp = featuresandErrs[2]

def readMixClassIII(min_chi_sq_cl3,PATH_CLASSIII,wlNorm =731,average = False):
    name_cl3,SpT_cl3 = readcol_py3(PATH_CLASSIII+'summary_classIII_final.txt',2,format='A,X,A,X',skipline=1)
    #compute the sptcode
    sptCodes = scod.spt_coding(SpT_cl3)
    #print(min_chi_sq_cl3.dtype)
    # calculate the difference array
    difference_array = np.absolute(sptCodes-float(min_chi_sq_cl3))

    # find the index of minimum element from the array
    index = np.argmin(difference_array)
    #the sptcode of the nearest spt in array
    WLnorm = wlNorm
    halfWidth=0.5
    nearestSptInArray = sptCodes[index]
    if not average:
        i = np.random.randint(0,len(name_cl3[sptCodes == nearestSptInArray]))
        cl3_toSelectModel = name_cl3[sptCodes == nearestSptInArray][i]
        print('!!!!!')
        print(cl3_toSelectModel)
        print('!!!!!')
        wl,fl = spec_readspec(PATH_CLASSIII +'VIS/flux_'+cl3_toSelectModel+'_VIS_corr_phot.fits')

        fwlnorm = np.nanmedian(fl[(wl<WLnorm+halfWidth)&(wl>WLnorm-halfWidth)])

        pathUVB = PATH_CLASSIII +'UVB/flux_'+cl3_toSelectModel+'_UVB_phot.fits'

        pathVIS = PATH_CLASSIII +'VIS/flux_'+cl3_toSelectModel+'_VIS_corr_phot.fits'

        pathNIR = PATH_CLASSIII +'NIR/flux_'+cl3_toSelectModel+'_NIR_corr_scaled_phot.fits'
        #else:
        #    raise Exception('you need to specify which arm you want usingthe three options uvb, vis or nir')
        wl_cl3UVB,fl_cl3UVB = spec_readspec(pathUVB)
        wl_cl3VIS,fl_cl3VIS = spec_readspec(pathVIS)
        wl_cl3NIR,fl_cl3NIR = spec_readspec(pathNIR)
        fl_cl3UVB = fl_cl3UVB/fwlnorm
        fl_cl3VIS = fl_cl3VIS/fwlnorm
        fl_cl3NIR = fl_cl3NIR/fwlnorm
    else:
        raise Exception('you have yet to implement this')
    return wl_cl3UVB,fl_cl3UVB,wl_cl3VIS,fl_cl3VIS,wl_cl3NIR,fl_cl3NIR,cl3_toSelectModel


def convertSpTtoTeffHH14(spt_code):
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = "SpT_Teff_relation_hh14_short_codes.dat"
    abs_file_path = os.path.join(script_dir, rel_path)
    #relDir = './SpT_Teff_relation_hh14_short_codes.dat'
    relation = np.genfromtxt(abs_file_path,usecols=(1,2),skip_header=1,dtype=[('sptCode',float),('Teff',float)])
    Teff = np.interp(spt_code,relation['sptCode'],relation['Teff'])

    return Teff

# modified version of cardelli_extinction that returns:
#a + b/R_v
# to be used in error propagation of extinction.

def cardelli_extinction_a_plus_bOverRv(wave,Rv=3.1):


  # print 'Av = ',Av, 'Rv = ',Rv

  x = 10000./ wave                # Convert to inverse microns
  npts = len(x)
  a = np.zeros(npts)
  b = np.zeros(npts)
#******************************

  good = (x > 0.3) & (x  < 1.1)	       #Infrared
  Ngood = np.count_nonzero(good == True)
  if Ngood > 0:
    a[good] =  0.574 * x[good]**(1.61)
    b[good] = -0.527 * x[good]**(1.61)

#******************************

  good = (x >= 1.1) & (x < 3.3)            #Optical/NIR
  Ngood = np.count_nonzero(good == True)
  if Ngood > 0:           #Use new constants from O'Donnell (1994)
    y = x[good] - 1.82
    c1 = [-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1.0]  #New coefficients
    c2 = [3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0.0] #from O'Donnell (1994)
#   c1 = [ 1. , 0.17699, -0.50447, -0.02427,  0.72085,    $ #Original
#                 0.01979, -0.77530,  0.32999 ]               #coefficients
#   c2 = [ 0.,  1.41338,  2.28305,  1.07233, -5.38434,    $ #from CCM89
#                -0.62251,  5.30260, -2.09002 ]   # If you use them remember to revert them

    a[good] = np.polyval(c1,y)
    b[good] = np.polyval(c2,y)


#******************************

  good = (x >= 3.3) & (x < 8)            #Mid-UV
  Ngood = np.count_nonzero(good == True)
  if Ngood > 0:
    y = x[good]
    F_a = np.zeros(Ngood)
    F_b = np.zeros(Ngood)
    good1 = (y > 5.9)
    Ngood1 = len(good1)
    if Ngood1 > 0:
    	y1 = y[good1] - 5.9
    	F_a[good1] = -0.04473 * y1**2 - 0.009779 * y1**3
    	F_b[good1] =   0.2130 * y1**2  +  0.1207 * y1**3
    a[good] =  1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a
    b[good] = -3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b


#   *******************************

  good = (x >= 8) & (x <= 11)         #Far-UV
  Ngood = np.count_nonzero(good == True)
  if Ngood > 0:
    y = x[good] - 8.
    c1 = [-0.07, 0.137, -0.628, -1.073]
    c2 = [0.374, -0.42, 4.257, 13.67]
    a[good] = np.polyval(c1,y)
    b[good] = np.polyval(c2,y)

#   *******************************

#=======

  return (a + b/Rv)[0]



class interactiveLineCuts:
	def __init__(self,wl,fl,wlline,size_cont):
		self.wl,self.fl = wl,fl
		self.wl_cut,self.fl_cut = wl,fl
		fig, ax = plt.subplots()
		self.fig = fig
		self.ax = ax
		self.wlline,self.size_cont = wlline,size_cont
		spectrum, = ax.plot(wl,fl,'k-',label='spectrum')
		ax.set_xlim(self.wlline-self.size_cont,self.wlline+self.size_cont)
		ax.set_ylim(0,np.max(self.fl[(self.wl>self.wlline-self.size_cont)& (self.wl<self.wlline+self.size_cont)]))
		ax.set_xlabel('wavelength [nm]')
		ax.set_ylabel(r'Flux [$erg s^{-1} cm^{-2} nm^{-1}$]')
		ax.set_title(str(wlline)+' nm line')
		fig.canvas.draw()
		fig.canvas.mpl_connect('key_press_event',self.ontype)
		fig.canvas.mpl_connect('button_press_event',self.onclick)
		fig.canvas.mpl_connect('pick_event',self.onpick)
		plt.show(block  = True)
		#ax.hold()
	# when none of the toolbar buttons is activated and the user clicks in the
    # plot somewhere, compute the median value of the spectrum in a 10angstrom
    # window around the x-coordinate of the clicked point. The y coordinate
    # of the clicked point is not important. Make sure the continuum points
    # `feel` it when it gets clicked, set the `feel-radius` (picker) to 5 points
	def onclick(self,event):
		toolbar = plt.get_current_fig_manager().toolbar
		if event.button==1 and toolbar.mode=='':
			print(self.wl_cut, event.xdata)
			nearestWL = find_nearest(self.wl_cut, event.xdata)#((event.xdata-0.05)<=wave) & (wave<=(event.xdata+5))
			y = self.fl_cut[self.wl_cut == nearestWL]
			self.ax.plot(event.xdata,y,'rs',ms=10,picker=5,label='cont_pnt')
		self.fig.canvas.draw()

	def onpick(self,event):
		if event.mouseevent.button==3:
			if hasattr(event.artist,'get_label') and event.artist.get_label()=='cont_pnt':
				event.artist.remove()

	def ontype(self,event):
		if event.key=='enter':
			cont_pnt_coord = []
			for artist in plt.gca().get_children():
				if hasattr(artist,'get_label') and artist.get_label()=='cont_pnt':
					cont_pnt_coord.append(artist.get_data())
					artist.remove()
				elif hasattr(artist,'get_label') and artist.get_label()=='cut':
					artist.remove()
			cont_pnt_coord = np.array(cont_pnt_coord)[...,0]
			#print(cont_pnt_coord)
			if len(cont_pnt_coord) != 2:
				print('can only remove the section between 2 points')
			else:
				sort_array = np.argsort(cont_pnt_coord[:,0])
				wl,y = cont_pnt_coord[sort_array].T
				self.fl_cut = self.fl_cut[(self.wl_cut <= wl[0])|(self.wl_cut >= wl[1])]
				self.wl_cut = self.wl_cut[(self.wl_cut <= wl[0])|(self.wl_cut >= wl[1])]
				#print(self.wl_cut,self.fl_cut)
				self.ax.plot(self.wl_cut,self.fl_cut,'r',lw=2,label='cut',zorder =-1)
		elif event.key=='r':
			self.ax.cla()
			self.ax.plot(self.wl,self.fl,'k-')
			self.wl_cut,self.fl_cut = self.wl,self.fl
			self.ax.set_xlim(self.wlline-self.size_cont,self.wlline+self.size_cont)
			self.ax.set_ylim(0,np.max(self.fl[(self.wl>self.wlline-self.size_cont)& (self.wl<self.wlline+self.size_cont)]))
		elif event.key=='w':
			plt.close()
		elif event.key=='n':
			self.fl_cut = None#self.fl_cut[(self.wl_cut <= wl[0])|(self.wl_cut >= wl[1])]
			self.wl_cut = None#self.wl_cut[(self.wl_cut <= wl[0])|(self.wl_cut >= wl[1])]
			plt.close()
		self.fig.canvas.draw()
