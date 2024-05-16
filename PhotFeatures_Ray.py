import numpy as np
import matplotlib.pyplot as plt
import glob
import numpy as np
from localreg import *
from scipy import interpolate as intp
import os
import ray


    """
    #+
    # NAME:
    #       class initialization for a "class III"
    # PURPOSE:
    #       Contains the information regarding an interpolated grid of templates.
    #       Provides methods to access and plot the interpolated spectrum contained in the object
    # CALLING SEQUENCE:
    #      ObjectName =  classIII() for an empty object used to generate a new interolated gridFile
    #      ObjectName =  classIII(dir = something.npz) to load an existing grid
    # INPUTS:
    #     self
    #
    # OPTIONAL INPUT KEYWORDS:
    #     dir the directory containing a .npz file of an interpolated grid
    #
    # EFFECT:
    #      None
    #
    """
class classIII:
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
        self.normWL = None
        if dir == None:
            print('You have to either set a grid of interpoleted features from a table using the readInterpFeat() method or create a new grid by running extractFeaturesXS() and nonParamFit')

        else:
            self.readInterpFeat(dir)
        # discrete points taken from observations and their errors





        """
    	#+
    	# NAME:
    	#       extractFeaturesXS_ray
    	# PURPOSE:
    	#       Computes the normalized fluxes and respective uncertainties to be used in the non paramteric fitting method
    	#
    	# CALLING SEQUENCE:
    	#     ObjectName.extractFeaturesXS_ray(DirSpec,nameList,SpTlist,usedFeatures,AvErr = 0.2,WLnorm = 751,wlNormHalfWidth = 0.5,SpTErr = [])
    	#
    	# INPUTS:
    	#     self = ObjectName
        #     DirSpec = the folder containing the spectra, should contain subfolders UVB,VIS,NIR for the different arms
        #     nameList = Array containing the names of the targets for to be used spectra. (This will be used to associate the fits file, so this name should be contained in the names of the fits files)
        #     SpTlist = Array containing the SpT of the to be used spectra, the SpT's should correspond to nameList.
        #     usedFeatures = np array containing the wavelength ranges to be considered, format should be the following: [[335-5,335+5],[340-5,340+5],[357.5-5,357.5+5],...]
    	# OPTIONAL INPUT KEYWORDS:
    	#     AvErr = the error on the extinction in mag.
    	#     WLnorm = the normalization wavelength same units as spectral axit of the spectra
        #     wlNormHalfWidth = width of the used normalization range
    	#     SpTErr = the uncertinty on the SpTs in SpTlist
    	# EFFECT:
        #      the parameters:
    	#      self.extractedFeatureValues
        #      self.extractedFeatureErrors
        #      self.extractedFeatureSptCodes
        #      are set to the extracted values so that nonParamFit_ray or nonParamFit_ray can be ran.
    	#
    	#
    	"""
    def extractFeaturesXS_ray(self,DirSpec,nameList,SpTlist,usedFeatures,AvErr = 0.2,WLnorm = 751,wlNormHalfWidth = 0.5,SpTErr = []):
        #values = np.array([0 for x in range(len(SptInfo['Name']))],dtype = float)
        #errors = np.array([0 for x in range(len(SptInfo['Name']))],dtype = float)
        Values = np.zeros((len(usedFeatures),len(SpTlist)))
        Errors = np.zeros((len(usedFeatures),len(SpTlist)))
        self.normWL = np.array([WLnorm,wlNormHalfWidth])

        if len(SpTErr) == 0:
            self.SpTErr = np.zeros(len(SpTlist))
        else:
            self.SpTErr = SpTErr
        fig =plt.figure(figsize=(10,10))
        #print(SpTlist)
        SptCodes = spt_coding(SpTlist)
        ticklabels = np.array(['G4','','G6','','','G9','K0','','','K3','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
        ticks =  np.arange(-14,11,1)
        ray.shutdown()
        ray.init()

        pool = ray.get([InerLoopExtract.remote(nameList[i],i,DirSpec,usedFeatures,AvErr,WLnorm,wlNormHalfWidth,SpTErr)for i in range(len(nameList))])

            #for i in range(len(nameList)):

        print(np.array(pool).shape)
        Values = np.array(pool)[:,0].transpose()
        Errors = np.array(pool)[:,1].transpose()
        mask = np.array(pool)[:,2].transpose().astype(np.bool)
        print(Values.shape)

        print(Errors.shape)
        print(mask)
        print(mask[0])
        #mask = Values > 0
        self.setExtarctedFeatures(usedFeatures, Values, Errors, SptCodes,mask)

        #del self.Values, self.Errors,
        ray.shutdown()



        """
    	#+
    	# NAME:
    	#       setExtarctedFeatures
    	# PURPOSE:
    	#       sets the "extracted features of the object", to be called in extractFeaturesXS_ray
    	#
    	# CALLING SEQUENCE:
    	#     ObjectName.setExtarctedFeatures(usedFeatures, Values, Errors, SptCodes,mask)
    	#
    	# INPUTS:
    	#     usedFeatures = np array containing the wavelength ranges to be considered, format should be the following: [[335-5,335+5],[340-5,340+5],[357.5-5,357.5+5],...]
    	#     Values = array containing the median normalized fluxes in the abaove wl ranges
    	#     Errors = the uncertainties on Values
        #     SptCodes = array with spts, corresponding to Values
    	#     mask = boolean array to indicate not to be used median fluxes
    	# EFFECT:
        #      the parameters:
    	#      self.extractedFeatureValues
        #      self.extractedFeatureErrors
        #      self.extractedFeatureSptCodes
        #      are set to the extracted values so that nonParamFit_ray or nonParamFit_ray can be ran.
    	#
    	#
    	"""
    def setExtarctedFeatures(self,usedFeatures, Values, Errors, SptCodes,mask):
        self.usedFeatures = usedFeatures
        self.extractedFeatureValues = Values
        self.extractedFeatureErrors = Errors
        self.extractedFeatureSptCodes = SptCodes
        self.Mask=mask

        """
    	#+
    	# NAME:
    	#       setExtarctedFeatures
    	# PURPOSE:
    	#       sets the "extracted features of the object", to be called in extractFeaturesXS_ray
    	#
    	# CALLING SEQUENCE:
    	#     ObjectName.getExtractedFeatures()
    	#
    	# INPUTS:
    	#     None
    	# Return:
        #      usedF = the wavelength ranges of the classIII object, format: [[335-5,335+5],[340-5,340+5],[357.5-5,357.5+5],...]
    	#      self.extractedFeatureValues = array containing the median normalized fluxes in the abaove wl ranges
        #      self.extractedFeatureErrors = the uncertainties on Values
        #      self.extractedFeatureSptCodes = array with spts, corresponding to Values
    	#
    	#
    	"""
    def getExtractedFeatures(self):
        usedF = self.usedFeatures
        return usedF, self.extractedFeatureSptCodes, self.extractedFeatureValues,self.extractedFeatureErrors

        """
    	#+
    	# NAME:
    	#       nonParamFit
    	# PURPOSE:
    	#       sets the "extracted features of the object", to be called in extractFeaturesXS_ray
    	#
    	# CALLING SEQUENCE:
    	#     ObjectName.nonParamFit(nrOfPoints = 200,mcSamples =1000,rad =3.5,deg = 2,outFile =None)
    	#
    	#  OPTIONAl INPUTS:
    	#     nrOfPoints = the number of SpT subclass point at which each non parametric fit is computed
        #     mcSamples = the number of Monte Carlo simulation samples to be drawn
        #     rad = the bandwidth of the local polynomial fits
        #     deg = the degree of the local polynomial fit
        #     outFile = dir+filename where to write the generated npz file, which containes the interpolation if not given, no file will be produced
    	# effect:
        #      sets the interpolted grid atributed of this class III OBJECT
        #      produces a .npz file which stores the interpolated grid
    	#
    	#
    	"""
    def nonParamFit(self,nrOfPoints = 200,mcSamples =1000,rad =2.5,deg = 2,outFile =None):

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
            medAndErr = np.array([medOut,lowerOut,upperOut])
        if outFile != None:
            np.savez(outFile, medAndErr =medAndErr,sptCode = outSPTcode,usedFeatures = features,normalWL = self.normWL)
        self.setInterpFeat(outSPTcode,medOut, lowerOut, upperOut)

        """
    	#+
    	# NAME:
    	#       nonParamFit_ray
    	# PURPOSE:
    	#       same as nonParamFit, but faster since it is parralellized using ray
    	# SEE nonParamFit!!
    	#
    	"""
    def nonParamFit_ray(self,nrOfPoints = 200,mcSamples =1000,rad =3.5,deg = 2,outFile =None):

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
            np.savez(outFile, medAndErr =medAndErr,sptCode = outSPTcode,usedFeatures = features,normalWL = self.normWL)
        self.setInterpFeat(outSPTcode,medOut,lowerOut,upperOut)





        """
    	#+
    	# NAME:
    	#       plotAllInterpIndividualy
    	# PURPOSE:
    	#       plots the nonparameteic fit for each wl range individualy
    	#
    	# CALLING SEQUENCE:
    	#     ObjectName.plotAllInterpIndividualy(outdir ='./',logScale = False)
    	#
    	#  OPTIONAl INPUTS:
    	#     outdir = output dir for the ccreated
        #     logScale = boolean if true the y axis will be in log scale
    	# effect:
        #      creates plots
    	#
    	#
    	"""
    def plotAllInterpIndividualy(self, outdir ='./',logScale = False):
        for i in range(len(self.medInterp)):
            fig, axs = plt.subplots(2, 1,figsize=(11,6),sharex='col', gridspec_kw={'height_ratios': [3, 1]})
            axs[0].set_title('wl range: '+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+' nm')
            axs[0].plot(self.sptCode,self.medInterp[i])
            mask= self.Mask
            axs[0].errorbar(self.extractedFeatureSptCodes[mask[i]],self.extractedFeatureValues[i][mask[i]],self.extractedFeatureErrors[i][mask[i]],xerr = self.SpTErr[mask[i]],c='k',linestyle='',marker ='o', label = str(self.usedFeatures[i]))
            axs[0].errorbar(self.extractedFeatureSptCodes[~mask[i]],self.extractedFeatureValues[i][~mask[i]],self.extractedFeatureErrors[i][~mask[i]],xerr = self.SpTErr[~mask[i]],c= 'r',alpha=0.5,linestyle='',marker ='o', label = str(self.usedFeatures[i]))

            #axs[0].plot(outSPTcode,med[i])
            axs[0].fill_between(self.sptCode,self.lowerErrInterp[i], self.upperErrInterp[i],alpha = 0.4)
            axs[0].set_xlabel('SpT code')
            axs[0].set_ylabel('f(range)/f'+str(self.normWL[0]))
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
            #axs[1].set_ylim(-0.6,0.6)
            #ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5',''])
            #ticks =  np.arange(-10,11-4,1)

            #ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5',''])
            ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
            ticks =  np.arange(-10,11,1)
            #plt.xlim(-10,6)
            plt.xticks(ticks,ticklabels)
            if logScale:
                plt.savefig(outdir+'wl_range:'+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+'nm_LOG.png')
            else:
                plt.savefig(outdir+'wl_range:'+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+'nm.png')
            plt.close()


            """
        	#+
        	# NAME:
        	#       plotAllInterpIndividualy
        	# PURPOSE:
        	#       plots all the non parametric fits together, without the error, or the used datapoints
        	#
        	# CALLING SEQUENCE:
        	#     ObjectName.plotInterpTogetherNoErr(outdir ='./')
        	#
        	#  OPTIONAl INPUTS:
        	#     outdir = output dir for the ccreated
            #     logScale = boolean if true the y axis will be in log scale
        	# effect:
            #      creates plots
        	#
        	#
        	"""
    def plotInterpTogetherNoErr(self, outdir ='./'):
        plt.figure()
        for i in range(len(self.medInterp)):
            plt.plot(self.sptCode,self.medInterp[i],label = str(self.usedFeatures[i]))
            #plt.fill_between(self.sptCode,self.lowerErrInterp[i], self.upperErrInterp[i],alpha = 0.4)
            legend = plt.legend()
            plt.xlabel('SpT code')

            plt.ylabel('f(range)/f'+str(self.normWL[0]))
            ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
            ticks =  np.arange(-10,11,1)
            plt.xticks(ticks,ticklabels)
            legend.set_title("wavelength range: [nm]")
            plt.savefig(outdir+'allInterpNoErr.png')
            plt.close()

        """
        #+
        # NAME:
        #       plotAllInterpIndividualy
        # PURPOSE:
        #       plots all the non parametric fits together, with the uncertainty and without the used data points
        #
        # CALLING SEQUENCE:
        #     ObjectName.plotInterpTogetherWithErr(outdir ='./')
        #
        #  OPTIONAl INPUTS:
        #     outdir = output dir for the ccreated
        #     logScale = boolean if true the y axis will be in log scale
        # effect:
        #      creates plots
        #
        #
        """
    def plotInterpTogetherWithErr(self, outdir ='./'):
        plt.figure(figsize = (5,4))
        x = np.linspace(len(self.medInterp)-1,0,len(self.medInterp)).astype(int)
        for i in x:
            plt.plot(self.sptCode,self.medInterp[i],label = str(self.usedFeatures[i][0])+'-'+ str(self.usedFeatures[i][1])+'[nm]')
            plt.fill_between(self.sptCode,self.lowerErrInterp[i], self.upperErrInterp[i],alpha = 0.4)
            legend = plt.legend()
            plt.xlabel('SpT code')

            plt.ylabel('f(range)/f'+str(self.normWL[0]))
            ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
            ticks =  np.arange(-10,11,1)
            plt.xticks(ticks,ticklabels)
            legend.set_title("wavelength range: ")
            if i == 0:
                plt.tight_layout()
                plt.savefig(outdir+'allInterpWithErr'+str((i)/3)+'.png')
                plt.close()
            if (len(self.medInterp)-1 - i)%3 == 0 and i != (len(self.medInterp) - 1) :
                plt.tight_layout()
                plt.savefig(outdir+'allInterpWithErr'+str((len(self.medInterp)-1 - i)/3 )+'.png')
                plt.close()
                plt.figure(figsize = (5,4))
                #plt.ylim(-0.1,2)
                #plt.ylim(-0.1,2)

        """
        #+
        # NAME:
        #       plotAllInterpIndividualy
        # PURPOSE:
        #       retunrs an interpolated spectrum at a given SpT, the errors are assumed to be symetric
        #
        # CALLING SEQUENCE:
        #     flux, err = ObjectName.plotAllInterpIndividualy(outdir ='./',logScale = False)
        #
        #  OPTIONAl INPUTS:
        #     outdir = output dir for the ccreated
        #     logScale = boolean if true the y axis will be in log scale
        # effect:
        #      meds = the normalized fluxes, corresponting to the used wl ranges
        #      error =  the corresponding uncertainties
        #
        """
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

        """
        #+
        # NAME:
        #       setInterpFeat
        # PURPOSE:
        #       To be called by nonParamFit and init to set the interpolated grid
        #
        # CALLING SEQUENCE:
        #     ObjectName.setInterpFeat(self,outSPTcode,med,lower,upper)
        #
        #  OPTIONAl INPUTS:
        #     outSPTcode = the spectran type code to be set
        #     med = the median fluxes to be set
        #     lower = the 1 sigma lower bound uncertaintie
        #     upper = the 1 sigma upper bound uncertaintie
        # effect:
        #      sets the
        #       self.sptCode
        #       self.medInterp
        #       self.lowerErrInterp
        #       self.upperErrInterp
        #       attributes of the class III object
        """
    def setInterpFeat(self,outSPTcode,med,lower,upper):
        self.sptCode = outSPTcode
        self.medInterp = med
        self.lowerErrInterp = lower
        self.upperErrInterp = upper

        """
        #+
        # NAME:
        #       setInterpFeat
        # PURPOSE:
        #       Returns all interpolated features
        #
        # CALLING SEQUENCE:
        #     usedFeatures,sptCode, medInterp, lowerErrInterp, upperErrInterp = ObjectName.setInterpFeat()
        #
        #  OPTIONAl INPUTS:
        # OUTPUTS:
        #       sets the
        #       usedFeatures =  array containing the wl ranges used in the interpolated grid
        #       sptCode = the spt codes corresponding to each point for which a non parametric fit was computed
        #       medInterp = 2d array containing the non parametric fits for all wl ranges
        #       lowerErrInterp = 2d array containing the 1 sigma lower bound uncertainty non parametric fits for all wl ranges
        #       upperErrInterp = 2d array containing the 1 sigma upper bound uncertainty non parametric fits for all wl ranges
        #
        """
    def getInterpFeat(self):
        return self.usedFeatures,self.sptCode, self.medInterp, self.lowerErrInterp, self.upperErrInterp

        """
        #+
        # NAME:
        #       getUsedInterpFeat
        # PURPOSE:
        #       Returns all interpolated features
        #
        # CALLING SEQUENCE:
        #     usedFeatures = ObjectName.setInterpFeat()
        #
        #  OPTIONAl INPUTS:
        # OUTPUTS:
        #       sets the
        #       usedFeatures =  array containing the wl ranges used in the interpolated grid
        #
        """
    def getUsedInterpFeat(self):
        return self.usedFeatures

        """
        #+
        # NAME:
        #       setInterpFeat
        # PURPOSE:
        #       Returns the normalization wl
        #
        # CALLING SEQUENCE:
        #     normWL = ObjectName.setInterpFeat()
        #
        #  OPTIONAl INPUTS:
        # OUTPUTS:
        #       sets the
        #       normWL =  the normalization wl of the interpolated grid
        #
        """
    def getUsedNormWl(self):
        return self.normWL

        """
        #+
        # NAME:
        #       readInterpFeat
        # PURPOSE:
        #       Returns the normalization wl
        #
        # CALLING SEQUENCE:
        #     ObjectName.readInterpFeat()
        #
        # INPUTS:
        #       dirInterpfeat = directory of the .npz file you want to load
        #   EFFECT:
        #       Loads th .npz file ands sets the relevant atributes og this classIII object
        #
        """
    def readInterpFeat(self,dirInterpfeat):
        npz = np.load(dirInterpfeat)
        featuresandErrs =npz['medAndErr']
        self.sptCode = npz['sptCode']
        self.usedFeatures = npz['usedFeatures']
        self.medInterp = featuresandErrs[0]
        self.lowerErrInterp = featuresandErrs[1]
        self.upperErrInterp = featuresandErrs[2]
        self.normWL = npz['normalWL']



    """
    #+
    # NAME:
    #       readMixClassIII
    # PURPOSE:
    #       reads in a normalized random class III template spectrum of the SpT closest to the one given
    #
    # CALLING SEQUENCE:
    #     readMixClassIII
    #
    # INPUTS:
    #       min_chi_sq_cl3 = the SpT at which you want a spectrum
    #       PATH_CLASSIII = dir pointing to the spectra should have subfolders UVB,VIS,NIR
    #       wlNorm = the wl at which the spectrum is normalized
    #       average = False, True to be implemented!!
    #   returns:
    #      wl_cl3UVB,fl_cl3UVB,wl_cl3VIS,fl_cl3VIS,wl_cl3NIR,fl_cl3NIR,cl3_toSelectModel
    # that is the wl and fluxed in the 3 XS arms and the name of the selected class III template
    #
    """
def readMixClassIII(min_chi_sq_cl3,PATH_CLASSIII,wlNorm =731,average = False):
    clsIIIinfo = np.genfromtxt(PATH_CLASSIII+'summary_classIII_final.txt',usecols=(0,2),skip_header=1,dtype=[('Name','U64'),('Spt','U4')])
    name_cl3 = clsIIIinfo['Name']
    SpT_cl3 = clsIIIinfo['Spt']
    #compute the sptcode
    sptCodes = spt_coding(SpT_cl3)
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


    """
    #+
    # NAME:
    #       readMixClassIII
    # PURPOSE:
    #       reads in a normalized random class III template spectrum of the SpT closest to the one given, also returns the SpT of this spectrum
    #
    # CALLING SEQUENCE:
    #     readMixClassIII
    #
    # INPUTS:
    #       min_chi_sq_cl3 = the SpT at which you want a spectrum
    #       PATH_CLASSIII = dir pointing to the spectra should have subfolders UVB,VIS,NIR
    #       wlNorm = the wl at which the spectrum is normalized
    #       average = False, True to be implemented!!
    #   returns:
    #      wl_cl3UVB,fl_cl3UVB,wl_cl3VIS,fl_cl3VIS,wl_cl3NIR,fl_cl3NIR,cl3_toSelectModel, spt
    # that is the wl and fluxed in the 3 XS arms and the name of the selected class III template and its SpT
    #
    """
def readMixClassIII_withSpT(min_chi_sq_cl3,PATH_CLASSIII,wlNorm =731,average = False):
    clsIIIinfo = np.genfromtxt(PATH_CLASSIII+'summary_classIII_final.txt',usecols=(0,2),skip_header=1,dtype=[('Name','U64'),('Spt','U4')])
    name_cl3 = clsIIIinfo['Name']
    SpT_cl3 = clsIIIinfo['Spt']
    #compute the sptcode
    sptCodes = spt_coding(SpT_cl3)
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
        spt = SpT_cl3[sptCodes == nearestSptInArray][i]
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
    return wl_cl3UVB,fl_cl3UVB,wl_cl3VIS,fl_cl3VIS,wl_cl3NIR,fl_cl3NIR,cl3_toSelectModel, spt


"""
#+
# NAME:
#       InerLoopExtract
# PURPOSE:
#       to be used in extractFeaturesXS_ray for parrallelization
#
# CALLING SEQUENCE:
#     IDK why you would want to
#
"""
@ray.remote#(num_returns=2)
def InerLoopExtract(name,i,DirSpec,usedFeatures,AvErr,WLnorm,wlNormHalfWidth,SpTErr):
        #name  = nameList[i]
        #print(name)
        #flux_correction(uvbFile,1.,fileout=uvbFile,flux_un='erg/s/cm2/nm')
        Values = np.zeros(len(usedFeatures))
        Errors = np.zeros(len(usedFeatures))
        mask = np.array([False for i in range(len(usedFeatures))])
        visFile = glob.glob(DirSpec+'VIS/*%s*' %name)[0]
        Wvis,Fvis = spec_readspec(visFile)
        uvbFile = glob.glob(DirSpec+'UVB/*%s*' %name)[0]
        Wuvb,Fuvb = spec_readspec(uvbFile)
        nirFile = glob.glob(DirSpec+'NIR/*%s*' %name)[0]
        Wnir,Fnir = spec_readspec(nirFile)

        print(i)
        wl = np.concatenate([Wuvb[Wuvb<550],Wvis[(Wvis>550)&(Wvis<=1020)],Wnir[Wnir>1020]])
        fl = np.concatenate([Fuvb[Wuvb<550],Fvis[(Wvis>550)&(Wvis<=1020)],Fnir[Wnir>1020]])
        #fl[fl<0] = 0#np.nan
        # this creates a problem: J1111 and DENIS will have too high of a flux at wl ~3500!!!!!


        #fl = (1/cardelli_extinction(wl*10,av))*fl

        #fl_red = (cardelli_extinction(wl*10,AvErr))*fl
        #fl_derred = fl/(cardelli_extinction(wl*10,np.abs(AvErr)))


        #halfWidth = 0.5
        #fwlnorm = np.nanmedian(fl[(wl<WLnorm+halfWidth)&(wl>WLnorm-halfWidth)])
        #fwlnormErr = np.nanstd(fl[(wl<WLnorm+halfWidth)&(wl>WLnorm-halfWidth)])#np.nanstd(normSpectrum[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])])
        #normSpectrum = fl/fwlnorm
        fwlnorm = np.nanmedian(fl[(wl<=WLnorm+wlNormHalfWidth)&(wl>=WLnorm-wlNormHalfWidth)])
        #fwlnormErr_noise = np.nanstd(fl[(wl<=WLnorm+wlNormHalfWidth)&(wl>=WLnorm-wlNormHalfWidth)])
        #####
        #new!!
        #####
        #fwlnormErr_fCalib = fwlnorm/20
        #fwlnormErr = np.sqrt((fwlnormErr_noise**2) + (fwlnormErr_fCalib**2))
        fwlnormErr = np.nanstd(fl[(wl<=WLnorm+wlNormHalfWidth)&(wl>=WLnorm-wlNormHalfWidth)])
        for j in range(len(usedFeatures)):

            #print(features[j,0])

            fluxNotScaledInRange  = np.nanmedian(fl[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])])
            fluxInRange = fluxNotScaledInRange/fwlnorm

            #ErrNotScaledInRange_noise = np.nanstd(fl[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])])
            #ErrNotScaledInRange_fCalib = fluxNotScaledInRange/20
            #ErrNotScaledInRange = np.sqrt((ErrNotScaledInRange_noise**2)+ (ErrNotScaledInRange_fCalib**2))
            ErrNotScaledInRange = np.nanstd(fl[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])])
            ErrFluxInRange = np.abs(fluxInRange)*np.sqrt((ErrNotScaledInRange/fluxNotScaledInRange)**2 + (fwlnormErr/fwlnorm)**2)
            #ErrFluxInRange = np.nanstd(fl[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])])


            Values[j]  = fluxInRange
            wlReddening = 10*(usedFeatures[j,0] + usedFeatures[j,1])/2
            CardelliCte = cardelli_extinction_a_plus_bOverRv(np.array([wlReddening]),Rv=3.1) - cardelli_extinction_a_plus_bOverRv(np.array([WLnorm]),Rv=3.1)
            errCardelliRatio = np.abs(-0.4*np.log(10) * CardelliCte*AvErr)
            #print(1 - (cardelli_extinction(np.array([wlReddening]),AvErr)))
            #print(ErrFluxInRange/fluxInRange)
            term1 = (ErrFluxInRange/fluxInRange) **2
            #print(term1)
            term2 = (errCardelliRatio)**2
            errFlDerred = np.abs(fluxInRange)* np.sqrt(term1+ term2) #np.sqrt((ErrFluxInRange/fluxInRange)**2
            #print(errCardelliRatio)
            Errors[j] =   errFlDerred
            mask[j] = ((fluxNotScaledInRange/ErrNotScaledInRange) >=0.)
            #mask[j] = ((((errFlDerred) <=0.25) or (fluxInRange>0.5))&(fluxInRange>0.0)) #ErrFin0.25
        print(mask)
        return Values, Errors, mask



"""
#+
# NAME:
#       InerLoopExtract
# PURPOSE:
#       to be used in nonParamFit_ray for parrallelization
#
# CALLING SEQUENCE:
#     IDK why you would want to
#
"""
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
        #print(N)
        featSample = feats + (np.random.normal(0,1,len(feats))*errs)
        sptCodeCutSample = sptCodeCut + (np.random.normal(0,1,len(feats))*SpTErrCut)
        #print(feats)
        #print(featSample)
        mcResults[N,:] = localreg(sptCodeCutSample, featSample, outSPTcode , degree=deg, kernel=rbf.gaussian, radius=rad)

    lowerOut = np.percentile(mcResults,  15.9, axis=0)
    upperOut = np.percentile(mcResults, 100-15.9, axis=0)
    medOut = np.nanmedian(mcResults,axis = 0)

    #self
    #medAndErr = np.array([medOut,lowerOut,upperOut])
    return medOut,lowerOut,upperOut

def spt_coding(spt_in):
	# give a number corresponding to the input SpT
	# the scale is 0 at M0, -1 at K7, -8 at K0 (K8 is counted as M0),  -18 at G0
	if np.size(spt_in) == 1:
		if spt_in[0] == 'M':
			spt_num = float(spt_in[1:])
		elif spt_in[0] == 'K':
			spt_num = float(spt_in[1:])-8.
		elif spt_in[0] == 'G':
			spt_num = float(spt_in[1:])-18.
		elif spt_in[0] == 'F':
			spt_num = float(spt_in[1:])-28.
		elif spt_in[0] == 'A':
			spt_num = float(spt_in[1:])-38.
		elif spt_in[0] == 'B':
			spt_num = float(spt_in[1:])-48.
		elif spt_in[0] == 'L':
			spt_num = float(spt_in[1:])+10.
		elif spt_in[0] == '.':
			spt_num = -99.
		else:
			sys.exit('what?')
		return spt_num
	else:
		spt_num = np.empty(len(spt_in))
		for i,s in enumerate(spt_in):
			if s[0] == 'M':
				spt_num[i] = float(s[1:])
			elif s[0] == 'K':
				spt_num[i] = float(s[1:])-8.
			elif s[0] == 'G':
				spt_num[i] = float(s[1:])-18.
			elif s[0] == 'F':
				spt_num[i] = float(s[1:])-28.
			elif s[0] == 'A':
				spt_num[i] = float(s[1:])-38.
			elif s[0] == 'B':
				spt_num[i] = float(s[1:])-48.
			elif s[0] == 'L':
				spt_num[i] = float(s[1:])+10.
			elif s[0] == '.':
				spt_num[i] = -99.
			else:
				sys.exit('what?')
		return spt_num


def convScodToSpTstring(scod):
	if np.size(scod) == 1:
		if scod<-18 or scod >10:
			print('out of bound')
			return None
		elif scod>=0:
			return 'M'+str(scod)
		elif scod<0 and scod>=-8:
			scodRet = 8+scod
			return 'K'+str(scodRet)
		elif scod<-8 and scod>-18:
			scodRed = 18+scod
			return 'G'+str(scodRed)
		return None
	else:
		spt_out = np.empty(len(scod),dtype = 'U64')
		for s,i in enumerate(scod):

			if i<-18 or i >10:
				print('out of bound')
				spt_out[s] = 'NaN'
				#return None
			elif i>=0:
				#return 'M'+str(i)
				spt_out[s] = 'M'+"%.1f" % (i)
			elif i<0 and i>=-8:
				iRet = 8+i
				spt_out[s] = 'K'+"%.1f" % (iRet)
			elif i<-8 and i>-18:
				iRed = 18+i
				spt_out[s] = 'G'+"%.1f" % (iRed)
			else: spt_out[s] ='NaN'
		return spt_out

"""
# FOR PLOTTING
ticks =  np.arange(-22,10,1)
ticklabels = np.array(['','','F8','','G0','','','','','G5','','','','','K0','','','','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
pl.xticks(ticks,ticklabels)
"""


"""
#+
# NAME:
#       readMixClassIII
# PURPOSE:
#       Provide the flux ratio for a given amount of extinction according to the (cardelli+ 1989) extinction law
# CALLING SEQUENCE:
#     ratio = cardelli_extinction wave,Av,Rv=3.1
#
# INPUTS:
#       wave = array with the wavelengths to compute the relation in angstrom
#       Av =  the extinction in mag
# Optional inputs:
#       Rv =  parameter wth same notation in (cardelli+ 1989), default is 3.1
#   returns:
#  the flux ratio as a function of wl
#  If you use it to apply a reddening to a spectrum, multiply it for the result of
#  this function, while you should divide by it in the case you want to deredden it.
#
"""

def cardelli_extinction(wave,Av,Rv=3.1):

  ebv = Av/Rv

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



  A_lambda = Av * (a + b/Rv)
  # print A_lambda

  ratio =  10.**(-0.4*A_lambda)


# I substitute zero for all the extreme UV wavelenghts (not covered by the cardelli law)
  good = x > 11
  Ngood = np.count_nonzero(good == True)
  if Ngood > 0:
  	ratio[good]=0.

# I extrapolate linearly the law for Mid-IR wavelenghts (not covered by the cardelli law)
# Right now it does not extrapolate outside the validity range --- TO BE DONE
  lasttwo= (x > 0.3)
  # lasttwosort=reverse(sort(lasttwo))
  # xlasttwosort=x[lasttwosort]
  # llasttwosort=wave[lasttwosort]
  # ratiolasttwosort=ratio[lasttwosort]
  xlasttwosort=x[lasttwo][::-1]
  llasttwosort=wave[lasttwo][::-1]
  ratiolasttwosort=ratio[lasttwo][::-1]

  mir = x<=0.3
  Nmir = np.count_nonzero(mir == True)
  if Nmir > 0:
    ratio[mir]=np.interp(x[mir],xlasttwosort,ratiolasttwosort)
  bad= ratio > 1
  nbad = np.count_nonzero(bad == True)
  if nbad > 0:
    ratio[bad]=1


  return ratio


"""
#+
# NAME:
#       readMixClassIII
# PURPOSE:
#       modified version of cardelli_extinction that returns:
#           a + b/R_v (see cardelli+ 1989)
#           to be used in error propagation of extinction.
# CALLING SEQUENCE:
#     cardelli_extinction_a_plus_bOverRv
#
# INPUTS:
#       wave = array with the wavelengths to compute the relation
# Optional inputs:
#       Rv =  parameter wth same notation in (cardelli+ 1989)
#   returns:
#      a + b/R_v
#
"""
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


def spec_readspec(file,header='NO OUTPUT HEADER',flag = False):
	wave=[]
	flux=[]
	hdr=[]
	split = file.split('.')
	exten = split[-1]
	if (exten == 'fits') or (exten == 'fit'):
		hdu = pyfits.open(file)
		hdr = hdu[0].header
		if 'crpix1' in hdu[0].header:
			flux = hdu[0].data
			wave = readlambda(flux,hdu,flag)
		else:
			print('!!!	Wavelength keyword not found in FITS HEADER 	!!!')
			return
	else:
		print("Not yet supported!")
		return
		#readcol, file, wave, lambda
	hdu.close()
	if header == 'NO OUTPUT HEADER':
		return wave,flux
	else:
		return wave,flux,hdr


def readlambda(spec, hdu_sp,flag):
	crpix1 = hdu_sp[0].header['crpix1']
#/ value of ref pixel
	crval1 = hdu_sp[0].header['crval1']
#/ delta per pixel
	if 'cd1_1' in hdu_sp[0].header:
		cd1_1 = hdu_sp[0].header['cd1_1']
	#cd1_1 is sometimes called cdelt1.
	else:
		cd1_1 = hdu_sp[0].header['cdelt1']
	if cd1_1 == 0:
		print("NOT WORKING")
		return
	n_lambda = len(spec)
	if flag:
		n_lambda = len(spec[0])
	wave = np.zeros(n_lambda)
	for l  in range(n_lambda):
		wave[l] = (l+1.0-crpix1)*cd1_1+crval1
#Use pixel number starting with 0 if no lambda information is found.
	if (np.min(wave)+np.max(wave) == 0.0):
		print('No lambda information found: used pixel number starting with 0')
		for l  in range(n_lambda):
			wave[l] = l
	return wave
