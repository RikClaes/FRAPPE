import numpy as np
from spec_readspec import spec_readspec
import matplotlib.pyplot as plt
from cardelli_extinction import cardelli_extinction
import spectral_classification.spt_coding as scod
import glob
import numpy as np
from localreg import *
from scipy import interpolate as intp
from readcol_py3 import *
import os
import ray


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
        SptCodes = scod.spt_coding(SpTlist)
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



    def extractFeaturesXS_ray_LOG(self,DirSpec,nameList,SpTlist,usedFeatures,AvErr = 0.2,WLnorm = 751,wlNormHalfWidth = 0.5,SpTErr = []):
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
        SptCodes = scod.spt_coding(SpTlist)
        ticklabels = np.array(['G4','','G6','','','G9','K0','','','K3','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
        ticks =  np.arange(-14,11,1)
        ray.shutdown()
        ray.init()

        pool = ray.get([InerLoopExtract_log.remote(nameList[i],i,DirSpec,usedFeatures,AvErr,WLnorm,wlNormHalfWidth,SpTErr)for i in range(len(nameList))])

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







    def setExtarctedFeatures(self,usedFeatures, Values, Errors, SptCodes,mask):
        self.usedFeatures = usedFeatures
        self.extractedFeatureValues = Values
        self.extractedFeatureErrors = Errors
        self.extractedFeatureSptCodes = SptCodes
        self.Mask=mask

    def getExtractedFeatures(self):
        usedF = self.usedFeatures
        return usedF, self.extractedFeatureSptCodes, self.extractedFeatureValues,self.extractedFeatureErrors


    def nonParamFit(self,nrOfPoints = 200,mcSamples =1000,rad =3.5,deg = 2,outFile =None):

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

    def nonParamFit_ray_LOG(self,nrOfPoints = 200,mcSamples =1000,rad =3.5,deg = 2,outFile =None):

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
        pool = ray.get([InerLoopFit_log.remote(i,featureMatrix,errorMatrix,sptCode,SpTErr,mask,mcSamples,outSPTcode,rad,deg)for i in range(len(features))])

        pool1 = np.array(pool)
        print("HERE")
        print(pool1.shape)
        medOut,lowerOut,upperOut  = pool1[:,0,:], pool1[:,1,:], pool1[:,2,:]

        medAndErr = np.array([medOut,lowerOut,upperOut])

        if outFile != None:
            np.savez(outFile, medAndErr =medAndErr,sptCode = outSPTcode,usedFeatures = features,normalWL = self.normWL)
        self.setInterpFeat(outSPTcode,medOut,lowerOut,upperOut)




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
            np.savez(outFile, medAndErr =medAndErr,sptCode = outSPTcode,usedFeatures = features,normalWL = self.normWL)
        self.setInterpFeat(outSPTcode,medOut, lowerOut, upperOut)

        #return ValuesOut,outSPTcode, medOut, lowerOut, upperOut
            #preform Monte carlo simulation to compute the errors


    ######
    #TODO: hightlight points not used!
    ######
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
            axs[1].scatter(self.extractedFeatureSptCodes[mask[i]],(self.extractedFeatureValues[i][mask[i]]-interp)/self.extractedFeatureErrors[i][mask[i]],c='k')
            interp = np.interp(self.extractedFeatureSptCodes[~mask[i]],self.sptCode,self.medInterp[i])
            axs[1].scatter(self.extractedFeatureSptCodes[~mask[i]],(self.extractedFeatureValues[i][~mask[i]]-interp)/self.extractedFeatureErrors[i][~mask[i]],c='r')
            axs[1].plot(self.sptCode,np.zeros(len(self.sptCode)))
            axs[1].set_xlabel('SpT code')
            axs[1].set_ylabel(r'$(f_{spec.} - f_{fit.})/\sigma$')
            #axs[1].set_ylim(-0.6,0.6)
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

    def plotAllInterpIndividualy_FitResiduals_LOG(self, outdir ='./',logScale = False):
        if logScale:
            for i in range(len(self.medInterp)):
                fig, axs = plt.subplots(2, 1,figsize=(11,6),sharex='col', gridspec_kw={'height_ratios': [3, 1]})
                axs[0].set_title('wl range: '+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+' nm')
                axs[0].plot(self.sptCode,np.log10(self.medInterp[i]))
                mask= self.Mask
                extractedFeat = self.extractedFeatureValues
                extractedFeatureErr = self.extractedFeatureErrors
                axs[0].errorbar(self.extractedFeatureSptCodes[mask[i]],extractedFeat[i][mask[i]],extractedFeatureErr[i][mask[i]],xerr = self.SpTErr[mask[i]],c='k',linestyle='',marker ='o', label = str(self.usedFeatures[i]))
                axs[0].errorbar(self.extractedFeatureSptCodes[~mask[i]],extractedFeat[i][~mask[i]],extractedFeatureErr[i][~mask[i]],xerr = self.SpTErr[~mask[i]],c= 'r',alpha=0.5,linestyle='',marker ='o', label = str(self.usedFeatures[i]))

                #axs[0].plot(outSPTcode,med[i])
                axs[0].fill_between(self.sptCode,np.log10(self.lowerErrInterp[i]), np.log10(self.upperErrInterp[i]),alpha = 0.4)
                axs[0].set_xlabel('SpT code')
                axs[0].set_ylabel('log(f(range)/f'+str(self.normWL[0])+')')
                #if logScale:
                #    axs[0].set_yscale('log')
                #residuals
                extractedFeat = 10**self.extractedFeatureValues
                extractedFeatureErr = self.extractedFeatureErrors*np.log(10)*extractedFeat
                interp = np.interp(self.extractedFeatureSptCodes[mask[i]],self.sptCode,self.medInterp[i])
                interpErr = np.interp(self.extractedFeatureSptCodes[mask[i]],self.sptCode,( self.upperErrInterp[i] - self.lowerErrInterp[i])/2)
                axs[1].scatter(self.extractedFeatureSptCodes[mask[i]],(extractedFeat[i][mask[i]]-interp)/interpErr,c='k')
                interp = np.interp(self.extractedFeatureSptCodes[~mask[i]],self.sptCode,self.medInterp[i])
                interpErr = np.interp(self.extractedFeatureSptCodes[~mask[i]],self.sptCode,( self.upperErrInterp[i] - self.lowerErrInterp[i])/2)
                axs[1].scatter(self.extractedFeatureSptCodes[~mask[i]],(extractedFeat[i][~mask[i]]-interp)/interpErr,c='r')
                axs[1].plot(self.sptCode,np.zeros(len(self.sptCode)))
                axs[1].set_xlabel('SpT code')
                axs[1].set_ylabel(r'$(f_{spec.} - f_{fit.})/\sigma$')
                #axs[1].set_ylim(-0.6,0.6)
                #ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5',''])
                ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
                ticks =  np.arange(-10,11,1)
                #plt.xlim(-10,6)
                plt.xticks(ticks,ticklabels)
                plt.savefig(outdir+'wl_range:'+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+'nm_LOG.png')
        else:
            for i in range(len(self.medInterp)):
                fig, axs = plt.subplots(2, 1,figsize=(11,6),sharex='col', gridspec_kw={'height_ratios': [3, 1]})
                axs[0].set_title('wl range: '+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+' nm')
                axs[0].plot(self.sptCode,self.medInterp[i])
                mask= self.Mask
                extractedFeat = 10**self.extractedFeatureValues
                extractedFeatureErr = self.extractedFeatureErrors*np.log(10)*extractedFeat
                axs[0].errorbar(self.extractedFeatureSptCodes[mask[i]],extractedFeat[i][mask[i]],extractedFeatureErr[i][mask[i]],xerr = self.SpTErr[mask[i]],c='k',linestyle='',marker ='o', label = str(self.usedFeatures[i]))
                axs[0].errorbar(self.extractedFeatureSptCodes[~mask[i]],extractedFeat[i][~mask[i]],extractedFeatureErr[i][~mask[i]],xerr = self.SpTErr[~mask[i]],c= 'r',alpha=0.5,linestyle='',marker ='o', label = str(self.usedFeatures[i]))

                #axs[0].plot(outSPTcode,med[i])
                axs[0].fill_between(self.sptCode,self.lowerErrInterp[i], self.upperErrInterp[i],alpha = 0.4)
                axs[0].set_xlabel('SpT code')
                axs[0].set_ylabel('f(range)/f'+str(self.normWL[0]))
                #residuals
                interp = np.interp(self.extractedFeatureSptCodes[mask[i]],self.sptCode,self.medInterp[i])
                interpErr = np.interp(self.extractedFeatureSptCodes[mask[i]],self.sptCode,( self.upperErrInterp[i] - self.lowerErrInterp[i])/2)
                axs[1].scatter(self.extractedFeatureSptCodes[mask[i]],(extractedFeat[i][mask[i]]-interp)/interpErr,c='k')
                interp = np.interp(self.extractedFeatureSptCodes[~mask[i]],self.sptCode,self.medInterp[i])
                interpErr = np.interp(self.extractedFeatureSptCodes[~mask[i]],self.sptCode,( self.upperErrInterp[i] - self.lowerErrInterp[i])/2)
                axs[1].scatter(self.extractedFeatureSptCodes[~mask[i]],(extractedFeat[i][~mask[i]]-interp)/interpErr,c='r')
                axs[1].plot(self.sptCode,np.zeros(len(self.sptCode)))
                axs[1].set_xlabel('SpT code')
                axs[1].set_ylabel(r'$(f_{spec.} - f_{fit.})/\sigma$')
                #axs[1].set_ylim(-0.6,0.6)
                #ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5',''])
                ticklabels = np.array(['','G9','','K1','','K3','','K5','','K7','','M1','','M3','','M5','','M7','','M9',''])
                ticks =  np.arange(-10,11,1)
                #plt.xlim(-10,6)
                plt.xticks(ticks,ticklabels)
                plt.savefig(outdir+'wl_range:'+str(self.usedFeatures[i][0])+'-'+str(self.usedFeatures[i][1])+'nm.png')
            plt.close()

    def plotAllInterpIndividualy_FitResiduals(self, outdir ='./',logScale = False):
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

    def getUsedNormWl(self):
        return self.normWL


    def readInterpFeat(self,dirInterpfeat):
        npz = np.load(dirInterpfeat)
        featuresandErrs =npz['medAndErr']
        self.sptCode = npz['sptCode']
        self.usedFeatures = npz['usedFeatures']
        self.medInterp = featuresandErrs[0]
        self.lowerErrInterp = featuresandErrs[1]
        self.upperErrInterp = featuresandErrs[2]
        self.normWL = npz['normalWL']

def readMixClassIII(min_chi_sq_cl3,PATH_CLASSIII,wlNorm =731,average = False):
    clsIIIinfo = np.genfromtxt(PATH_CLASSIII+'summary_classIII_final.txt',usecols=(0,2),skip_header=1,dtype=[('Name','U64'),('Spt','U4')])
    name_cl3 = clsIIIinfo['Name']
    SpT_cl3 = clsIIIinfo['Spt']
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

def readMixClassIII_withSpT(min_chi_sq_cl3,PATH_CLASSIII,wlNorm =731,average = False):
    #name_cl3,SpT_cl3 = readcol_py3(PATH_CLASSIII+'summary_classIII_final.txt',2,format='A,X,A,X',skipline=1)
    clsIIIinfo = np.genfromtxt(PATH_CLASSIII+'summary_classIII_final.txt',usecols=(0,2),skip_header=1,dtype=[('Name','U64'),('Spt','U4')])
    name_cl3 = clsIIIinfo['Name']
    SpT_cl3 = clsIIIinfo['Spt']
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

@ray.remote#(num_returns=2)
def InerLoopExtract_log(name,i,DirSpec,usedFeatures,AvErr,WLnorm,wlNormHalfWidth,SpTErr):
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
        fwlnorm = np.log10(np.nanmedian(fl[(wl<=WLnorm+wlNormHalfWidth)&(wl>=WLnorm-wlNormHalfWidth)]))
        #fwlnormErr_noise = np.nanstd(fl[(wl<=WLnorm+wlNormHalfWidth)&(wl>=WLnorm-wlNormHalfWidth)])
        #####
        #new!!
        #####
        #fwlnormErr_fCalib = fwlnorm/20
        #fwlnormErr = np.sqrt((fwlnormErr_noise**2) + (fwlnormErr_fCalib**2))
        fwlnormErr = np.abs(np.nanstd(fl[(wl<=WLnorm+wlNormHalfWidth)&(wl>=WLnorm-wlNormHalfWidth)])/(fwlnorm*np.log(10)))
        for j in range(len(usedFeatures)):

            #print(features[j,0])

            fluxNotScaledInRange  = np.log10( np.nanmedian(fl[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])]))
            fluxInRange = fluxNotScaledInRange-fwlnorm
            logFluxInRange =  fluxInRange
            #ErrNotScaledInRange_noise = np.nanstd(fl[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])])
            #ErrNotScaledInRange_fCalib = fluxNotScaledInRange/20
            #ErrNotScaledInRange = np.sqrt((ErrNotScaledInRange_noise**2)+ (ErrNotScaledInRange_fCalib**2))
            ErrNotScaledInRange = np.abs(np.nanstd(fl[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])])/(fluxNotScaledInRange*np.log10(10)))
            ErrFluxInRange = np.sqrt((ErrNotScaledInRange)**2 + (fwlnormErr)**2)
            #ErrFluxInRange = np.nanstd(fl[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])])


            Values[j]  = fluxInRange
            wlReddening = 10*(usedFeatures[j,0] + usedFeatures[j,1])/2
            CardelliCte = cardelli_extinction_a_plus_bOverRv(np.array([wlReddening]),Rv=3.1) - cardelli_extinction_a_plus_bOverRv(np.array([WLnorm]),Rv=3.1)
            errCardelliRatio = np.abs(-0.4 * CardelliCte*AvErr)
            #errCardelliRatio = np.abs(-0.4*np.log(10) * CardelliCte*AvErr)
            #print(1 - (cardelli_extinction(np.array([wlReddening]),AvErr)))
            #print(ErrFluxInRange/fluxInRange)
            term1 = (ErrFluxInRange) **2
            #print(term1)
            term2 = (errCardelliRatio)**2
            errFlDerred = np.sqrt(term1+ term2) #np.sqrt((ErrFluxInRange/fluxInRange)**2
            #print(errCardelliRatio)
            Errors[j] =   errFlDerred
            #mask[j] = ((np.nanmedian((fl[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])]))/np.nanstd((fl[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])]))) >=0.5)
            mask[j] = ((np.nanmedian((fl[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])]))/np.nanstd((fl[(wl>usedFeatures[j,0])&(wl<usedFeatures[j,1])]))) >=0.0)
        print(mask)
        return Values, Errors, mask

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


@ray.remote
def InerLoopFit_log(i,featureMatrix,errorMatrix,sptCode,SpTErr,mask,mcSamples,outSPTcode,rad,deg):
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

    lowerOut = 10**np.percentile(mcResults,  15.9, axis=0)
    upperOut = 10**np.percentile(mcResults, 100-15.9, axis=0)
    medOut = 10**np.nanmedian(mcResults,axis = 0)

    #self
    #medAndErr = np.array([medOut,lowerOut,upperOut])
    return medOut,lowerOut,upperOut

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
