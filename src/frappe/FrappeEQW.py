import numpy as np
import matplotlib as mp
import pylab as pl

import time
import sys
import string
import os

PATH = os.path.dirname(os.path.realpath(__file__))
sys.path = [PATH+'/FrappeHelper/'] + sys.path

from spec_readspec import *
from readcol_py3 import *
from int_tabulated import *
from scipy.io.idl import readsav
from cardelli_extinction import *
from eqw_auto import *
import csv
from spectrum_resample import *
from isochrone_alpha import *
from macc_calc import *
import spt_coding as scod
from astropy import units as u
from spec_phase3 import *
import ray

import PhotFeatures_Ray as pf
import EqwVSspt as eqw

ray.shutdown()

PATH_SLAB = PATH+'/models_grid/slab_models_grid/'
PATH_SLAB_RESAMPLED_SAV = PATH+'/models_grid/Slab_SAV_TWA7only/'
gridFile = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_500iter_rad2.5_deg2_LongWL_improved/interp.npz'
eqwFile = PATH+'/models_grid/TestEQWExtract_MoreLines_Booth_deg1_rad2/EQWmore.npz'

from utils_fitter_py3 import *


@ray.remote


def main_process(fit_data,cl3_spt):

	classIIIreadIn            = fit_data['classIIIreadIn']
	wl_UVB                    = fit_data['wl_UVB']
	wl_VIS                    = fit_data['wl_VIS']
	fl_UVB_in                 = fit_data['fl_UVB_in']
	fl_VIS_in                 = fit_data['fl_VIS_in']
	usedFeatures              = fit_data['usedFeatures']
	normWLandWidth            = fit_data['normWLandWidth']
	Av_list                   = fit_data['Av_list']
	cl3_in_toSelectModel      = fit_data['cl3_in_toSelectModel']
	wlObs                     = fit_data['wlObs']
	flObs                     = fit_data['flObs']
	Rv                        = fit_data['Rv']
	EQWreadIn                 = fit_data['EQWreadIn']
	usedEQWlines              = fit_data['usedEQWlines']
	classIIIFeatPlottingVIS   = fit_data['classIIIFeatPlottingVIS']
	classIIIFeatPlottingUVB   = fit_data['classIIIFeatPlottingUVB']
	usedFeaturesPlottingVIS   = fit_data['usedFeaturesPlottingVIS']
	usedFeaturesPlottingUVB   = fit_data['usedFeaturesPlottingUVB']
	eqwValObs                 = fit_data['eqwValObs']
	eqw_errObs                = fit_data['eqw_errObs']

	features,errors = classIIIreadIn.getFeatsAtSpt_symetricErr(cl3_spt)

	eqwInterp,eqwInterpErr = EQWreadIn.getFeatsAtSpt_symetricErr(cl3_spt)
	usedEQWlines = usedEQWlines

	featuresPlotVIS,errorsPlotVIS = classIIIFeatPlottingVIS.getFeatsAtSpt_symetricErr(cl3_spt)
	featuresPlotUVB,errorsPlotUVB = classIIIFeatPlottingUVB.getFeatsAtSpt_symetricErr(cl3_spt)

	usedFeaturesPlotting = np.concatenate((usedFeaturesPlottingUVB,usedFeaturesPlottingVIS))
	featuresPlot = np.concatenate((featuresPlotUVB,featuresPlotVIS))
	errorsPlot = np.concatenate((errorsPlotUVB,errorsPlotVIS))
	wlFeatPlot = (usedFeaturesPlotting[:,0]+usedFeaturesPlotting[:,1])/2

	T_slab = ['5000','5500','6000','6500','7000','7500','7750','8000','8250','8500','8750','9000','9250','9500','9750',\
	'10000','10500']#,'11000']
	Ne_slab = ['1e+11','1e+12','1e+13','3e+13','5e+13','7e+13','1e+14','5e+14','1e+15','1e+16']
	tau_slab = ['0.01', '0.05','0.1','0.3','0.5','0.75','1','3','5']

	chi_sq = {}
	chi_sq_max = 1.e15
	H_fin = {}
	K_fin = {}
	Chiterm1List = {}
	Chiterm2List = {}
	file_read = 0
	p1 = None

	wlObs = wlObs
	flObs = flObs
	Rv = Rv

	obs_cont,stddev_cont = np.zeros(len(usedFeatures)),np.zeros(len(usedFeatures))

	for i in range(len(usedFeatures)):
		obs_cont[i],stddev_cont[i] = compute_flux_inRange(wlObs,flObs,usedFeatures[i,0],usedFeatures[i,1])
	normWL, normWlHalfWidth = normWLandWidth[0], normWLandWidth[1]
	obs_cont_CLIIIScaling, stddev_cont_CLIIIScaling = compute_flux_at_wl_std(wlObs,flObs,normWL,interval=normWlHalfWidth*2)
	obs_cont_360, stddev_cont_360 = compute_flux_at_wl_std(wlObs,flObs,355,interval=6)
	for Av_iter in Av_list:

		wl_dered = (usedFeatures[:,0]+usedFeatures[:,1])/2
		obs_cont_dered = obs_cont/cardelli_extinction(wl_dered*10.,Av_iter, Rv=Rv)
		stddev_cont_dered = stddev_cont/cardelli_extinction(wl_dered*10.,Av_iter, Rv=Rv)

		obs_cont_CLIIIScaling_dered = obs_cont_CLIIIScaling/cardelli_extinction(np.array([normWL*10.]),Av_iter, Rv=Rv)
		stddev_cont_CLIIIScaling_dered = stddev_cont_CLIIIScaling/cardelli_extinction(np.array([normWL*10.]),Av_iter, Rv=Rv)

		obs_cont_360_dered = obs_cont_360/cardelli_extinction(np.array([355*10.]),Av_iter, Rv=Rv)
		if 355 not in wl_dered:
			print('the feature to scale the acc slab model is not included!!')
		cl3_cont_360 = features[wl_dered == 355][0]
		cl3_stddev_cont_360 = errors[wl_dered == 355][0]

		for T_iter in T_slab:
			for Ne_iter in Ne_slab:
				for tau_iter in tau_slab:

					wl_slab_UVB,fl_slab_UVB = read_slab_sav_RC(T_iter,Ne_iter,tau_iter,cl3_in_toSelectModel,'UVB',PATH_SLAB_RESAMPLED_SAV)
					wl_slab_VIS,fl_slab_VIS = read_slab_sav_RC(T_iter,Ne_iter,tau_iter,cl3_in_toSelectModel,'VIS',PATH_SLAB_RESAMPLED_SAV)
					wl_slab = np.concatenate([wl_slab_UVB[wl_slab_UVB<550],wl_slab_VIS[wl_slab_VIS>550]])#,wn10[wn10>1020]])
					fl_slab = np.concatenate([fl_slab_UVB[wl_slab_UVB<550],fl_slab_VIS[wl_slab_VIS>550]])#,fn10[wn10>1020]])

					slab_cont_360 = compute_cont_360_nostd(wl_slab_UVB,fl_slab_UVB)
					slab_cont_CLIIIScaling = compute_flux_at_wl_nostd(wl_slab_VIS,fl_slab_VIS,normWL,interval=normWlHalfWidth*2)

					K_try = float(np.squeeze(obs_cont_CLIIIScaling_dered*1)) ###1 is the flux at the normalization wl in the class III interpolation

					H_try = float(np.squeeze((obs_cont_360_dered-K_try*cl3_cont_360)/slab_cont_360))

					K_old = K_try
					K_iter = float(np.squeeze(obs_cont_CLIIIScaling_dered-slab_cont_CLIIIScaling*H_try))#/cl3_cont_751
					H_iter = H_try
					if K_iter < 0 or H_try < 0:
						print( 'NOT POSSIBLE')
						if H_try <0:
							print('initial H already negative' )
						if K_iter<0:
							print('initial K already negative' )
						continue
					max_iter = 100 # maximum number of possible iterations
					min_discr = 0.005e-14 # minimum difference between two iterate values of K. when this value is reached, stop iteration
					iterations = 0
					while abs(K_iter-K_old)/np.mean([K_iter,K_old]) > min_discr and iterations <= max_iter:
						K_old = K_iter
						H_iter = float(np.squeeze((obs_cont_360_dered-K_iter*cl3_cont_360)/slab_cont_360))
						K_iter = float(np.squeeze((obs_cont_CLIIIScaling_dered-slab_cont_CLIIIScaling*H_iter)/1)) ###1 is the flux at the normalization wl in the class III interp grid
						iterations+=1
					K = K_iter
					H = H_iter
					if K < 0 or H <= 0:
						print( 'NOT POSSIBLE')
						continue

					fit_cont = np.zeros(len(usedFeatures))
					for i in range(len(usedFeatures)):
						fit_cont[i] = K*features[i]+H*compute_flux_inRange(wl_slab,fl_slab,usedFeatures[i,0],usedFeatures[i,1])[0]
					stdTerm1 = 0#(K*errors)**2
					stdTerm2 = (stddev_cont_dered**2)
					fit_std = np.sqrt(stdTerm1 +stdTerm2)

					fit_eqw = np.zeros(len(usedEQWlines))
					fit_eqw_err = np.zeros(len(usedEQWlines))
					for i in range(len(usedEQWlines)):
						fluxSlabAtWl = H*compute_flux_at_wl_std(wl_slab,fl_slab,usedEQWlines[i,0],interval=usedEQWlines[i,1])[0]
						contAroundLine = compute_flux_at_wl_std(wlFeatPlot,featuresPlot,usedEQWlines[i,0],interval=2*usedEQWlines[i,1])[0]
						fluxclIIIAtWl = K*contAroundLine/2
						fit_eqw[i] = eqwInterp[i]*fluxclIIIAtWl/(fluxSlabAtWl+fluxclIIIAtWl)
						fit_eqw_err[i] = 0#eqwInterpErr[i]*fluxclIIIAtWl/(fluxSlabAtWl+fluxclIIIAtWl)
					eqw_std = np.sqrt((eqw_errObs**2) +(fit_eqw_err**2))
					chiterm1 = np.sum(((fit_cont - obs_cont_dered) / fit_std)**2)
					chiterm2 = np.nansum(((fit_eqw - eqwValObs) / eqw_std)**2)
					chi_sq_temp = chiterm1  + chiterm2

					key = str(cl3_spt)+'/'+str(Av_iter)+'/'+T_iter+'/'+Ne_iter+'/'+tau_iter
					chi_sq[key] = chi_sq_temp

					H_fin[key] = H
					K_fin[key] = K
					Chiterm1List[key] = chiterm1
					Chiterm2List[key] = chiterm2
	return [chi_sq, H_fin, K_fin,Chiterm1List,Chiterm2List]

'''				############################
'''

interactive_mode_on = 'YES'
max_wl_uvb = 552.
min_wl_vis = 552.
max_wl_vis = 1019.
cl3_in = [3,3.25,3.5]

dist_pc = 400. #[pc]
Av = None #[mag] - If None, then it can be varied
Rv = 3.1

plot_smooth = True

perAA = False
fitsTab = False


class Fit():

	def __init__(self,f_uvb = None,f_vis = None,f_nir = None,obj_in ='Unknown',dist=None ,dirOut = None,Posterior = False,**kwargs):

		max_wl_uvb = 552.
		min_wl_vis = 552.
		max_wl_vis = 1019.

		self.filename_UVB = os.path.abspath(f_uvb) if f_uvb is not None else None
		self.filename_VIS = os.path.abspath(f_vis) if f_vis is not None else None
		self.filename_NIR = os.path.abspath(f_nir) if f_nir is not None else None
		if dist == None:
			print( sys.argv[0],': dist=',dist)
			sys.exit(1)
		else:
			self.dist_pc = dist
		fitsTab = False
		self.perAA = False
		Rv = 3.1
		for option, value in kwargs.items():
			if option == 'spt':
				cl3_in = value  # Dec, 2nd 2014 - in this way I can give as input a sequence of names and becomes a list
			elif option == 'max_u' or option == '--max_wl_uvb':
				self.max_wl_uvb = value
			elif option == 'min_v' or option == '--min_wl_vis':
				self.min_wl_vis = value
			elif option == 'max_v' or option == '--max_wl_vis':
				self.max_wl_vis = value
			elif option == 'Av' or option == '--Av_fix':
				Av = value # Dec, 4th 2014
			elif option == 'Rv' or option == '--reddening_law':
				Rv = value
			elif option == 'perAA':
				self.perAA = value
			elif option == 'fitsTab':
				self.fitsTab = value
			else:
				print( sys.argv[0],': option=',option)
				sys.exit(1)

		self.Rv = Rv

		classIIIreadIn = pf.classIII(gridFile)
		self.classIIIreadIn =classIIIreadIn
		usedFeatures = classIIIreadIn.getUsedInterpFeat()
		self.usedFeatures = usedFeatures
		normWLandWidth = classIIIreadIn.getUsedNormWl()
		self.normWLandWidth = normWLandWidth
		
		cl3_in_toSelectModel = 'TWA7'

		now = time.localtime()[0:6]
		if dirOut != None:
			path = dirOut
		else:
			path = os.getcwd()
		self.path = path
		self.PATH_OUT = path+obj_in+'_%4d-%02d-%02d_%02d.%02d.%02d' % now
		print("The will be stored in:")
		print(self.PATH_OUT)
		os.mkdir(self.PATH_OUT)
		self.PATH_OUT = self.PATH_OUT+'/'

		if self.fitsTab == False:
			wl_UVB,fl_UVB_in,hdr_UVB=spec_readspec(self.filename_UVB, 'hdr')	#the 'hdr' string is there to say that I want to save the header
		else:
			wl_UVB,fl_UVB_in,hdr_UVB=readspec_phase3(self.filename_UVB,hdr_out='y')

		self.ind_uvb = (wl_UVB <= max_wl_uvb) #select only the part of the spectrum that is nice(\lambda<550 nm in the UVB)
		if self.fitsTab == False:
			wl_VIS,fl_VIS_in,hdr_VIS=spec_readspec(self.filename_VIS, 'hdr')	#the 'hdr' string is there to say that I want to save the header
		else:
			wl_VIS,fl_VIS_in,hdr_VIS=readspec_phase3(self.filename_VIS,hdr_out='Y')
		self.ind_vis = (wl_VIS >= min_wl_vis) #& (wave_VIS <=1024) #select only the part of the spectrum that is nice(\lambda>550 nm in the VIS)
		wlObs = np.concatenate([wl_UVB[wl_UVB<550],wl_VIS[wl_VIS>550]])#,wn10[wn10>1020]])
		flObs = np.concatenate([fl_UVB_in[wl_UVB<550],fl_VIS_in[wl_VIS>550]])#,fn10[wn10>1020]])

		if self.perAA == True:
			fl_UVB_in = fl_UVB_in*10
			fl_VIS_in = fl_VIS_in*10
		self.wl_UVB = wl_UVB
		self.fl_UVB_in = fl_UVB_in
		self.wl_VIS = wl_VIS
		self.fl_VIS_in = fl_VIS_in
		print(obj_in)
		if obj_in != None:
			self.obj_in = obj_in
		else:
			self.obj_in = hdr_UVB['OBJECT'].replace(" ", "")

		if Av != None:
			print( Av)
			Av_list = np.array(Av,dtype=np.float32)
			self.Av_list = Av_list
		else:
			Av_list = np.concatenate((np.linspace(0,1.5,16),np.linspace(1.5,3,4)))	#[  0. ,   0.1,   0.2,   0.3,   0.4,   0.5,   0.6,   0.7,   0.8, 0.9,   1. ,   1.1,   1.2,   1.3,   1.4,   1.5,   1.6,   1.7,1.8,   1.9,   2. ,   2.1,   2.2,   2.3,   2.4,   2.5,   2.6,2.7,   2.8,   2.9,   3. ,   3. ,   4. ,   5. ,   6. ,   7. ,8. ,   9. ,  10. ]

		if cl3_in != None:
			print('You have given a selection of spectral types')
			cl3_in_list = np.array(cl3_in,dtype=np.float32)#cl3_in
		else:
			print('No spectral types given, will try entire range')
			cl3_in_list = np.array(range(-10,9))

		self.cl3_in_list = cl3_in_list

		EQWreadIn = eqw.EQWvsSpT(eqwFile)
		usedEQWlines = EQWreadIn.getUsedInterpFeat()
		cl3_spt_max = np.max(cl3_in_list)
		cl3_spt_min = np.min(cl3_in_list)
		eqwInterpMax,eqwInterpErrMax = EQWreadIn.getFeatsAtSpt_symetricErr(cl3_spt_max)
		eqwInterpMin,eqwInterpErrMin = EQWreadIn.getFeatsAtSpt_symetricErr(cl3_spt_min)

		eqwValObs = np.zeros(len(usedEQWlines))
		eqw_errObs = np.zeros(len(usedEQWlines))
		for i in range(len(usedEQWlines)):
			eqwValObs[i], eqw_errObs[i] = eqw.computeEW(wlObs,flObs,usedEQWlines[i,0],size_cont=usedEQWlines[i,1],dire = self.PATH_OUT+str(obj_in)+'_'+str(usedEQWlines[i,0])+'nm.png')
			if eqwValObs[i] < 2 * np.abs(eqw_errObs[i]):
				eqwValObs[i] = np.nan
				print('the absorption line at'+ str(usedEQWlines[i,0])+ 'nm is not detected and will therefore not be used in the best fit determination')
			if np.isnan(eqwInterpMax[i])  or np.isnan(eqwInterpMin[i]):
				eqwValObs[i] = np.nan
				print('the absorption line at'+ str(usedEQWlines[i,0])+ 'nm does not have an eqw measurement at some of the considered SpTs and will therefore not be used in the best fit determination')
		eqwValObs = eqwValObs
		eqw_errObs = eqw_errObs

		gridFilePlot = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeUVB/interp.npz'
		classIIIFeatPlottingUVB = pf.classIII(gridFilePlot)
		usedFeaturesPlottingUVB = classIIIFeatPlottingUVB.getUsedInterpFeat()

		gridFilePlotVIS = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeVIS/interp.npz'
		classIIIFeatPlottingVIS = pf.classIII(gridFilePlotVIS)
		usedFeaturesPlottingVIS = classIIIFeatPlottingVIS.getUsedInterpFeat()

		time_init = time.time()
		ray.init(runtime_env={"working_dir":PATH+'/FrappeHelper/'})
		fit_data_ref = ray.put({
			'classIIIreadIn':           classIIIreadIn,
			'wl_UVB':                   wl_UVB,
			'wl_VIS':                   wl_VIS,
			'fl_UVB_in':                fl_UVB_in,
			'fl_VIS_in':                fl_VIS_in,
			'usedFeatures':             usedFeatures,
			'normWLandWidth':           normWLandWidth,
			'Av_list':                  Av_list,
			'cl3_in_toSelectModel':     cl3_in_toSelectModel,
			'wlObs':                    wlObs,
			'flObs':                    flObs,
			'Rv':                       self.Rv,
			'EQWreadIn':                EQWreadIn,
			'usedEQWlines':             usedEQWlines,
			'classIIIFeatPlottingVIS':  classIIIFeatPlottingVIS,
			'classIIIFeatPlottingUVB':  classIIIFeatPlottingUVB,
			'usedFeaturesPlottingVIS':  usedFeaturesPlottingVIS,
			'usedFeaturesPlottingUVB':  usedFeaturesPlottingUVB,
			'eqwValObs':                eqwValObs,
			'eqw_errObs':               eqw_errObs,
		})
		pool_outputs1 = ray.get([main_process.remote(fit_data_ref,cl3_in_list[i])for i in range(len(cl3_in_list))])

		print( 'Execution time:', time.time() - time_init, " seconds")

		dim = len(pool_outputs1)
		chi_sq = pool_outputs1[0][0]
		H_fin = pool_outputs1[0][1]
		K_fin = pool_outputs1[0][2]
		Chiterm1List = pool_outputs1[0][3]
		Chiterm2List = pool_outputs1[0][4]

		for i in range(dim):

			chi_sq.update(pool_outputs1[i][0])
			H_fin.update(pool_outputs1[i][1])
			K_fin.update(pool_outputs1[i][2])
			Chiterm1List.update(pool_outputs1[i][3])
			Chiterm2List.update(pool_outputs1[i][4])

		self.chi_sq = chi_sq
		self.H_fin = H_fin
		self.K_fin = K_fin
		self.Chiterm1List = Chiterm1List
		self.Chiterm2List = Chiterm2List

		best_chi_sq_val = np.sort(np.array(list(chi_sq.values())))[:40]
		best_chi_sq = {}

		for k in chi_sq:
			if chi_sq[k] in best_chi_sq_val:
				best_chi_sq[k] = chi_sq[k]
		self.best_chi_sq =best_chi_sq
		self.min_chi_sq = min(chi_sq.values())
		for k in best_chi_sq:
			if best_chi_sq[k] == self.min_chi_sq:
				self.min_chi_sq_cl3 = float(k.split('/')[0])
				self.min_chi_sq_Av = float(k.split('/')[1])
				self.min_chi_sq_T = k.split('/')[2]
				self.min_chi_sq_Ne = k.split('/')[3]
				self.min_chi_sq_tau = k.split('/')[4]
				self.min_chi_sq_H = H_fin[k]
				self.min_chi_sq_K = K_fin[k]
		ray.shutdown()
		print( '    ')
		print('    ')
		print( '---------------')
		print( '---------------')
		print( 'BEST CHI SQUARED: ', self.min_chi_sq)
		print( ' ')
		print( 'PARAMETERS OF BEST FIT: ClassIII =', self.min_chi_sq_cl3,', Av=', self.min_chi_sq_Av, ', T=', self.min_chi_sq_T, ', Ne=',self.min_chi_sq_Ne,', tau=',self.min_chi_sq_tau)
		print( 'NORMALIZATION CONSTANTS FOR THE BEST FIT: H=', self.min_chi_sq_H, ', K=', self.min_chi_sq_K)
		print( '---------------')
		print( '---------------')

	def writeCSV(self):
		""" CSV FILE FOR CHI2 """
		w = csv.writer(open(self.PATH_OUT+'%s_clIII_%s_chi2.csv'% (self.obj_in,self.min_chi_sq_cl3), 'w'))
		for key, val in self.chi_sq.items():
			w.writerow([key, val, self.H_fin[key],self.K_fin[key]])
		if len(list(csv.DictReader(open(self.PATH_OUT+'%s_clIII_%s_chi2.csv'% (self.obj_in,self.min_chi_sq_cl3)))))+1 != len(self.chi_sq):
			print('MA PERCHE???????????')
			while len(list(csv.DictReader(open(self.PATH_OUT+'%s_clIII_%s_chi2.csv'% (self.obj_in,self.min_chi_sq_cl3)))))+1 != len(self.chi_sq):
				print('MA PERCHE???????????')
				w = csv.writer(open(self.PATH_OUT+'%s_clIII_%s_chi2.csv'% (self.obj_in,self.min_chi_sq_cl3), 'w'))
				for key, val in self.chi_sq.items():
					w.writerow([key, val, self.H_fin[key],self.K_fin[key]])

	def resultsToFile(self):
		self.PATH_OUT =self.PATH_OUT
		self.obj_in  = self.obj_in
		wl_VIS = self.wl_VIS
		fl_VIS = self.fl_VIS_in/cardelli_extinction(wl_VIS*10.,self.min_chi_sq_Av, Rv=self.Rv)
		self.filename_VIS =self.filename_VIS
		if os.path.isfile(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out'):
			wl_slab,fl_slab = readcol_py3(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')
		else:
			f = open(PATH_ACC+'in.slab', 'w')
			outLine = self.min_chi_sq_T+'   '+self.min_chi_sq_Ne+'   '+self.min_chi_sq_tau
			f.write(outLine)
			f.close()
			os.chdir(PATH_ACC)
			os.system('./hydrogen_slab')
			os.chdir(PATH)
			wl_slab,fl_slab = readcol_py3(PATH_ACC+'results/continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')

		F_acc = float(int_tabulated(wl_slab,fl_slab*self.min_chi_sq_H))  #  [erg s-1 cm-2]

		dist = self.dist_pc*3.1e18	# [cm]
		lum_acc = float(F_acc*4.*np.pi*(dist**2.)) #  [erg s-1]

		Lacc_Lsun = float(lum_acc/3.84e33) # in L_sun

		s = readsav(PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,cl3_in_toSelectModel))
		wl_slab_UVB_c,fl_slab_UVB_c = s['w'],s['f']
		s = readsav(PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,cl3_in_toSelectModel))
		wl_slab_VIS_c,fl_slab_VIS_c = s['w'],s['f']

		wl_slab = np.concatenate([wl_slab_UVB_c[wl_slab_UVB_c<550],wl_slab_VIS_c[wl_slab_VIS_c>550]])
		fl_slab = np.concatenate([fl_slab_UVB_c[wl_slab_UVB_c<550],fl_slab_VIS_c[wl_slab_VIS_c>550]])

		relDir = PATH+'/models_grid/SpT_Teff_relation_hh14_short_codes.dat'
		relation = np.genfromtxt(relDir,usecols=(1,2),skip_header=1,dtype=[('sptCode',float),('Teff',float)])

		BolCorrDir = PATH+'/models_grid/BolCorr_hh14_myCorr_wInteroplationOverVO.txt'
		BolCorr = np.genfromtxt(BolCorrDir,usecols=(0,1),skip_header=1,dtype=[('Tphot',float),('f751Fbol',float)])

		Teff = float(np.interp(float(self.min_chi_sq_cl3),relation['sptCode'],relation['Teff']))

		f751_obs = np.median(fl_VIS[(wl_VIS>747)&(wl_VIS<755)])
		wlScaling = 751

		if Teff < 3500:
			xlow = 730
			xhi =758
			flow = np.nanmedian(fl_VIS[(wl_VIS>xlow-0.4)&(wl_VIS<xlow+0.4)]) - (self.min_chi_sq_H*np.nanmedian(fl_slab[(wl_slab>xlow-0.4)&(wl_slab<xlow+0.4)]))
			fhigh = np.nanmedian(fl_VIS[(wl_VIS>xhi-0.4)&(wl_VIS<xhi+0.4)]) - (self.min_chi_sq_H*np.nanmedian(fl_slab[(wl_slab>xhi-0.4)&(wl_slab<xhi+0.4)]))
			f751_obs = ((flow*(xhi - wlScaling) )+ ((wlScaling - xlow)*fhigh))/(xhi-xlow)
			fl_photosphere751 = f751_obs
		else:
			f751_obs = np.median(fl_VIS[(wl_VIS>747)&(wl_VIS<755)])
			fl_slab751_forLum = np.median(fl_slab[(wl_slab>747)&(wl_slab<755)])
			fl_photosphere751 = f751_obs - (fl_slab751_forLum*self.min_chi_sq_H)

		fact = float(np.interp(Teff,BolCorr['Tphot'] ,BolCorr['f751Fbol']))
		fbol = float(fl_photosphere751/(10*fact))
		CmPerPars = 3.08567758128e18
		Lsun = 3.826e33 #ergs/s
		Lstar_input = float(4*np.pi*((self.dist_pc*CmPerPars)**2)*fbol * (1/Lsun))

		logLstar_input = float(np.log10(Lstar_input))

		print('the equivalent widths in the observed spectra are:  ',eqwValObs)
		featuresPlotVIS,errorsPlotVIS = classIIIFeatPlottingVIS.getFeatsAtSpt_symetricErr(self.min_chi_sq_cl3)
		featuresPlotUVB,errorsPlotUVB = classIIIFeatPlottingUVB.getFeatsAtSpt_symetricErr(self.min_chi_sq_cl3)
		eqwInterp,eqwInterpErr = EQWreadIn.getFeatsAtSpt_symetricErr(self.min_chi_sq_cl3)
		usedFeaturesPlotting = np.concatenate((usedFeaturesPlottingUVB,usedFeaturesPlottingVIS))
		featuresPlot = np.concatenate((featuresPlotUVB,featuresPlotVIS))
		errorsPlot = np.concatenate((errorsPlotUVB,errorsPlotVIS))
		wlFeatPlot = (usedFeaturesPlotting[:,0]+usedFeaturesPlotting[:,1])/2
		self.usedEQWlines = usedEQWlines
		fit_eqw = np.zeros(len(usedEQWlines))
		fit_eqw_err = np.zeros(len(usedEQWlines))
		for i in range(len(usedEQWlines)):
			fluxSlabAtWl = self.min_chi_sq_H*compute_flux_at_wl_std(wl_slab,fl_slab,usedEQWlines[i,0],interval=usedEQWlines[i,1])[0]
			contAroundLine = compute_flux_at_wl_std(wlFeatPlot,featuresPlot,usedEQWlines[i,0]+(2*usedEQWlines[i,1]),interval=2*usedEQWlines[i,1])[0] + compute_flux_at_wl_std(wlFeatPlot,featuresPlot,usedEQWlines[i,0]-(2*usedEQWlines[i,1]),interval=2*usedEQWlines[i,1])[0]
			fluxclIIIAtWl = self.min_chi_sq_K*contAroundLine/2
			fit_eqw[i] = eqwInterp[i]*fluxclIIIAtWl/(fluxSlabAtWl+fluxclIIIAtWl)
			fit_eqw_err[i] = 0
		print('the equivalent widths in the (computed using the rescaling) model is:  ',fit_eqw)

		SpTBestFit = scod.convScodToSpTstring(float(self.min_chi_sq_cl3))

		f = open(self.PATH_OUT+self.obj_in+'_best_fit.dat','w')
		f.write('FIT OF THE OBJECT %s USING THE CLASS III %s \n' % (self.obj_in,self.min_chi_sq_cl3))
		f.write('Executed on '+time.asctime( time.localtime(time.time()) )+'\n')
		f.write('Using the file %s\n' % self.filename_VIS)
		f.write('\n')
		f.write('INPUT PARAMETERS:\n')
		f.write('dist = %i pc\n' % self.dist_pc)
		f.write('self.Rv = %0.1f\n' % self.Rv)
		f.write('\n')
		f.write('BEST FIT:\n')
		f.write('CLASS III: %s \n' % (self.min_chi_sq_cl3))
		f.write('\n')
		f.write('spectral type: '+ SpTBestFit)
		f.write('\n')
		f.write('Teff: %i \n' % (Teff))
		f.write('Chi2 = %0.3e \n' % self.min_chi_sq)
		f.write('\n')
		f.write('OBJECT PARAMETERS:\n')
		f.write('Av = %0.2f mag\n' % self.min_chi_sq_Av)
		f.write('Lacc/Lsun = %0.2e \n'% Lacc_Lsun)
		f.write('log(Lacc/Lsun) = %0.3f \n'% np.log10(Lacc_Lsun))
		f.write('Lstar = %0.2f Lsun\n' % Lstar_input)
		f.write('log(Lstar/Lsun) = %0.2f \n' % logLstar_input)
		f.write('\n')
		f.write('NORMALIZATION CONSTANTS: \n')
		f.write('H = %0.3e \n' % (self.min_chi_sq_H))
		f.write('K = %0.3e \n'% (self.min_chi_sq_K))
		f.write('\n')
		f.write('SLAB PARAMETERS:\n')
		f.write('T = %s K \n'% self.min_chi_sq_T)
		f.write('n_e = %s cm-3\n'% self.min_chi_sq_Ne)
		f.write('tau (300 nm) =  %s \n'% self.min_chi_sq_tau)
		f.write('Area = %0.2e cm2 \n' % (self.min_chi_sq_H*dist**2.))
		f.write('Radius = %0.2e cm \n'% (np.sqrt(self.min_chi_sq_H*dist**2./np.pi)))
		f.write('\n')
		f.write('OBJECT DERIVED PARAMETERS:\n')
		mass_siess,age_siess = isochrone_interp(np.array([np.log10(Teff)]),np.array([logLstar_input]),model='Siess',PATH = PATH)
		mass_bara,age_bara = isochrone_interp(np.array([np.log10(Teff)]),np.array([logLstar_input]),model='Baraffe',PATH = PATH)
		mass_palla,age_palla = isochrone_interp(np.array([np.log10(Teff)]),np.array([logLstar_input]),model='Palla',PATH = PATH)
		mass_danto,age_danto = isochrone_interp(np.array([np.log10(Teff)]),np.array([logLstar_input]),model='Dantona',PATH = PATH)
		mass_b15,age_b15 = isochrone_interp(np.array([np.log10(Teff)]),np.array([logLstar_input]),model='B15',PATH = PATH)
		mass_Feiden,age_Feiden = isochrone_interp(np.array([np.log10(Teff)]),np.array([logLstar_input]),model='Feiden',PATH = PATH)
		mstar,macc_siess = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Siess',PATH = PATH)
		mstar,macc_bara = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Baraffe',PATH = PATH)
		mstar,macc_palla = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Palla',PATH = PATH)
		mstar,macc_danto = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Dantona',PATH = PATH)
		mstar,macc_b15 = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='B15',PATH = PATH)
		mstar,macc_Feiden = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Feiden',PATH = PATH)
		f.write('M = %0.2f Msun Age = %.2f (BARAFFE+15)\n' % (float(np.squeeze(mass_b15)),10.**(float(np.squeeze(age_b15)))/1e6))
		f.write('Macc = %0.2e Msun/yr (BARAFFE+15)\n' % float(np.squeeze(macc_b15)))
		f.write('M = %0.2f Msun Age = %.2f (SIESS)\n' % (float(np.squeeze(mass_siess)),10.**(float(np.squeeze(age_siess)))/1e6))
		f.write('Macc = %0.2e Msun/yr (SIESS)\n' % float(np.squeeze(macc_siess)))
		f.write('M = %0.2f Msun Age = %.2f (BARAFFE)\n' % (float(np.squeeze(mass_bara)),10.**(float(np.squeeze(age_bara)))/1e6))
		f.write('Macc = %0.2e Msun/yr (BARAFFE)\n' % float(np.squeeze(macc_bara)))
		f.write('M = %0.2f Msun Age = %.2f (PALLA)\n' % (float(np.squeeze(mass_palla)),10.**(float(np.squeeze(age_palla)))/1e6))
		f.write('Macc = %0.2e Msun/yr (PALLA)\n' % float(np.squeeze(macc_palla)))
		f.write('M = %0.2f Msun Age = %.2f (DANTONA)\n' % (float(np.squeeze(mass_danto)),10.**(float(np.squeeze(age_danto)))/1e6))
		f.write('Macc = %0.2e Msun/yr (DANTONA)\n' % float(np.squeeze(macc_danto)))
		f.write('M = %0.2f Msun Age = %.2f (Feiden)\n' % (float(np.squeeze(mass_Feiden)),10.**(float(np.squeeze(age_Feiden)))/1e6))
		f.write('Macc = %0.2e Msun/yr (Feiden)\n' % float(np.squeeze(macc_Feiden)))
		f.write('\n')
		f.write('\n')
		f.write('Other best chi2 results:\n')
		f.write('%s' % self.best_chi_sq)
		f.write('\n')
		f.write('\n')
		f.write('FITTED PARAMETERS:\n')
		f.write('Class IIIs: %s \n' % cl3_in)
		f.write('Av: %s \n' % self.Av_list)
		f.write('\n')
		f.write('\n')
		f.write('These values were computed using the grit found at: \n')
		f.write(gridFile +'\n')
		f.write('this grid contains the following features: \n')
		f.write(str(usedFeatures)+'\n')
		f.write('and is normalized at: \n')
		f.write(str(normWLandWidth[0])+'nm')
		f.write('\n')
		f.write('\n')
		f.close()

	def addEntryToTable(self, FileOut = None):
		self.dist_pc = self.dist_pc
		self.best_chi_sq = self.best_chi_sq
		self.min_chi_sq = self.min_chi_sq
		self.min_chi_sq_cl3 = self.min_chi_sq_cl3
		self.min_chi_sq_Av = self.min_chi_sq_Av
		self.min_chi_sq_T = self.min_chi_sq_T
		self.min_chi_sq_Ne = self.min_chi_sq_Ne
		self.min_chi_sq_tau = self.min_chi_sq_tau
		self.min_chi_sq_H = self.min_chi_sq_H
		self.min_chi_sq_K = self.min_chi_sq_K
		self.Av_list = self.Av_list
		self.PATH_OUT =self.PATH_OUT
		self.obj_in  = self.obj_in
		wl_VIS = self.wl_VIS
		fl_VIS = self.fl_VIS_in/cardelli_extinction(wl_VIS*10.,self.min_chi_sq_Av, Rv=self.Rv)
		self.filename_VIS =self.filename_VIS
		if os.path.isfile(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out'):
			wl_slab,fl_slab = readcol_py3(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')
		else:
			f = open(PATH_ACC+'in.slab', 'w')
			outLine = self.min_chi_sq_T+'   '+self.min_chi_sq_Ne+'   '+self.min_chi_sq_tau
			f.write(outLine)
			f.close()
			os.chdir(PATH_ACC)
			os.system('./hydrogen_slab')
			os.chdir(PATH)
			wl_slab,fl_slab = readcol_py3(PATH_ACC+'results/continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')

		F_acc = float(int_tabulated(wl_slab,fl_slab*self.min_chi_sq_H))  #  [erg s-1 cm-2]

		dist = self.dist_pc*3.1e18	# [cm]
		lum_acc = float(F_acc*4.*np.pi*(dist**2.)) #  [erg s-1]

		Lacc_Lsun = float(lum_acc/3.84e33) # in L_sun

		s = readsav(PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,cl3_in_toSelectModel))
		wl_slab_UVB_c,fl_slab_UVB_c = s['w'],s['f']
		s = readsav(PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,cl3_in_toSelectModel))
		wl_slab_VIS_c,fl_slab_VIS_c = s['w'],s['f']

		wl_slab = np.concatenate([wl_slab_UVB_c[wl_slab_UVB_c<550],wl_slab_VIS_c[wl_slab_VIS_c>550]])
		fl_slab = np.concatenate([fl_slab_UVB_c[wl_slab_UVB_c<550],fl_slab_VIS_c[wl_slab_VIS_c>550]])

		relDir = PATH+'/models_grid/SpT_Teff_relation_hh14_short_codes.dat'
		relation = np.genfromtxt(relDir,usecols=(1,2),skip_header=1,dtype=[('sptCode',float),('Teff',float)])

		BolCorrDir = PATH+'/models_grid/BolCorr_hh14_myCorr_wInteroplationOverVO.txt'
		BolCorr = np.genfromtxt(BolCorrDir,usecols=(0,1),skip_header=1,dtype=[('Tphot',float),('f751Fbol',float)])

		Teff = float(np.interp(float(self.min_chi_sq_cl3),relation['sptCode'],relation['Teff']))

		f751_obs = np.median(fl_VIS[(wl_VIS>747)&(wl_VIS<755)])
		wlScaling = 751

		if Teff < 3500:
			xlow = 730
			xhi =758
			flow = np.nanmedian(fl_VIS[(wl_VIS>xlow-0.4)&(wl_VIS<xlow+0.4)]) - (self.min_chi_sq_H*np.nanmedian(fl_slab[(wl_slab>xlow-0.4)&(wl_slab<xlow+0.4)]))
			fhigh = np.nanmedian(fl_VIS[(wl_VIS>xhi-0.4)&(wl_VIS<xhi+0.4)]) - (self.min_chi_sq_H*np.nanmedian(fl_slab[(wl_slab>xhi-0.4)&(wl_slab<xhi+0.4)]))
			f751_obs = ((flow*(xhi - wlScaling) )+ ((wlScaling - xlow)*fhigh))/(xhi-xlow)
			fl_photosphere751 = f751_obs
		else:
			f751_obs = np.median(fl_VIS[(wl_VIS>747)&(wl_VIS<755)])
			fl_slab751_forLum = np.median(fl_slab[(wl_slab>747)&(wl_slab<755)])
			fl_photosphere751 = f751_obs - (fl_slab751_forLum*self.min_chi_sq_H)

		fact = float(np.interp(Teff,BolCorr['Tphot'] ,BolCorr['f751Fbol']))
		fbol = float(fl_photosphere751/(10*fact))
		CmPerPars = 3.08567758128e18
		Lsun = 3.826e33 #ergs/s
		Lstar_input = float(4*np.pi*((self.dist_pc*CmPerPars)**2)*fbol * (1/Lsun))

		logLstar_input = float(np.log10(Lstar_input))

		SpTBestFit = scod.convScodToSpTstring(float(self.min_chi_sq_cl3))


		mass_siess,age_siess = isochrone_interp(np.array([np.log10(Teff)]),np.array([logLstar_input]),model='Siess',PATH = PATH)
		mass_bara,age_bara = isochrone_interp(np.array([np.log10(Teff)]),np.array([logLstar_input]),model='Baraffe',PATH = PATH)
		mass_palla,age_palla = isochrone_interp(np.array([np.log10(Teff)]),np.array([logLstar_input]),model='Palla',PATH = PATH)
		mass_danto,age_danto = isochrone_interp(np.array([np.log10(Teff)]),np.array([logLstar_input]),model='Dantona',PATH = PATH)
		mass_b15,age_b15 = isochrone_interp(np.array([np.log10(Teff)]),np.array([logLstar_input]),model='B15',PATH = PATH)
		mass_Feiden,age_Feiden = isochrone_interp(np.array([np.log10(Teff)]),np.array([logLstar_input]),model='Feiden',PATH = PATH)
		mstar,macc_siess = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Siess',PATH =PATH)
		mstar,macc_bara = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Baraffe',PATH =PATH)
		mstar,macc_palla = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Palla',PATH =PATH)
		mstar,macc_danto = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Dantona',PATH =PATH)
		mstar,macc_b15 = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='B15',PATH =PATH)
		mstar,macc_Feiden = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Feiden',PATH =PATH)

		if FileOut == None:
			pathTable = self.path+'someSetOfresults.dat'
		else:
			pathTable = FileOut

		if os.path.isfile(pathTable):
			f = open(pathTable,'a')
		else:
			f = open(pathTable,'a')
			header = 'Name  \t dist \t SpT \t SpTCode \t Teff \t Av \t Rv \t lstar \t loglacc \t mstar_b15  \t macc_b15 \t mstar_siess  \t macc_siess \t Tslab \t Neslab \t Tauslab \t	Hslab \t KclIII \n'
			f.write(header)

		f.write('%s \t %0.1f \t %s \t %0.2f \t %0.1f \t %0.2f \t %0.2f  \t %0.3e \t %0.3f \t %0.3f \t %0.4e \t %0.3f \t %0.4e \t %s \t %s \t %s \t %0.4e \t %0.4e \n'
		% (self.obj_in,self.dist_pc,SpTBestFit,float(self.min_chi_sq_cl3),Teff,self.min_chi_sq_Av,Rv,Lstar_input,np.log10(Lacc_Lsun),mass_b15,
		float(np.squeeze(macc_b15)),float(np.squeeze(mass_siess)),float(np.squeeze(macc_siess)),self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,self.min_chi_sq_H,self.min_chi_sq_K))
		f.write('\t')

		f.close()

	def plotRegFit(self,close = False):
		usedFeatures = self.usedFeatures
		classIIIreadIn = self.classIIIreadIn
		normWLandWidth = self.normWLandWidth

		wl_UVB = self.wl_UVB
		fl_UVB = self.fl_UVB_in/cardelli_extinction(wl_UVB*10.,self.min_chi_sq_Av, Rv=self.Rv)
		wl_VIS = self.wl_VIS
		fl_VIS = self.fl_VIS_in/cardelli_extinction(wl_VIS*10.,self.min_chi_sq_Av, Rv=self.Rv)
		self.filename_VIS =self.filename_VIS

		self.filename_NIR = self.filename_NIR
		self.fitsTab = self.fitsTab
		if self.filename_NIR == None:
			self.filename_NIR = path+'data_final/flux_%s_nir_tell.fits' % self.obj_in
		print('NIR path check:', self.filename_NIR)
		if os.path.isfile(self.filename_NIR):
			if self.fitsTab == False:
				wl_NIR,fl_NIR_in,hdr_NIR=spec_readspec(self.filename_NIR, 'hdr')	#the 'hdr' string is there to say that I want to save the header
			else:
				wl_NIR,fl_NIR_in,hdr_NIR=readspec_phase3(self.filename_NIR, hdr_out='YES')
			if self.perAA == True:
				fl_NIR_in = 10*fl_NIR_in
		else:
			print('No NIR spectrum')
			wl_NIR,fl_NIR_in = np.array([0 for x in range(100)]),np.array([0 for x in range(100)])

		fl_NIR = fl_NIR_in/cardelli_extinction(wl_NIR*10.,self.min_chi_sq_Av, Rv=self.Rv)

		if os.path.isfile(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out'):
			wl_slab,fl_slab = readcol_py3(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')
		else:
			f = open(PATH_ACC+'in.slab', 'w')
			outLine = self.min_chi_sq_T+'   '+self.min_chi_sq_Ne+'   '+self.min_chi_sq_tau
			f.write(outLine)
			f.close()
			os.chdir(PATH_ACC)
			os.system('./hydrogen_slab')
			os.chdir(PATH)
			wl_slab,fl_slab = readcol_py3(PATH_ACC+'results/continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')

		p1 = pl.figure(figsize=(10,9))
		pl.subplot(211)

		gridFilePlot = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeUVB/interp.npz'
		classIIIFeatPlotting = pf.classIII(gridFilePlot)
		usedFeaturesPlotting = classIIIFeatPlotting.getUsedInterpFeat()
		featuresPlot,errorsPlot = classIIIFeatPlotting.getFeatsAtSpt_symetricErr(self.min_chi_sq_cl3)

		gridFilePlotVIS = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeVIS/interp.npz'
		classIIIFeatPlottingVIS = pf.classIII(gridFilePlotVIS)
		usedFeaturesPlottingVIS = classIIIFeatPlottingVIS.getUsedInterpFeat()
		featuresPlotVIS,errorsPlotVIS = classIIIFeatPlottingVIS.getFeatsAtSpt_symetricErr(self.min_chi_sq_cl3)

		usedFeaturesPlotting = np.concatenate((usedFeaturesPlotting,usedFeaturesPlottingVIS))
		featuresPlot = np.concatenate((featuresPlot,featuresPlotVIS))
		errorsPlot = np.concatenate((errorsPlot,errorsPlotVIS))

		if plot_smooth == False:
			pl.plot(wl_UVB[self.ind_uvb],fl_UVB[self.ind_uvb],'k',zorder = 1)#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
			print('best fit sptCode = '+self.min_chi_sq_cl3)

			wlFeat = (usedFeatures[:,0]+usedFeatures[:,1])/2
			Xrange = np.abs(usedFeatures[:,0] - wlFeat)
			features,errors = classIIIreadIn.getFeatsAtSpt_symetricErr(self.min_chi_sq_cl3)

			fit_cont = np.zeros(len(usedFeatures))
			for i in range(len(usedFeatures)):
				fit_cont[i] = self.min_chi_sq_K*features[i]+self.min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeatures[i,0],usedFeatures[i,1])[0]
			stdTerm1 = (self.min_chi_sq_K*errors)**2
			stdTerm2 = 0#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not used for the plot, this is a hold over!!!
			fit_std = np.sqrt(stdTerm1 +stdTerm2)

			pl.errorbar(wlFeat,self.min_chi_sq_K*features,self.min_chi_sq_K*errors,xerr = Xrange,fmt='.',c=(213/255,94/255,0/255),zorder = 10)
			pl.errorbar(wlFeat,fit_cont,yerr=fit_std,xerr= Xrange,fmt='.',c=(204/255 ,121/255,167/255),zorder = 10)

			wlFeatPlot = (usedFeaturesPlotting[:,0]+usedFeaturesPlotting[:,1])/2
			XrangePlot = np.abs(usedFeaturesPlotting[:,0] - wlFeatPlot)

			fit_contPlot = np.zeros(len(usedFeaturesPlotting))
			for i in range(len(usedFeaturesPlotting[:,0])):
				fit_contPlot[i] = self.min_chi_sq_K*featuresPlot[i]+self.min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeaturesPlotting[i,0],usedFeaturesPlotting[i,1])[0]
			stdTerm1plot = (self.min_chi_sq_K*errorsPlot)**2
			stdTerm2plot = 0#(stddev_cont_dered**2)#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not taken into account!!!
			fit_stdPlot = np.sqrt(stdTerm1plot +stdTerm2plot)

			pl.plot(wlFeatPlot,self.min_chi_sq_K*featuresPlot ,c=(240/255,228/255,66/255),zorder = 10)
			pl.plot(wlFeatPlot,fit_contPlot ,c=(86/255,180/255,233/255),zorder = 10)

			pl.plot(wl_cl3_UVB[ind_uvb_3],self.min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],c= (0,158/255,115/255),zorder = 9)
		else:
			wl_UVB_smooth = wl_UVB[self.ind_uvb][::8]
			fl_UVB_smooth = spectrum_resample(fl_UVB,wl_UVB,wl_UVB_smooth)
			pl.plot(wl_UVB_smooth,fl_UVB_smooth,'k',zorder = 1)#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')

			wlFeat = (usedFeatures[:,0]+usedFeatures[:,1])/2
			Xrange = np.abs(usedFeatures[:,0] - wlFeat)
			features,errors = classIIIreadIn.getFeatsAtSpt_symetricErr(self.min_chi_sq_cl3)

			fit_cont = np.zeros(len(usedFeatures))
			for i in range(len(usedFeatures)):
				fit_cont[i] = self.min_chi_sq_K*features[i]+self.min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeatures[i,0],usedFeatures[i,1])[0]
			stdTerm1 = (self.min_chi_sq_K*errors)**2
			stdTerm2 = 0#(stddev_cont_dered**2)
			fit_std = np.sqrt(stdTerm1 +stdTerm2)

			pl.errorbar(wlFeat,self.min_chi_sq_K*features,self.min_chi_sq_K*errors,xerr = Xrange,fmt='.',c=(213/255,94/255,0/255),zorder = 10)
			pl.errorbar(wlFeat,fit_cont,yerr=fit_std,xerr= Xrange,fmt='.',c=(204/255,121/255,167/255),zorder = 10)

			wlFeatPlot = (usedFeaturesPlotting[:,0]+usedFeaturesPlotting[:,1])/2
			XrangePlot = np.abs(usedFeaturesPlotting[:,0] - wlFeatPlot)

			fit_contPlot = np.zeros(len(usedFeaturesPlotting))
			for i in range(len(usedFeaturesPlotting[:,0])):
				fit_contPlot[i] = self.min_chi_sq_K*featuresPlot[i]+self.min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeaturesPlotting[i,0],usedFeaturesPlotting[i,1])[0]
			stdTerm1plot = (self.min_chi_sq_K*errorsPlot)**2
			stdTerm2plot = 0#(stddev_cont_dered**2)#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not taken into account!!!
			fit_stdPlot = np.sqrt(stdTerm1plot +stdTerm2plot)

			pl.plot(wlFeatPlot,self.min_chi_sq_K*featuresPlot ,c=(240/255,228/255,66/255),zorder = 10)
			pl.plot(wlFeatPlot,fit_contPlot ,c=(86/255,180/255,233/255),zorder = 10)

			pl.plot(wl_slab,self.min_chi_sq_H*fl_slab,'c',zorder = 9)

		pl.title(self.obj_in)#+' - Chi$^2$= %0.3e' % (self.min_chi_sq))
		pl.ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$]')
		pl.axis([330,470,-1e-16,3.5*np.mean(fl_UVB[(wl_UVB < 360) & (wl_UVB > 330)])])
		pl.savefig(self.PATH_OUT+'%s_clIII_%s.png'% (self.obj_in,self.min_chi_sq_cl3))

		PATH_MY_CLASSIII = PATH+'/models_grid/RunnableGrid/'

		wl_cl3_UVB,fl_cl3_UVB,wl_cl3_VIS,fl_cl3_VIS,wl_cl3_NIR,fl_cl3_NIR,nameWls = pf.readMixClassIII(self.min_chi_sq_cl3,PATH_MY_CLASSIII,wlNorm =normWLandWidth[0])
		ind_uvb_3 = (wl_cl3_UVB <= max_wl_uvb)
		ind_vis_3 = (wl_cl3_VIS >= min_wl_vis)

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c= s['w'],s['f']
		else:
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c = s['w'],s['f']
			fl_slab_UVB_c = np.interp(wl_cl3_UVB,wl_slab_UVB_c,fl_slab_UVB_c )
			wl_slab_UVB_c = wl_cl3_UVB

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c= s['w'],s['f']
		else:
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c = s['w'],s['f']
			fl_slab_VIS_c = np.interp(wl_cl3_VIS,wl_slab_VIS_c,fl_slab_VIS_c )
			wl_slab_VIS_c = wl_cl3_VIS

		f_in_norm = np.nanmedian(fl_UVB[(wl_UVB > 417.5) & (wl_UVB < 419.5)])
		f_tot_norm = np.nanmedian(fl_cl3_UVB[(wl_cl3_UVB > 417.5)&(wl_cl3_UVB < 419.5)]*self.min_chi_sq_K+fl_slab_UVB_c[(wl_cl3_UVB > 417.5)&(wl_cl3_UVB < 419.5)]*self.min_chi_sq_H)
		f_clIII_norm = np.nanmedian(fl_cl3_UVB[(wl_cl3_UVB > 417.5)&(wl_cl3_UVB < 419.5)]*self.min_chi_sq_K)
		f_ClIIIFeat_norm = self.min_chi_sq_K*featuresPlot[(wlFeatPlot > 417.5) & (wlFeatPlot < 419.5)]
		f_totFeat_norm = fit_contPlot[(wlFeatPlot > 417.5) & (wlFeatPlot < 419.5)]

		pl.subplot(224)
		pl.plot(wl_UVB[self.ind_uvb],fl_UVB[self.ind_uvb]/f_in_norm,'k')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
		pl.xlabel('Wavelength [nm]')
		pl.axis([420,425,0,1.6])

		pl.plot(wl_cl3_UVB[ind_uvb_3],self.min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3]/f_tot_norm,'c')

		pl.plot(wlFeatPlot,(self.min_chi_sq_K*featuresPlot)/f_ClIIIFeat_norm ,c=(240/255,228/255,66/255),zorder = 10)
		pl.plot(wlFeatPlot,(fit_contPlot)/f_totFeat_norm ,c=(86/255,180/255,233/255),zorder = 10)

		pl.plot(wl_cl3_UVB[ind_uvb_3],(self.min_chi_sq_K*fl_cl3_UVB[ind_uvb_3]+self.min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3])/f_tot_norm,c='b')
		pl.plot(wl_cl3_UVB[ind_uvb_3],self.min_chi_sq_K*fl_cl3_UVB[ind_uvb_3]/f_clIII_norm,c ='g',alpha =0.6)

		ind_vis = (wl_VIS >= min_wl_vis)
		pl.subplot(223)
		pl.plot(wl_VIS[ind_vis],fl_VIS[ind_vis],'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
		pl.xlabel('Wavelength [nm]')
		pl.ylabel('Flux')
		pl.axis([700,720,-1e-16,1.5*np.mean(fl_VIS[(wl_VIS < 720) & (wl_VIS > 700)])])
		pl.errorbar(wlFeat,fit_cont,yerr=fit_std,xerr= Xrange,fmt='.',c='b',zorder = 10)
		pl.errorbar(wlFeat,self.min_chi_sq_K*features,self.min_chi_sq_K*errors,xerr = Xrange,fmt='.',c='g',zorder = 10)
		pl.plot(wl_slab,self.min_chi_sq_H*fl_slab,'c',zorder = 9)

		pl.savefig(self.PATH_OUT+'%s_clIII_%s_%s.png'% (self.obj_in,self.min_chi_sq_cl3,nameWls))
		pl.show()
		if close:
			pl.close()

	def plotVeil(self,close =False):
		
		usedFeatures = self.usedFeatures
		classIIIreadIn = self.classIIIreadIn
		normWLandWidth = self.normWLandWidth

		wl_UVB = self.wl_UVB
		fl_UVB = self.fl_UVB_in/cardelli_extinction(wl_UVB*10.,self.min_chi_sq_Av, Rv=self.Rv)
		wl_VIS = self.wl_VIS
		fl_VIS = self.fl_VIS_in/cardelli_extinction(wl_VIS*10.,self.min_chi_sq_Av, Rv=self.Rv)
		self.filename_VIS =self.filename_VIS

		self.filename_NIR = self.filename_NIR
		self.fitsTab = self.fitsTab
		if self.filename_NIR == None:
			self.filename_NIR = path+'data_final/flux_%s_nir_tell.fits' % self.obj_in
		print('NIR path check:', self.filename_NIR)
		if os.path.isfile(self.filename_NIR):
			if self.fitsTab == False:
				wl_NIR,fl_NIR_in,hdr_NIR=spec_readspec(self.filename_NIR, 'hdr')	#the 'hdr' string is there to say that I want to save the header
			else:
				wl_NIR,fl_NIR_in,hdr_NIR=readspec_phase3(self.filename_NIR, hdr_out='YES')
			if self.perAA == True:
				fl_NIR_in = 10*fl_NIR_in
		else:
			print('No NIR spectrum')
			wl_NIR,fl_NIR_in = np.array([0 for x in range(100)]),np.array([0 for x in range(100)])

		fl_NIR = fl_NIR_in/cardelli_extinction(wl_NIR*10.,self.min_chi_sq_Av, Rv=self.Rv)

		if os.path.isfile(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out'):
			wl_slab,fl_slab = readcol_py3(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')
		else:
			f = open(PATH_ACC+'in.slab', 'w')
			outLine = self.min_chi_sq_T+'   '+self.min_chi_sq_Ne+'   '+self.min_chi_sq_tau
			f.write(outLine)
			f.close()
			os.chdir(PATH_ACC)
			os.system('./hydrogen_slab')
			os.chdir(PATH)
			wl_slab,fl_slab = readcol_py3(PATH_ACC+'results/continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')
		PATH_MY_CLASSIII = PATH+'/models_grid/RunnableGrid/'

		wl_cl3_UVB,fl_cl3_UVB,wl_cl3_VIS,fl_cl3_VIS,wl_cl3_NIR,fl_cl3_NIR,nameWls = pf.readMixClassIII(self.min_chi_sq_cl3,PATH_MY_CLASSIII,wlNorm =normWLandWidth[0])
		ind_uvb_3 = (wl_cl3_UVB <= max_wl_uvb)
		ind_vis_3 = (wl_cl3_VIS >= min_wl_vis)

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c= s['w'],s['f']
		else:
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c = s['w'],s['f']
			fl_slab_VIS_c = np.interp(wl_cl3_VIS,wl_slab_VIS_c,fl_slab_VIS_c )
			wl_slab_VIS_c = wl_cl3_VIS

		"""
		PLOT PHOTOSPHERIC FEATURES
		"""
		eqw, err_eqw, mode, fwhm, lline = eqw_auto(wl_VIS,fl_VIS,670.78,size_cont=1.,plot='NO',mode='gauss',fwhm_out='Ja',lline_out='Ja')
		if self.min_chi_sq_cl3 == 'LM601':
			lline3 = 670.8
		elif self.min_chi_sq_cl3 == 'Sz94':
			lline3 = 670.3
		else:
			eqw3, err_eqw3, mode3, fwhm3, lline3 = eqw_auto(wl_cl3_VIS,fl_cl3_VIS,670.78,size_cont=1.,plot='NO',mode='gauss',fwhm_out='Ja',lline_out='Ja')

		shift = lline - lline3

		p1 = pl.figure(figsize=(10,9))
		f_in_norm = np.mean(fl_VIS[(wl_VIS > 817.0) & (wl_VIS < 818.0)])
		f_tot_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 817.)&(wl_cl3_VIS < 818.)]*self.min_chi_sq_K+fl_slab_VIS_c[(wl_cl3_VIS > 817.)&(wl_cl3_VIS < 818.)]*self.min_chi_sq_H)
		f_clIII_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 817.)&(wl_cl3_VIS < 818.)]*self.min_chi_sq_K)

		pl.subplot(221)
		pl.plot(wl_VIS[self.ind_vis]-shift,fl_VIS[self.ind_vis]/f_in_norm,'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
		pl.title(self.obj_in)
		pl.ylabel('Flux')
		pl.axis([816,822,0.1,1.3])
		pl.plot(wl_cl3_VIS[ind_vis_3],(self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3])/f_tot_norm,'b')
		pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3]/f_clIII_norm,'g',alpha =0.6)
		pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3]/f_tot_norm,'c')
		pl.text(821,0.4,'NaI')

		f_in_norm = np.mean(fl_VIS[(wl_VIS > 767.50) & (wl_VIS < 769.0)])
		f_tot_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 767.5)&(wl_cl3_VIS < 769.)]*self.min_chi_sq_K+fl_slab_VIS_c[(wl_cl3_VIS > 767.5)&(wl_cl3_VIS < 769.)]*self.min_chi_sq_H)
		f_clIII_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 767.5)&(wl_cl3_VIS < 769.)]*self.min_chi_sq_K)

		pl.subplot(222)
		pl.plot(wl_VIS[self.ind_vis]-shift,fl_VIS[self.ind_vis]/f_in_norm,'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
		pl.axis([764,774,0.1,1.3])
		pl.plot(wl_cl3_VIS[ind_vis_3],(self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3])/f_tot_norm,'b')
		pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3]/f_clIII_norm,'g',alpha =0.6)
		pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3]/f_tot_norm,'c')
		pl.text(772.5,0.4,'KI')

		f_in_norm = np.mean(fl_VIS[(wl_VIS > 615.) & (wl_VIS < 615.8)])
		f_tot_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 615.)&(wl_cl3_VIS < 615.8)]*self.min_chi_sq_K+fl_slab_VIS_c[(wl_cl3_VIS > 615.)&(wl_cl3_VIS < 615.8)]*self.min_chi_sq_H)
		f_clIII_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 615.)&(wl_cl3_VIS < 615.8)]*self.min_chi_sq_K)

		pl.subplot(223)
		pl.plot(wl_VIS[self.ind_vis]-shift,fl_VIS[self.ind_vis]/f_in_norm,'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
		pl.xlabel('Wavelength [nm]')
		pl.ylabel('Flux')
		pl.axis([614,618,0.1,1.3])
		pl.xticks([614,615,616,617,618])
		pl.plot(wl_cl3_VIS[ind_vis_3],(self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3])/f_tot_norm,'b')
		pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3]/f_clIII_norm,'g',alpha =0.6)
		pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3]/f_tot_norm,'c')
		pl.text(617.3,0.4,'CaI')

		f_in_norm = np.mean(fl_VIS[(wl_VIS > 845.50) & (wl_VIS < 846.50)])
		f_tot_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 845.5)&(wl_cl3_VIS < 846.5)]*self.min_chi_sq_K+fl_slab_VIS_c[(wl_cl3_VIS > 845.5)&(wl_cl3_VIS < 846.5)]*self.min_chi_sq_H)
		f_clIII_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 845.5)&(wl_cl3_VIS < 846.5)]*self.min_chi_sq_K)

		pl.subplot(224)
		pl.plot(wl_VIS[self.ind_vis]-shift,fl_VIS[self.ind_vis]/f_in_norm,'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
		pl.xlabel('Wavelength [nm]')
		pl.axis([840,848,0.1,1.3])
		pl.plot(wl_cl3_VIS[ind_vis_3],(self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3])/f_tot_norm,'b')
		pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3]/f_clIII_norm,'g',alpha =0.6)
		pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3]/f_tot_norm,'c')
		pl.text(846,0.4,'TiO')

		pl.tight_layout()
		pl.savefig(self.PATH_OUT+'%s_clIII_%s_photosph_%s.png'% (self.obj_in,self.min_chi_sq_cl3,nameWls))
		pl.show()
		if close:
			pl.close()

	def plotLinesUsed(self):
		usedFeatures =self.usedFeatures
		classIIIreadIn = self.classIIIreadIn
		normWLandWidth = self.normWLandWidth
		usedEQWlines = self.usedEQWlines

		wl_UVB = self.wl_UVB
		fl_UVB = self.fl_UVB_in/cardelli_extinction(wl_UVB*10.,self.min_chi_sq_Av, Rv=self.Rv)
		wl_VIS = self.wl_VIS
		fl_VIS = self.fl_VIS_in/cardelli_extinction(wl_VIS*10.,self.min_chi_sq_Av, Rv=self.Rv)
		self.filename_VIS =self.filename_VIS

		wl_OBS = np.concatenate((wl_UVB[wl_UVB<550],wl_VIS[wl_VIS>550]))
		fl_OBS = np.concatenate((fl_UVB[wl_UVB<550],fl_VIS[wl_VIS>550]))

		self.filename_NIR = self.filename_NIR
		self.fitsTab = self.fitsTab
		if self.filename_NIR == None:
			self.filename_NIR = path+'data_final/flux_%s_nir_tell.fits' % self.obj_in
		print('NIR path check:', self.filename_NIR)
		if os.path.isfile(self.filename_NIR):
			if self.fitsTab == False:
				wl_NIR,fl_NIR_in,hdr_NIR=spec_readspec(self.filename_NIR, 'hdr')	#the 'hdr' string is there to say that I want to save the header
			else:
				wl_NIR,fl_NIR_in,hdr_NIR=readspec_phase3(self.filename_NIR, hdr_out='YES')
			if self.perAA == True:
				fl_NIR_in = 10*fl_NIR_in
		else:
			print('No NIR spectrum')
			wl_NIR,fl_NIR_in = np.array([0 for x in range(100)]),np.array([0 for x in range(100)])

		fl_NIR = fl_NIR_in/cardelli_extinction(wl_NIR*10.,self.min_chi_sq_Av, Rv=self.Rv)

		if os.path.isfile(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out'):
			wl_slab,fl_slab = readcol_py3(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')
		else:
			f = open(PATH_ACC+'in.slab', 'w')
			outLine = self.min_chi_sq_T+'   '+self.min_chi_sq_Ne+'   '+self.min_chi_sq_tau
			f.write(outLine)
			f.close()
			os.chdir(PATH_ACC)
			os.system('./hydrogen_slab')
			os.chdir(PATH)
			wl_slab,fl_slab = readcol_py3(PATH_ACC+'results/continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')
		PATH_MY_CLASSIII = PATH+'/models_grid/RunnableGrid/'

		wl_cl3_UVB,fl_cl3_UVB,wl_cl3_VIS,fl_cl3_VIS,wl_cl3_NIR,fl_cl3_NIR,nameWls = pf.readMixClassIII(self.min_chi_sq_cl3,PATH_MY_CLASSIII,wlNorm =normWLandWidth[0])
		ind_uvb_3 = (wl_cl3_UVB <= max_wl_uvb)
		ind_vis_3 = (wl_cl3_VIS >= min_wl_vis)

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c= s['w'],s['f']
		else:
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c = s['w'],s['f']
			fl_slab_UVB_c = np.interp(wl_cl3_UVB,wl_slab_UVB_c,fl_slab_UVB_c )
			wl_slab_UVB_c = wl_cl3_UVB

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c= s['w'],s['f']
		else:
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c = s['w'],s['f']
			fl_slab_VIS_c = np.interp(wl_cl3_VIS,wl_slab_VIS_c,fl_slab_VIS_c )
			wl_slab_VIS_c = wl_cl3_VIS

		wl_cl3 = np.concatenate([wl_cl3_UVB[wl_cl3_UVB<550],wl_cl3_VIS[wl_cl3_VIS>550]])
		fl_cl3 = np.concatenate([fl_cl3_UVB[wl_cl3_UVB<550],fl_cl3_VIS[wl_cl3_VIS>550]])

		wl_slab = np.concatenate([wl_slab_UVB_c[wl_slab_UVB_c<550],wl_slab_VIS_c[wl_slab_VIS_c>550]])
		fl_slab = np.concatenate([fl_slab_UVB_c[wl_slab_UVB_c<550],fl_slab_VIS_c[wl_slab_VIS_c>550]])

		"""
		PLOT PHOTOSPHERIC FEATURES
		"""
		eqw, err_eqw, mode, fwhm, lline = eqw_auto(wl_VIS,fl_VIS,670.78,size_cont=1.,plot='NO',mode='gauss',fwhm_out='Ja',lline_out='Ja')
		if self.min_chi_sq_cl3 == 'LM601':
			lline3 = 670.8
		elif self.min_chi_sq_cl3 == 'Sz94':
			lline3 = 670.3
		else:
			eqw3, err_eqw3, mode3, fwhm3, lline3 = eqw_auto(wl_cl3_VIS,fl_cl3_VIS,670.78,size_cont=1.,plot='NO',mode='gauss',fwhm_out='Ja',lline_out='Ja')

		shift = lline - lline3

		for i in range(len(usedEQWlines)):
			p1 = pl.figure(figsize=(10,9))
			f_in_norm = compute_flux_at_wl_std(wl_OBS,fl_OBS,usedEQWlines[i,0],interval=2*usedEQWlines[i,1])[0]

			f_tot_norm = compute_flux_at_wl_std(wl_cl3,(fl_cl3*self.min_chi_sq_K)+(fl_slab*self.min_chi_sq_H),usedEQWlines[i,0],interval=2*usedEQWlines[i,1])[0]

			f_clIII_norm = compute_flux_at_wl_std(wl_cl3,fl_cl3*self.min_chi_sq_K,usedEQWlines[i,0],interval=2*usedEQWlines[i,1])[0]

			pl.plot(wl_OBS-shift,fl_OBS/f_in_norm,'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
			pl.plot(wl_cl3,(self.min_chi_sq_K*fl_cl3+self.min_chi_sq_H*fl_slab)/f_tot_norm,'b')
			pl.plot(wl_cl3,self.min_chi_sq_K*fl_cl3/f_clIII_norm,'g',alpha =0.6)
			pl.plot(wl_slab,self.min_chi_sq_H*fl_slab/f_tot_norm,'c')

			pl.xlim(usedEQWlines[i,0]-(1*usedEQWlines[i,1]),usedEQWlines[i,0]+(1*usedEQWlines[i,1]))

			pl.ylim(0.01,1.3)
			pl.show()

			pl.savefig(self.PATH_OUT+'%s_clIII_%s_photosph_EQWLineAt_%snm.png'% (self.obj_in,self.min_chi_sq_cl3,usedEQWlines[i,0]))

	def plotTerms(self,close = False):
		pl.figure()
		pl.scatter(self.Chiterm1List.values(),self.Chiterm2List.values(),marker='.',alpha =0.3)
		sumArr = np.array(list(self.Chiterm1List.values())) + np.array(list(self.Chiterm2List.values()))
		minTot = np.min(sumArr)
		pl.scatter(np.array(list(self.Chiterm1List.values()))[sumArr == minTot],np.array(list(self.Chiterm2List.values()))[sumArr == minTot],marker='x',alpha =0.9)
		pl.xlabel('Cont Chi2')
		pl.ylabel('veil Chi2')
		pl.xscale('log')
		pl.yscale('log')
		pl.savefig(self.PATH_OUT+'%s_Terms.png'% (self.obj_in))
		pl.xlim(0.5*np.array(list(self.Chiterm1List.values()))[sumArr == minTot],10*np.array(list(self.Chiterm1List.values()))[sumArr == minTot])
		pl.ylim(0.5*np.array(list(self.Chiterm1List.values()))[sumArr == minTot],10*np.array(list(self.Chiterm1List.values()))[sumArr == minTot])
		pl.savefig(self.PATH_OUT+'%s_Terms_Zoom.png'% (self.obj_in))
		pl.show()
		if close:
			pl.close()

	def plotPaschen(self,CLIII =False,close = False):
		usedFeatures =self.usedFeatures
		classIIIreadIn = self.classIIIreadIn
		normWLandWidth = self.normWLandWidth

		wl_UVB = self.wl_UVB
		fl_UVB = self.fl_UVB_in/cardelli_extinction(wl_UVB*10.,self.min_chi_sq_Av, Rv=self.Rv)
		wl_VIS = self.wl_VIS
		fl_VIS = self.fl_VIS_in/cardelli_extinction(wl_VIS*10.,self.min_chi_sq_Av, Rv=self.Rv)
		self.filename_VIS =self.filename_VIS

		self.filename_NIR = self.filename_NIR
		self.fitsTab = self.fitsTab
		if self.filename_NIR == None:
			self.filename_NIR = path+'data_final/flux_%s_nir_tell.fits' % self.obj_in
		print('NIR path check:', self.filename_NIR)
		if os.path.isfile(self.filename_NIR):
			if self.fitsTab == False:
				wl_NIR,fl_NIR_in,hdr_NIR=spec_readspec(self.filename_NIR, 'hdr')	#the 'hdr' string is there to say that I want to save the header
			else:
				wl_NIR,fl_NIR_in,hdr_NIR=readspec_phase3(self.filename_NIR, hdr_out='YES')
			if self.perAA == True:
				fl_NIR_in = 10*fl_NIR_in
		else:
			print('No NIR spectrum')
			wl_NIR,fl_NIR_in = np.array([0 for x in range(100)]),np.array([0 for x in range(100)])

		fl_NIR = fl_NIR_in/cardelli_extinction(wl_NIR*10.,self.min_chi_sq_Av, Rv=self.Rv)

		if os.path.isfile(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out'):
			wl_slab,fl_slab = readcol_py3(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')
		else:
			f = open(PATH_ACC+'in.slab', 'w')
			outLine = self.min_chi_sq_T+'   '+self.min_chi_sq_Ne+'   '+self.min_chi_sq_tau
			f.write(outLine)
			f.close()
			os.chdir(PATH_ACC)
			os.system('./hydrogen_slab')
			os.chdir(PATH)
			wl_slab,fl_slab = readcol_py3(PATH_ACC+'results/continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')
		PATH_MY_CLASSIII = PATH+'/models_grid/RunnableGrid/'

		wl_cl3_UVB,fl_cl3_UVB,wl_cl3_VIS,fl_cl3_VIS,wl_cl3_NIR,fl_cl3_NIR,nameWls = pf.readMixClassIII(self.min_chi_sq_cl3,PATH_MY_CLASSIII,wlNorm =normWLandWidth[0])
		ind_uvb_3 = (wl_cl3_UVB <= max_wl_uvb)
		ind_vis_3 = (wl_cl3_VIS >= min_wl_vis)

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c= s['w'],s['f']
		else:
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c = s['w'],s['f']
			fl_slab_VIS_c = np.interp(wl_cl3_VIS,wl_slab_VIS_c,fl_slab_VIS_c )
			wl_slab_VIS_c = wl_cl3_VIS

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c= s['w'],s['f']
		else:
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c = s['w'],s['f']
			fl_slab_UVB_c = np.interp(wl_cl3_UVB,wl_slab_UVB_c,fl_slab_UVB_c )
			wl_slab_UVB_c = wl_cl3_UVB

		if CLIII == True:
			p1 = pl.figure(figsize=(10,9))
			pl.subplot(211)
			pl.plot(wl_UVB[self.ind_uvb],fl_UVB[self.ind_uvb],'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
			pl.title(self.obj_in)#+' - Chi$^2$= %0.3e' % (self.min_chi_sq))
			pl.ylabel('Flux')
			pl.axis([460,550,-1e-16,3.5*np.mean(fl_UVB[(wl_UVB < 460) & (wl_UVB > 440)])])
			pl.plot(wl_cl3_UVB[ind_uvb_3],self.min_chi_sq_K*fl_cl3_UVB[ind_uvb_3]+self.min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'b')
			pl.plot(wl_cl3_UVB[ind_uvb_3],self.min_chi_sq_K*fl_cl3_UVB[ind_uvb_3],'g',alpha =0.6)
			pl.plot(wl_cl3_UVB[ind_uvb_3],self.min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'c')

			pl.subplot(212)
			pl.plot(wl_VIS[self.ind_vis],fl_VIS[self.ind_vis],'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
			pl.xlabel('Wavelength [nm]')
			pl.ylabel('Flux')
			pl.axis([550,700,-1e-16,1.5*np.mean(fl_VIS[(wl_VIS < 650) & (wl_VIS > 600)])])
			pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'b')
			pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3],'g',alpha =0.6)
			pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'c')

			pl.tight_layout()
			pl.savefig(self.PATH_OUT+'%s_clIII_%s_other_%s.png'% (self.obj_in,self.min_chi_sq_cl3,nameWls))
			pl.show()

		else:
			wlFeat = (usedFeatures[:,0]+usedFeatures[:,1])/2
			Xrange = np.abs(usedFeatures[:,0] - wlFeat)
			features,errors = classIIIreadIn.getFeatsAtSpt_symetricErr(self.min_chi_sq_cl3)
			p1 = pl.figure(figsize=(10,9))
			fit_cont = np.zeros(len(usedFeatures))
			for i in range(len(usedFeatures)):
				fit_cont[i] = self.min_chi_sq_K*features[i]+self.min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeatures[i,0],usedFeatures[i,1])[0]
			stdTerm1 = (self.min_chi_sq_K*errors)**2

			gridFilePlot = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeUVB/interp.npz'
			classIIIFeatPlotting = pf.classIII(gridFilePlot)
			usedFeaturesPlotting = classIIIFeatPlotting.getUsedInterpFeat()
			featuresPlot,errorsPlot = classIIIFeatPlotting.getFeatsAtSpt_symetricErr(self.min_chi_sq_cl3)

			gridFilePlotVIS = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeVIS/interp.npz'
			classIIIFeatPlottingVIS = pf.classIII(gridFilePlotVIS)
			usedFeaturesPlottingVIS = classIIIFeatPlottingVIS.getUsedInterpFeat()
			featuresPlotVIS,errorsPlotVIS = classIIIFeatPlottingVIS.getFeatsAtSpt_symetricErr(self.min_chi_sq_cl3)

			usedFeaturesPlotting = np.concatenate((usedFeaturesPlotting,usedFeaturesPlottingVIS))
			featuresPlot = np.concatenate((featuresPlot,featuresPlotVIS))
			errorsPlot = np.concatenate((errorsPlot,errorsPlotVIS))

			wlFeatPlot = (usedFeaturesPlotting[:,0]+usedFeaturesPlotting[:,1])/2
			XrangePlot = np.abs(usedFeaturesPlotting[:,0] - wlFeatPlot)

			fit_contPlot = np.zeros(len(usedFeaturesPlotting))
			for i in range(len(usedFeaturesPlotting[:,0])):
				fit_contPlot[i] = self.min_chi_sq_K*featuresPlot[i]+self.min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeaturesPlotting[i,0],usedFeaturesPlotting[i,1])[0]
			stdTerm1plot = (self.min_chi_sq_K*errorsPlot)**2
			stdTerm2plot = 0#(stddev_cont_dered**2)#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not taken into account!!!
			fit_stdPlot = np.sqrt(stdTerm1plot +stdTerm2plot)

			stdTerm2 = 0#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not used for the plot, this is a hold over!!!
			fit_std = np.sqrt(stdTerm1 +stdTerm2)
			pl.subplot(211)
			pl.plot(wl_UVB[self.ind_uvb],fl_UVB[self.ind_uvb],'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
			pl.title(self.obj_in)#+' - Chi$^2$= %0.3e' % (self.min_chi_sq))
			pl.ylabel('Flux')
			pl.axis([460,550,-1e-16,3.5*np.mean(fl_UVB[(wl_UVB < 460) & (wl_UVB > 440)])])
			pl.plot(wl_cl3_UVB[ind_uvb_3],self.min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'c')
			pl.errorbar(wlFeat,self.min_chi_sq_K*features,self.min_chi_sq_K*errors,xerr = Xrange,fmt='.',c='g',zorder = 10)
			pl.errorbar(wlFeat,fit_cont,yerr=fit_std,xerr= Xrange,fmt='.',c='b',zorder = 10)

			pl.plot(wlFeatPlot,self.min_chi_sq_K*featuresPlot ,c='tab:olive',zorder = 10)
			pl.plot(wlFeatPlot,fit_contPlot ,c='tab:cyan',zorder = 10)

			pl.subplot(212)
			pl.plot(wl_VIS[self.ind_vis],fl_VIS[self.ind_vis],'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
			pl.xlabel('Wavelength [nm]')
			pl.ylabel('Flux')
			pl.axis([550,700,-1e-16,1.5*np.mean(fl_VIS[(wl_VIS < 650) & (wl_VIS > 600)])])
			pl.errorbar(wlFeat,self.min_chi_sq_K*features,self.min_chi_sq_K*errors,xerr = Xrange,fmt='.',c='g',zorder = 10)
			pl.errorbar(wlFeat,fit_cont,yerr=fit_std,xerr= Xrange,fmt='.',c='b',zorder = 10)

			pl.plot(wlFeatPlot,self.min_chi_sq_K*featuresPlot ,c='tab:olive',zorder = 10)
			pl.plot(wlFeatPlot,fit_contPlot ,c='tab:cyan',zorder = 10)

			pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'c')

			pl.tight_layout()
			pl.savefig(self.PATH_OUT+'%s_clIII_%s_other_InterpFeat.png'% (self.obj_in,self.min_chi_sq_cl3))
			pl.show()
		if close:
			pl.close()



	def plotAll(self,CLIII = False,smooth = False, close = False):
		self.best_chi_sq = self.best_chi_sq
		self.min_chi_sq = self.min_chi_sq
		self.min_chi_sq_cl3 = self.min_chi_sq_cl3
		self.min_chi_sq_Av = self.min_chi_sq_Av
		self.min_chi_sq_T = self.min_chi_sq_T
		self.min_chi_sq_Ne = self.min_chi_sq_Ne
		self.min_chi_sq_tau = self.min_chi_sq_tau
		self.min_chi_sq_H = self.min_chi_sq_H
		self.min_chi_sq_K = self.min_chi_sq_K
		self.Av_list =self.Av_list
		self.PATH_OUT =self.PATH_OUT
		self.obj_in  = self.obj_in
		self.ind_uvb = self.ind_uvb
		self.ind_vis = self.ind_vis
		self.usedFeatures =self.usedFeatures
		self.classIIIreadIn = self.classIIIreadIn
		self.normWLandWidth = self.normWLandWidth

		self.wl_UVB = self.wl_UVB
		fl_UVB = self.fl_UVB_in/cardelli_extinction(self.wl_UVB*10.,self.min_chi_sq_Av, Rv=self.Rv)
		self.wl_VIS = self.wl_VIS
		fl_VIS = self.fl_VIS_in/cardelli_extinction(self.wl_VIS*10.,self.min_chi_sq_Av, Rv=self.Rv)
		self.filename_VIS =self.filename_VIS

		# load NIR file!
		self.filename_NIR = self.filename_NIR
		self.fitsTab = self.fitsTab
		if self.filename_NIR == None:
			self.filename_NIR = path+'data_final/flux_%s_nir_tell.fits' % self.obj_in
		if os.path.isfile(self.filename_NIR):
			if self.fitsTab == False:
				wl_NIR,fl_NIR_in,hdr_NIR=spec_readspec(self.filename_NIR, 'hdr')	#the 'hdr' string is there to say that I want to save the header
			else:
				wl_NIR,fl_NIR_in,hdr_NIR=readspec_phase3(self.filename_NIR, hdr_out='YES')
			if self.perAA == True:
				fl_NIR_in = 10*fl_NIR_in
		else:
			print('No NIR spectrum')
			wl_NIR,fl_NIR_in = np.array([0 for x in range(100)]),np.array([0 for x in range(100)])
			#sys.exit(1)

		fl_NIR = fl_NIR_in/cardelli_extinction(wl_NIR*10.,self.min_chi_sq_Av, Rv=Rv)

		#Load Slab!!
		if os.path.isfile(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out'):
			wl_slab,fl_slab = readcol_py3(PATH_SLAB+'continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')
		else:
		# otherwise, create it and then read it
			# first, write the input file
			f = open(PATH_ACC+'in.slab', 'w')
			outLine = self.min_chi_sq_T+'   '+self.min_chi_sq_Ne+'   '+self.min_chi_sq_tau
			f.write(outLine)
			#f.write(string.join([self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau],'   '))
			f.close()
			# run the C++ slab model program using the best fit parameters to calculate the slab model from 50 nm to 2477 nm (whole range)
			os.chdir(PATH_ACC)
			os.system('./hydrogen_slab')
			os.chdir(PATH)
			# read the result of the C++ program
			wl_slab,fl_slab = readcol_py3(PATH_ACC+'results/continuum_tot_T'+self.min_chi_sq_T+'_ne'+self.min_chi_sq_Ne+'tau'+self.min_chi_sq_tau+'.out',2,format='F,F')
		PATH_MY_CLASSIII = PATH+'/models_grid/RunnableGrid/'

		wl_cl3_UVB,fl_cl3_UVB,wl_cl3_VIS,fl_cl3_VIS,wl_cl3_NIR,fl_cl3_NIR,nameWls = pf.readMixClassIII(self.min_chi_sq_cl3,PATH_MY_CLASSIII,wlNorm =self.normWLandWidth[0])
		ind_uvb_3 = (wl_cl3_UVB <= max_wl_uvb)
		ind_vis_3 = (wl_cl3_VIS >= min_wl_vis)

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c= s['w'],s['f']
		else:
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c = s['w'],s['f']
			fl_slab_VIS_c = np.interp(wl_cl3_VIS,wl_slab_VIS_c,fl_slab_VIS_c )
			wl_slab_VIS_c = wl_cl3_VIS

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c= s['w'],s['f']
			#wl_slab,fl_slab = wl_slab_UVB_c,fl_slab_UVB_c
		else:
			#fluxcon = FluxConservingResampler()
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (self.min_chi_sq_T,self.min_chi_sq_Ne,self.min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c = s['w'],s['f']
			#spec = Spectrum1D(spectral_axis=wl_slab_UVB_c, flux=fl_slab_UVB_c)
			fl_slab_UVB_c = np.interp(wl_cl3_UVB,wl_slab_UVB_c,fl_slab_UVB_c )
			wl_slab_UVB_c = wl_cl3_UVB
			#,fl_slab_UVB_c = np.array(resampledSpecUVB.spectral_axis/u.Unit('nm')), np.array(resampledSpecUVB.flux/u.Unit('erg cm-2 s-1 nm-1'))

		p1 = pl.figure(figsize=(10,7))
		#pl.rcParams.update({'font.size': 13})
		if smooth == False:
			pl.plot (self.wl_UVB[self.ind_uvb],fl_UVB[self.ind_uvb],'k')
			pl.plot(self.wl_VIS[self.ind_vis],fl_VIS[self.ind_vis],'k')
			pl.plot(wl_NIR,fl_NIR,'k')
		else:

			##############
			#
			##############
			wl_UVB_smooth = self.wl_UVB[self.ind_uvb][::12]
			fl_UVB_smooth = spectrum_resample(fl_UVB,self.wl_UVB,wl_UVB_smooth)
			wl_VIS_smooth = self.wl_VIS[self.ind_vis][::12]
			fl_VIS_smooth = spectrum_resample(fl_VIS,self.wl_VIS,wl_VIS_smooth)
			wl_NIR_smooth = wl_NIR[::12 ]
			fl_NIR_smooth = spectrum_resample(fl_NIR,wl_NIR,wl_NIR_smooth)
			pl.plot(wl_UVB_smooth,fl_UVB_smooth,'k')
			pl.plot(wl_VIS_smooth,fl_VIS_smooth,'k')
			pl.plot(wl_NIR_smooth,fl_NIR_smooth,'k')

		if CLIII == True:
			"""
			PLOT WHOLE SPECTRUM
			"""

			##############
			#
			##############

			pl.title(self.obj_in)#+' - Chi$^2$= %0.3e' % (self.min_chi_sq))
			pl.xlabel('Wavelength [nm]')
			pl.ylabel('Flux')
			pl.axis([330,2048,0.1*np.median(fl_UVB[(self.wl_UVB < 450) & (self.wl_UVB > 400)]),4*np.median(fl_VIS[(self.wl_VIS < 820) & (self.wl_VIS > 800)])])
			pl.plot(wl_cl3_UVB[ind_uvb_3],self.min_chi_sq_K*fl_cl3_UVB[ind_uvb_3]+self.min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'b',alpha =0.5)
			pl.plot(wl_cl3_UVB[ind_uvb_3],self.min_chi_sq_K*fl_cl3_UVB[ind_uvb_3],'g',alpha =0.6)
			pl.plot(wl_cl3_UVB[ind_uvb_3],self.min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'r')

			pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'b',alpha =0.5)
			#print(self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3])
			pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3],'g',alpha =0.6)
			pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'r')

			pl.plot(wl_cl3_NIR,self.min_chi_sq_K*fl_cl3_NIR,'g',alpha =0.6)

			pl.xscale('log')
			pl.yscale('log')

			pl.tight_layout()
			pl.savefig(self.PATH_OUT+'%s_clIII_%s_ALL_%s.png'% (self.obj_in,self.min_chi_sq_cl3,nameWls))
			pl.show()

		else:
			"""
			PLOT WHOLE SPECTRUM WITH INTERP FEAT
			"""
			wlFeat = (self.usedFeatures[:,0]+self.usedFeatures[:,1])/2
			Xrange = np.abs(self.usedFeatures[:,0] - wlFeat)
			features,errors = self.classIIIreadIn.getFeatsAtSpt_symetricErr(self.min_chi_sq_cl3)
			#p1 = pl.figure(5,figsize=(10,9))
			fit_cont = np.zeros(len(self.usedFeatures))
			for i in range(len(self.usedFeatures)):
				fit_cont[i] = self.min_chi_sq_K*features[i]+self.min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,self.usedFeatures[i,0],self.usedFeatures[i,1])[0]
			stdTerm1 = (self.min_chi_sq_K*errors)**2

			gridFilePlot = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeUVB/interp.npz'
			classIIIFeatPlotting = pf.classIII(gridFilePlot)
			usedFeaturesPlotting = classIIIFeatPlotting.getUsedInterpFeat()
			featuresPlot,errorsPlot = classIIIFeatPlotting.getFeatsAtSpt_symetricErr(self.min_chi_sq_cl3)




			gridFilePlotVIS = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeVIS/interp.npz'
			classIIIFeatPlottingVIS = pf.classIII(gridFilePlotVIS)
			usedFeaturesPlottingVIS = classIIIFeatPlottingVIS.getUsedInterpFeat()
			featuresPlotVIS,errorsPlotVIS = classIIIFeatPlottingVIS.getFeatsAtSpt_symetricErr(self.min_chi_sq_cl3)

			usedFeaturesPlotting = np.concatenate((usedFeaturesPlotting,usedFeaturesPlottingVIS))
			featuresPlot = np.concatenate((featuresPlot,featuresPlotVIS))
			errorsPlot = np.concatenate((errorsPlot,errorsPlotVIS))

			wlFeatPlot = (usedFeaturesPlotting[:,0]+usedFeaturesPlotting[:,1])/2
			XrangePlot = np.abs(usedFeaturesPlotting[:,0] - wlFeatPlot)

			fit_contPlot = np.zeros(len(usedFeaturesPlotting))
			for i in range(len(usedFeaturesPlotting[:,0])):
				fit_contPlot[i] = self.min_chi_sq_K*featuresPlot[i]+self.min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeaturesPlotting[i,0],usedFeaturesPlotting[i,1])[0]
			stdTerm1plot = (self.min_chi_sq_K*errorsPlot)**2
			stdTerm2plot = 0#(stddev_cont_dered**2)#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not taken into account!!!
			fit_stdPlot = np.sqrt(stdTerm1plot +stdTerm2plot)

			stdTerm2 = 0#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not used for the plot, this is a hold over!!!
			fit_std = np.sqrt(stdTerm1 +stdTerm2)
			#p1 = pl.figure(figsize=(10,7))
			##############
			#
			##############

			pl.title(self.obj_in)#+' - Chi$^2$= %0.3e' % (self.min_chi_sq))
			pl.xlabel('Wavelength [nm]')
			pl.ylabel('Flux')
			pl.axis([330,2048,0.1*np.median(fl_UVB[(self.wl_UVB < 450) & (self.wl_UVB > 400)]),4*np.median(fl_VIS[(self.wl_VIS < 820) & (self.wl_VIS > 800)])])
			#pl.loglog(wl_cl3_UVB[ind_uvb_3],self.min_chi_sq_K*fl_cl3_UVB[ind_uvb_3]+self.min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'b')
			#pl.loglog(wl_cl3_UVB[ind_uvb_3],self.min_chi_sq_K*fl_cl3_UVB[ind_uvb_3],'g')
			pl.plot(wl_cl3_UVB[ind_uvb_3],self.min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'r')

			#pl.loglog(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'b')
			#pl.loglog(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_K*fl_cl3_VIS[ind_vis_3],'g')
			pl.plot(wl_cl3_VIS[ind_vis_3],self.min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'r')

			#pl.loglog(wl_cl3_NIR,self.min_chi_sq_K*fl_cl3_NIR,'g')

			#Used feat
			pl.errorbar(wlFeat,self.min_chi_sq_K*features,self.min_chi_sq_K*errors,xerr = Xrange,fmt='.',c='g',alpha =0.6,zorder = 10)
			pl.errorbar(wlFeat,fit_cont,yerr=fit_std,xerr= Xrange,fmt='.',c='b',zorder = 10)

			#Additional features
			pl.plot(wlFeatPlot,self.min_chi_sq_K*featuresPlot ,c='tab:olive',zorder = 10)
			pl.plot(wlFeatPlot,fit_contPlot ,c='tab:cyan',zorder = 10)


			#pl.errorbar(wlFeatPlot_NIR,self.min_chi_sq_K*featuresPlot_NIR,self.min_chi_sq_K*errorsPlot_NIR,xerr = XrangePlot_NIR,fmt='.',c='tab:olive',zorder = 10)
			#pl.errorbar(wlFeatPlot_NIR,fit_contPlot_NIR,yerr=fit_stdPlot_NIR,xerr= XrangePlot_NIR,fmt='.',c='tab:cyan',zorder = 10)

			pl.xscale('log')
			pl.yscale('log')

			pl.tight_layout()
			pl.show()
			pl.savefig(self.PATH_OUT+'%s_clIII_%s_ALL_INTERP_FEAT.png'% (self.obj_in,self.min_chi_sq_cl3))
		if close:
			pl.close()

	"""
	CHI2 ANALYSIS
	"""

	def Chi2SpT(self,close =False):
		self.chi_sq = self.chi_sq
		self.H_fin = self.H_fin
		self.K_fin = self.K_fin
		self.cl3_in_list = self.cl3_in_list
		self.obj_in =self.obj_in
		self.min_chi_sq_cl3 = self.min_chi_sq_cl3
		if len(self.cl3_in_list) > 1:
			chi_sq_distr_cl3(self.chi_sq,self.H_fin,self.K_fin,self.cl3_in_list,PATH)
			tit = '%s - %s' % (self.obj_in,self.min_chi_sq_cl3)
			pl.title(tit)
			pl.tight_layout()
			pl.savefig(self.PATH_OUT+'%s_clIII_%s_cl3_chi2.png'% (self.obj_in,self.min_chi_sq_cl3))
			pl.show()
		if close:
			pl.close()

	def Chi2Av(self,close =False):
		self.chi_sq = self.chi_sq
		self.H_fin = self.H_fin
		self.K_fin = self.K_fin
		self.Av_list = self.Av_list
		self.min_chi_sq_Av = self.min_chi_sq_Av
		self.min_chi_sq_cl3 = self.min_chi_sq_cl3
		self.obj_in =self.obj_in
		if len(self.Av_list) > 1:
			chi_sq_distr_Av(self.chi_sq,self.H_fin,self.K_fin,self.Av_list,self.min_chi_sq_Av)
			tit = '%s - %s' % (self.obj_in,self.min_chi_sq_cl3)
			pl.title(tit)
			pl.tight_layout()
			pl.savefig(self.PATH_OUT+'%s_clIII_%s_Av_chi2.png'% (self.obj_in,self.min_chi_sq_cl3))
			pl.show()
		if close:
			pl.close()

	def Chi2AvAndSpT(self,close =False):
		self.chi_sq = self.chi_sq
		self.H_fin = self.H_fin
		self.K_fin = self.K_fin
		self.cl3_in_list = self.cl3_in_list
		self.Av_list = self.Av_list
		self.obj_in =self.obj_in
		self.min_chi_sq_cl3 = self.min_chi_sq_cl3
		if len(self.cl3_in_list) > 2 and len(self.Av_list) > 2:
			chi_sq_distr_cl3_Av(self.chi_sq,self.H_fin,self.K_fin,self.cl3_in_list,self.Av_list,PATH)
			tit = '%s - %s' % (self.obj_in,self.min_chi_sq_cl3)
			pl.title(tit)
			pl.savefig(self.PATH_OUT+'%s_clIII_%s_Av_SpT_chi2.png'% (self.obj_in,self.min_chi_sq_cl3))
		if close:
			pl.close()

	def Chi2AvAndSpT_Posterior(self,close =False):
		self.chi_sq = self.chi_sq
		self.H_fin = self.H_fin
		self.K_fin = self.K_fin
		self.cl3_in_list = self.cl3_in_list
		self.Av_list = self.Av_list
		self.obj_in =self.obj_in
		self.min_chi_sq_cl3 = self.min_chi_sq_cl3
		if len(self.cl3_in_list) > 2 and len(self.Av_list) > 2:
			posterior_distr_cl3_Av(self.chi_sq,self.H_fin,self.K_fin,self.cl3_in_list,self.Av_list,PATH)
			tit = '%s - %s' % (self.obj_in,self.min_chi_sq_cl3)
			pl.title(tit)
			pl.savefig(self.PATH_OUT+'%s_clIII_%s_Av_SpT_chi2_posterior.png'% (self.obj_in,self.min_chi_sq_cl3))
		if close:
			pl.close()
