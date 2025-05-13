#!/usr/bin/env python

			############################################
			#		the_ultimate_fitter.py			   #
			# FIT BOTH SPT AND ACCRETION IN CLASS IIs  #
			############################################



import numpy as np
import matplotlib as mp
import pylab as pl

import time
import sys
import string
import os

PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH)
sys.path = [PATH+'/FrappeHelper/'] + sys.path
print(PATH)

from spec_readspec import *
#from linfit import *
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
#from specutils.manipulation import FluxConservingResampler
from astropy import units as u
from spec_phase3 import *
import ray
ray.shutdown()
import PhotFeatures_Ray as pf

####################
### this is the good one!!!!
####################


				############################
				# 	  PATH DEFINITION	   #
				############################

# local paths!!!

PATH_SLAB = PATH+'/models_grid/slab_models_grid/'
#PATH_CLASSIII = PATH+'models_grid/ClassIII_final/'
PATH_SLAB_RESAMPLED_SAV = PATH+'/models_grid/Slab_SAV_TWA7only/'
#gridFile = '/Users/rclaes/python/functions/MyFitter/earlyK_norm731_200p_500iter_rad2.5_deg2_LongWL_improved/GoodOne.npz'
gridFile = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_500iter_rad2.5_deg2_LongWL_improved/interp.npz'
				############################
				# 		 FUNCTIONS		   #
				############################



from utils_fitter_py3 import *






@ray.remote
def main_process(self,cl3_spt):

	### load in the interpolated grid at the spt requested IN SPT CODE!!!
	features,errors = self.classIIIreadIn.getFeatsAtSpt_symetricErr(cl3_spt)


	# ------------------------------
	# 3) SLAB MODELS
	# ------------------------------
	#list of all the combinations of T,Ne and tau of the slab I'm using
	T_slab = ['5000','5500','6000','6500','7000','7500','7750','8000','8250','8500','8750','9000','9250','9500','9750',\
	'10000','10500']#,'11000']
	Ne_slab = ['1e+11','1e+12','1e+13','3e+13','5e+13','7e+13','1e+14','5e+14','1e+15','1e+16']
	tau_slab = ['0.01', '0.05','0.1','0.3','0.5','0.75','1','3','5']
	###comment this!!!!
	#T_slab = ['7750','8000','8250']
	#Ne_slab = ['1e+14','5e+14','1e+15']
	#tau_slab = ['0.1','0.3','0.5','0.75','1']

	#for on all the slab models
	chi_sq = {}
	chi_sq_max = 1.e15
	H_fin = {}
	K_fin = {}
	file_read = 0
	p1 = None

	### Concat the observations for ease of use
	wlObs = np.concatenate([self.wl_UVB[self.wl_UVB<550],self.wl_VIS[self.wl_VIS>550]])#,wn10[wn10>1020]])
	flObs = np.concatenate([self.fl_UVB_in[self.wl_UVB<550],self.fl_VIS_in[self.wl_VIS>550]])#,fn10[wn10>1020]])

	### Extract the used features from the obs outside of the Av loop, no use derredening the entire spectrum
	obs_cont,stddev_cont = np.zeros(len(self.usedFeatures)),np.zeros(len(self.usedFeatures))

	### it would be nice to get rid of this llop, maybe later
	for i in range(len(self.usedFeatures)):
		obs_cont[i],stddev_cont[i] = compute_flux_inRange(wlObs,flObs,self.usedFeatures[i,0],self.usedFeatures[i,1])
	normWL, normWlHalfWidth = self.normWLandWidth[0], self.normWLandWidth[1]
	obs_cont_CLIIIScaling, stddev_cont_CLIIIScaling = compute_flux_at_wl_std(wlObs,flObs,normWL,interval=normWlHalfWidth*2)
	obs_cont_360, stddev_cont_360 = compute_flux_at_wl_std(wlObs,flObs,355,interval=6)
	for Av_iter in self.Av_list:
	#Reddening correction

		### Here I assume the deredening factor is ~ constant in the range of a used features
		### I could redo this at some comp. cost to use the values in each feature BC_fit_in
		#obs_cont_dered,stddev_cont_dered = np.zeros(len(usedFeatures)),np.zeros(len(usedFeatures))

		wl_dered = (self.usedFeatures[:,0]+self.usedFeatures[:,1])/2
		obs_cont_dered = obs_cont/cardelli_extinction(wl_dered*10.,Av_iter, Rv=Rv)
		stddev_cont_dered = stddev_cont/cardelli_extinction(wl_dered*10.,Av_iter, Rv=Rv)
			### also for the norm WL!

		obs_cont_CLIIIScaling_dered = obs_cont_CLIIIScaling/cardelli_extinction(np.array([normWL*10.]),Av_iter, Rv=Rv)
		stddev_cont_CLIIIScaling_dered = stddev_cont_CLIIIScaling/cardelli_extinction(np.array([normWL*10.]),Av_iter, Rv=Rv)

		obs_cont_360_dered = obs_cont_360/cardelli_extinction(np.array([355*10.]),Av_iter, Rv=Rv)
	#						--------- GOODNESS OF FIT CALCULATION ---------
	#	FIT PARAMETERS
			#CONTINUUM AT ~360 nm - if highly accreting, slab >> cl3
		#riks way of computing the class III and
		###
		### this one is used for the scaling!!!
		###
		if 355 not in wl_dered:
			print('the feature to scale the acc slab model is not included!!')
		cl3_cont_360 = features[wl_dered == 355][0]
		cl3_stddev_cont_360 = errors[wl_dered == 355][0]

		for T_iter in T_slab:
			for Ne_iter in Ne_slab:
				for tau_iter in tau_slab:
					### here i pick use arbitrary class III to read in the slab model values

					wl_slab_UVB,fl_slab_UVB = read_slab_sav_RC(T_iter,Ne_iter,tau_iter,self.cl3_in_toSelectModel,'UVB',PATH_SLAB_RESAMPLED_SAV)
					wl_slab_VIS,fl_slab_VIS = read_slab_sav_RC(T_iter,Ne_iter,tau_iter,self.cl3_in_toSelectModel,'VIS',PATH_SLAB_RESAMPLED_SAV)
					wl_slab = np.concatenate([wl_slab_UVB[wl_slab_UVB<550],wl_slab_VIS[wl_slab_VIS>550]])#,wn10[wn10>1020]])
					fl_slab = np.concatenate([fl_slab_UVB[wl_slab_UVB<550],fl_slab_VIS[wl_slab_VIS>550]])#,fn10[wn10>1020]])

		# ------------>  model = K*CL3 + H*slab ---> Obs = best_model

		# 							--------- GOODNESS OF FIT CALCULATION ---------
		#			FIT PARAMETERS
						#CONTINUUM AT ~360 nm - if highly accreting, slab >> cl3
					slab_cont_360 = compute_cont_360_nostd(wl_slab_UVB,fl_slab_UVB)
					# 	#CONTINUUM AT ~460 nm
					# slab_cont_460 = compute_cont_460(wl_slab_UVB,fl_slab_UVB)
						#CONTINUUM AT ~710 nm - here there should be almost no slab emission
					### I changed this to 751 nm so that the bol correction of HH14+adjustment can be applied
					slab_cont_CLIIIScaling = compute_flux_at_wl_nostd(wl_slab_VIS,fl_slab_VIS,normWL,interval=normWlHalfWidth*2)


		#  ---  FIRST GUESS OF THE NORMALIZATION CONSTANTS ---
		#	The first guess of the CLASS III NORMALIZATION is given by the fact that at ~700 nm there should be almost no
		#		emission from the slab, but only from the class III.

					K_try = obs_cont_CLIIIScaling_dered*1 ###1 is the flux at the normalization wl in the class III interpolation

		#	The first guess on the SLAB MODEL NORMALIZATION is due to the fact that at ~360 nm most of the flux should arise
		#		from the slab, if the observed object is heavily accreting, and by the fact that the classIII is already
		#		constrained by the normalization at ~700 nm
					H_try = (obs_cont_360_dered-K_try*cl3_cont_360)/slab_cont_360
					# print 'FIRST GUESS ON THE NORMALIZATION CONSTANTS: %.4e, %.4f' % (H_try,K_try)

		#  ---	ITERATION ON THE NORMALIZATION CONSTANTS ---
		#		Recalculate K using the H_try derived before, then iterate to get again H until convergence (with max_iter = 100)
					K_old = K_try
					K_iter = (obs_cont_CLIIIScaling_dered-slab_cont_CLIIIScaling*H_try)#/cl3_cont_751
					H_iter = H_try
		# 		check that the normalization constants are > 0, otherwise it means that this model is not good
					if K_iter < 0 or H_try < 0:
						print( 'NOT POSSIBLE')
						if H_try <0:
							print('initial H already negative' )
						if K_iter<0:
							print('initial K already negative' )
						break
		#		iteration to derive H again
					max_iter = 100 # maximum number of possible iterations
					min_discr = 0.005e-14 # minimum difference between two iterate values of K. when this value is reached, stop iteration
					iterations = 0
					# H_iter = H_try
					while abs(K_iter-K_old)/np.mean([K_iter,K_old]) > min_discr and iterations <= max_iter:
						# print 'deltaK = %.4f - iteration #%i' % (abs(K_iter-K_old)/np.mean([K_iter,K_old]),iterations)
						K_old = K_iter
						H_iter = (obs_cont_360_dered-K_iter*cl3_cont_360)/slab_cont_360
						K_iter = (obs_cont_CLIIIScaling_dered-slab_cont_CLIIIScaling*H_iter)/1 ###1 is the flux at the normalization wl in the class III interp grid
						iterations+=1
					# print 'FINAL NORMALIZATION CONSTANTS: %.4e, %.4f'% (H_iter,K_iter)
		#		store the best values
					K = K_iter
					H = H_iter
		# 		check that the normalization constants are > 0, otherwise it means that this model is not good
					if K < 0 or H <= 0:
						print( 'NOT POSSIBLE')
						break

		#		BEST-FIT PARAMETERS of the fitting spectrum (model = K*CL3 + H*slab)
						#CONTINUUM AT ~360 nm - if highly accreting, slab >> cl3
					fit_cont = np.zeros(len(self.usedFeatures))
					for i in range(len(self.usedFeatures)):
						fit_cont[i] = K*features[i]+H*compute_flux_inRange(wl_slab,fl_slab,self.usedFeatures[i,0],self.usedFeatures[i,1])[0]
					stdTerm1 = (K*errors)**2
					stdTerm2 = (stddev_cont_dered**2)
					fit_std = np.sqrt(stdTerm1 +stdTerm2)


					chi_sq_temp =  np.sum(((fit_cont - obs_cont_dered) / fit_std)**2)



		#save the results of the Chi2 calculation with a dictionary, where the key is 'T-Ne-tau' and the value is the Chi2
					#chi_sq[string.join([cl3_in,str(Av_iter),T_iter,Ne_iter,tau_iter],'/')] = chi_sq_temp
					key = str(cl3_spt)+'/'+str(Av_iter)+'/'+T_iter+'/'+Ne_iter+'/'+tau_iter
					chi_sq[key] = chi_sq_temp

					H_fin[key] = H
					K_fin[key] = K
	#print( 'Execution time:', time.time() - time_init, " seconds")
	return [chi_sq, H_fin, K_fin]


'''				############################
				# 		   SCRIPT		   #
				############################
'''

# DEFAULT PARAMETERS THAT CAN BE CHANGED WITH THE OPTIONS

#path = '/Users/rclaes/Work/StructuredSources/Fits2014_shortWl/'
#obj_in = 'CIDA1'

#filename_UVB = None
#filename_VIS = None
#filename_NIR = None

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

#check input parameters
####
class Fit():
	def __init__(self,f_uvb = None,f_vis = None,f_nir = None,obj_in ='Unknown',dist=None ,dirOut = None,**kwargs):

		max_wl_uvb = 552.
		min_wl_vis = 552.
		max_wl_vis = 1019.
		#self.cl3_in = [3,3.25,3.5]

		self.filename_UVB = f_uvb
		self.filename_VIS = f_vis
		self.filename_NIR = f_nir
		self.dist_pc = dist
		if dist == None:
			print( sys.argv[0],': dist=',dist)
			sys.exit(1)
		fitsTab = False
		self.perAA = False
		self.Rv = 3.1
		for option, value in kwargs.items():
			#print("here")
			#print(option)
			#print(sys.argv[1])
			#print("here")
				#print(obj_in)
			if option == 'spt':
				# cl3_in = sys.argv[1]; del sys.argv[1]
				cl3_in = value  # Dec, 2nd 2014 - in this way I can give as input a sequence of names and becomes a list
			elif option == 'max_u' or option == '--max_wl_uvb':
				self.max_wl_uvb = value
			elif option == 'min_v' or option == '--min_wl_vis':
				self.min_wl_vis = value
			elif option == 'max_v' or option == '--max_wl_vis':
				self.max_wl_vis = value
			elif option == 'dist' or option == '--distance':
				self.dist_pc = value
			elif option == 'Av' or option == '--Av_fix':
				# Av = sys.argv[1]; del sys.argv[1]
				Av = value # Dec, 4th 2014
			elif option == 'Rv' or option == '--reddening_law':
				self.Rv = value
			###################
			# additional options added by RC
			###################
			elif option == 'perAA':
				self.perAA = value
			elif option == 'fitsTab':
				self.fitsTab = value
			else:
				print( sys.argv[0],': option=',option)
				sys.exit(1)


		#gridFile = '/Users/rclaes/python/functions/MyFitter/earlyK_norm731_200p_1000iter_rad2.5_NoInstrRes_SignalNonZero/GoodOne.npz'

		self.classIIIreadIn = pf.classIII(gridFile)
		self.usedFeatures = self.classIIIreadIn.getUsedInterpFeat()
		self.normWLandWidth = self.classIIIreadIn.getUsedNormWl()

		### this is put outside the loop since the wls are an input to retrieve the slab models!!
		### i use an arbitrary class III to retrieve the slab templates
		self.cl3_in_toSelectModel = 'TWA7'

		# CREATE OUTPUT FOLDER FROM WHAT YOU DECLARED
		now = time.localtime()[0:6]
		if dirOut != None:
			path = dirOut
		else:
			path = os.getcwd()
		self.path = path
		self.PATH_OUT = path+obj_in+'_%4d-%02d-%02d_%02d.%02d.%02d' % now
		print("here")
		print(self.PATH_OUT)
		os.mkdir(self.PATH_OUT)
		self.PATH_OUT = self.PATH_OUT+'/'


		# ------------------------------
		# 1) read the input spectrum
		# ------------------------------
		#read the spectrum and save the flux and wavelengths and header
		if self.fitsTab == False:
			self.wl_UVB,self.fl_UVB_in,hdr_UVB=spec_readspec(self.filename_UVB, 'hdr')	#the 'hdr' string is there to say that I want to save the header
		else:
			self.wl_UVB,self.fl_UVB_in,hdr_UVB=readspec_phase3(self.filename_UVB,hdr_out='y')

		self.ind_uvb = (self.wl_UVB <= max_wl_uvb) #select only the part of the spectrum that is nice(\lambda<550 nm in the UVB)
		if self.fitsTab == False:
			self.wl_VIS,self.fl_VIS_in,hdr_VIS=spec_readspec(self.filename_VIS, 'hdr')	#the 'hdr' string is there to say that I want to save the header
		else:
			self.wl_VIS,self.fl_VIS_in,hdr_VIS=readspec_phase3(self.filename_VIS,hdr_out='Y')
		self.ind_vis = (self.wl_VIS >= min_wl_vis) #& (wave_VIS <=1024) #select only the part of the spectrum that is nice(\lambda>550 nm in the VIS)

		if self.perAA == True:
			self.fl_UVB_in = self.fl_UVB_in*10
			self.fl_VIS_in = self.fl_VIS_in*10
		# name the object!
		print(obj_in)
		if obj_in != None:
			self.obj_in = obj_in
		else:
			self.obj_in = hdr_UVB['OBJECT'].replace(" ", "")


		# --------------------------------
		# 1b) EXTINCTION CORRECTION - If input Av = None, do a for with values of Av from 0 to 10, else use the value in input to deredden
		# --------------------------------
		# Remember that cardelli_extinction is in AA and my spectra in nm.
		if Av != None:
			print( Av)
			self.Av_list = np.array(Av,dtype=np.float32)
			print(self.Av_list)
		else:
			# Av_list = np.linspace(0,0.5,6)
			# Av_list = np.linspace(0,2,21)
			self.Av_list = np.concatenate((np.linspace(0,1.5,16),np.linspace(1.5,3,4)))	#[  0. ,   0.1,   0.2,   0.3,   0.4,   0.5,   0.6,   0.7,   0.8, 0.9,   1. ,   1.1,   1.2,   1.3,   1.4,   1.5,   1.6,   1.7,1.8,   1.9,   2. ,   2.1,   2.2,   2.3,   2.4,   2.5,   2.6,2.7,   2.8,   2.9,   3. ,   3. ,   4. ,   5. ,   6. ,   7. ,8. ,   9. ,  10. ]
			# Av_list = np.concatenate((np.linspace(0,3,19),np.linspace(3,10,8)))	#[0.,0.16666667,0.33333333,0.5,0.66666667,0.83333333,1.,1.16666667,1.33333333,1.5,   1.66666667,   1.83333333,2.,   2.16666667,   2.33333333,   2.5,2.66666667,   2.83333333,   3.,   3.,4.,   5.,   6.,   7.,8.,   9.,  10.]
			# Av_list = (np.linspace(0,0.5,10))	#[0.,0.16666667,0.33333333,0.5,0.66666667,0.83333333,1.,1.16666667,1.33333333,1.5,   1.66666667,   1.83333333,2.,   2.16666667,   2.33333333,   2.5,2.66666667,   2.83333333,   3.,   3.,4.,   5.,   6.,   7.,8.,   9.,  10.]


		# ------------------------------
		# 2) read the Class III spectrum
		# ------------------------------
		# if no for on cl3 is needed, but you just want to use one peculiar Class III, then you should have typed it in
		if cl3_in != None:
			print('You have given a selection of spectral types')
			# if len(cl3_in) == 1:
			# 	cl3_in_list = [cl3_in]
			# else:
				# cl3_in_list = cl3_in
			cl3_in_list = np.array(cl3_in,dtype=np.float32)#cl3_in
		# otherwise, I assume you want to cycle on ALL the class III that we have
		else:
			print('No spectral types given, will try entire range')
			cl3_in_list = np.array(range(-10,9))

		self.cl3_in_list = cl3_in_list



		# 3) BIG FOR
		# now makes use of the RAY package, RC
		time_init = time.time()
		print("working_dir   " + PATH+'/models_grid/')
		ray.init(runtime_env={"working_dir":os.path.dirname(PATH), "excludes": [
						'src/models_grid/**',
						'tests/**/*',
						'tests/**',
            "src/models_grid/**/*.gif",
            "src/models_grid/**/*.txt",
            "src/models_grid/**/*.sav",
            "src/models_grid/**/*.tgz",
            "src/models_grid/**/*.zip"
        ]})#
		pool_outputs1 = ray.get([main_process.remote(self,cl3_in_list[i])for i in range(len(cl3_in_list))])

		print( 'Execution time:', time.time() - time_init, " seconds")
		######

		dim = len(pool_outputs1)
		chi_sq = pool_outputs1[0][0]
		H_fin = pool_outputs1[0][1]
		K_fin = pool_outputs1[0][2]

		for i in range(dim):

			chi_sq.update(pool_outputs1[i][0])
			H_fin.update(pool_outputs1[i][1])
			K_fin.update(pool_outputs1[i][2])

		self.chi_sq = chi_sq
		self.H_fin = H_fin
		self.K_fin = K_fin

		# ------------------------------
		# 4) SELECT THE BEST FIT
		# ------------------------------
		# - save the 40 best values
		best_chi_sq_val = np.sort(np.array(list(chi_sq.values())))[:40]
		# - save the 40 elements where chi_sq equal min_chi_sq
		best_chi_sq = {}

		for k in chi_sq:
			if chi_sq[k] in best_chi_sq_val:
				best_chi_sq[k] = chi_sq[k]
		# - value of the chi squared of the best fit
		self.best_chi_sq =best_chi_sq
		self.min_chi_sq = min(chi_sq.values())
		for k in best_chi_sq:
			if best_chi_sq[k] == self.min_chi_sq:
				self.min_chi_sq_cl3 = k.split('/')[0]
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
		# save all the values of the chi2 in a csv file for later use (e.g. chi_sq_analysis.py)
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
		best_chi_sq = self.best_chi_sq
		min_chi_sq = self.min_chi_sq
		min_chi_sq_cl3 = self.min_chi_sq_cl3
		min_chi_sq_Av = self.min_chi_sq_Av
		min_chi_sq_T = self.min_chi_sq_T
		min_chi_sq_Ne = self.min_chi_sq_Ne
		min_chi_sq_tau = self.min_chi_sq_tau
		min_chi_sq_H = self.min_chi_sq_H
		min_chi_sq_K = self.min_chi_sq_K
		Av_list =self.Av_list
		PATH_OUT =self.PATH_OUT
		dist_pc = self.dist_pc
		obj_in  = self.obj_in
		wl_VIS = self.wl_VIS
		fl_VIS = self.fl_VIS_in/cardelli_extinction(wl_VIS*10.,min_chi_sq_Av, Rv=self.Rv)
		filename_VIS =self.filename_VIS
		# ------------------------------
		# 6) CALCULATE Lacc from the BEST FIT slab
		# ------------------------------
		# remember to use the whole slab spectra, not the resampled ones!!!!
		# check if it has already been calculated this model, in which case you should read it
		if os.path.isfile(PATH_SLAB+'continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out'):
			wl_slab,fl_slab = readcol_py3(PATH_SLAB+'continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out',2,format='F,F')
		else:
		# otherwise, create it and then read it
			# first, write the input file
			f = open(PATH_ACC+'in.slab', 'w')
			outLine = min_chi_sq_T+'   '+min_chi_sq_Ne+'   '+min_chi_sq_tau
			f.write(outLine)
			#f.write(string.join([min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau],'   '))
			f.close()
			# run the C++ slab model program using the best fit parameters to calculate the slab model from 50 nm to 2477 nm (whole range)
			os.chdir(PATH_ACC)
			os.system('./hydrogen_slab')
			os.chdir(PATH)
			# read the result of the C++ program
			wl_slab,fl_slab = readcol_py3(PATH_ACC+'results/continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out',2,format='F,F')


		# GET THE Lacc FROM THE FITTING SLAB MODEL
		# fl_slab is in erg s-1 cm-2 nm-1 sr-1
		# H = Area_slab/D^2 (solid angle) [sr]

		#    - integrate over all the wavelengths
		F_acc = int_tabulated(wl_slab,fl_slab*min_chi_sq_H)  #  [erg s-1 cm-2]

		#    - multiply times 4pi*D^2 [cm2]
		dist = dist_pc*3.1e18	# [cm]
		lum_acc = F_acc*4.*np.pi*(dist**2.) #  [erg s-1]

		#    - in solar luminosity
		Lacc_Lsun = lum_acc/3.84e33 # in L_sun


		# ------------------------------
		# 7) CALCULATE Lstar from the BEST FIT classIII luminosity
		# ------------------------------
		# get distance and luminosity of the best fit class III
		#name_cl3,SpT_cl3,dist_cl3_pc,logL_cl3 = readcol_py3(PATH_CLASSIII+'data_classIII.txt',4,format='A,X,A,X,I,F',skipline=1)
		#dist_cl3_pc_fin = dist_cl3_pc[np.where(name_cl3 == min_chi_sq_cl3)]

		#Lstar_cl3_fin = 10.**(logL_cl3[np.where(name_cl3 == min_chi_sq_cl3)])
		# read the sav file with the correctly sampled slab model
		s = readsav(PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,self.cl3_in_toSelectModel))
		wl_slab_UVB_c,fl_slab_UVB_c = s['w'],s['f']
		s = readsav(PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,self.cl3_in_toSelectModel))
		wl_slab_VIS_c,fl_slab_VIS_c = s['w'],s['f']

		wl_slab = np.concatenate([wl_slab_UVB_c[wl_slab_UVB_c<550],wl_slab_VIS_c[wl_slab_VIS_c>550]])
		fl_slab = np.concatenate([fl_slab_UVB_c[wl_slab_UVB_c<550],fl_slab_VIS_c[wl_slab_VIS_c>550]])

		# Here I load the SpT -Teff relation of HH14
		relDir = PATH+'/models_grid/SpT_Teff_relation_hh14_short_codes.dat'
		relation = np.genfromtxt(relDir,usecols=(1,2),skip_header=1,dtype=[('sptCode',float),('Teff',float)])

		### This is still the HH14 bolometric correction with my changes made to it
		BolCorrDir = PATH+'/models_grid/BolCorr_hh14_myCorr_wInteroplationOverVO.txt'
		BolCorr = np.genfromtxt(BolCorrDir,usecols=(0,1),skip_header=1,dtype=[('Tphot',float),('f751Fbol',float)])


		# here T eff is obtained from the Spt
		Teff = np.array([np.interp(float(min_chi_sq_cl3),relation['sptCode'],relation['Teff'])])#[0]



		#Here the bolometric correction is aplied to the slab model subtracted observed spectrum
		f751_obs = np.median(fl_VIS[(wl_VIS>747)&(wl_VIS<755)])
		#####
		#THIS bit is the interpolation over the VO feature.
		wlScaling = 751

		if Teff < 3500:
			xlow = 730
			xhi =758
			flow = np.nanmedian(fl_VIS[(wl_VIS>xlow-0.4)&(wl_VIS<xlow+0.4)]) - (min_chi_sq_H*np.nanmedian(fl_slab[(wl_slab>xlow-0.4)&(wl_slab<xlow+0.4)]))
			fhigh = np.nanmedian(fl_VIS[(wl_VIS>xhi-0.4)&(wl_VIS<xhi+0.4)]) - (min_chi_sq_H*np.nanmedian(fl_slab[(wl_slab>xhi-0.4)&(wl_slab<xhi+0.4)]))
			f751_obs = ((flow*(xhi - wlScaling) )+ ((wlScaling - xlow)*fhigh))/(xhi-xlow)
			fl_photosphere751 = f751_obs
		else:
			f751_obs = np.median(fl_VIS[(wl_VIS>747)&(wl_VIS<755)])
			fl_slab751_forLum = np.median(fl_slab[(wl_slab>747)&(wl_slab<755)])
			fl_photosphere751 = f751_obs - (fl_slab751_forLum*min_chi_sq_H)

		fact = np.interp(Teff,BolCorr['Tphot'] ,BolCorr['f751Fbol'])#[0]
		#fbol = min_chi_sq_K/(10*fact)
		fbol = fl_photosphere751/(10*fact)
		CmPerPars = 3.08567758128e18
		Lsun = 3.826e33 #ergs/s
		# using the normalization constant of the Class III and the distances get the input object luminosity
		Lstar_input = 4*np.pi*((dist_pc*CmPerPars)**2)*fbol * (1/Lsun)

		logLstar_input = np.log10(Lstar_input)



		# ------------------------------
		# 8) PRINT SOME OUTPUTS
		# ------------------------------
		SpTBestFit = scod.convScodToSpTstring(float(min_chi_sq_cl3))




		# BETTER file
		f = open(PATH_OUT+obj_in+'_best_fit.dat','w')
		f.write('FIT OF THE OBJECT %s USING THE CLASS III %s \n' % (obj_in,min_chi_sq_cl3))
		f.write('Executed on '+time.asctime( time.localtime(time.time()) )+'\n')
		f.write('Using the file %s\n' % filename_VIS)
		f.write('\n')
		f.write('INPUT PARAMETERS:\n')
		f.write('dist = %i pc\n' % dist_pc)
		f.write('Rv = %0.1f\n' % Rv)
		f.write('\n')
		f.write('BEST FIT:\n')
		f.write('CLASS III: %s \n' % (min_chi_sq_cl3))
		f.write('\n')
		f.write('spectral type: '+ SpTBestFit)
		f.write('\n')
		#name_cl3,SpT_cl3, Teff_cl3 = readcol_py3(PATH_CLASSIII+'summary_classIII.txt',3,format='A,X,A,I',skipline=1)
		#f.write('SpT: %s \n' % (SpT_cl3[np.where(min_chi_sq_cl3 == name_cl3)]))
		f.write('Teff: %i \n' % (Teff))
		f.write('Chi2 = %0.3e \n' % min_chi_sq)
		f.write('\n')
		f.write('OBJECT PARAMETERS:\n')
		f.write('Av = %0.2f mag\n' % min_chi_sq_Av)
		f.write('Lacc/Lsun = %0.2e \n'% Lacc_Lsun)
		f.write('log(Lacc/Lsun) = %0.3f \n'% np.log10(Lacc_Lsun))
		f.write('Lstar = %0.2f Lsun\n' % Lstar_input)
		f.write('log(Lstar/Lsun) = %0.2f \n' % logLstar_input)
		f.write('\n')
		f.write('NORMALIZATION CONSTANTS: \n')
		f.write('H = %0.3e \n' % (min_chi_sq_H))
		f.write('K = %0.3e \n'% (min_chi_sq_K))
		f.write('\n')
		f.write('SLAB PARAMETERS:\n')
		f.write('T = %s K \n'% min_chi_sq_T)
		f.write('n_e = %s cm-3\n'% min_chi_sq_Ne)
		f.write('tau (300 nm) =  %s \n'% min_chi_sq_tau)
		f.write('Area = %0.2e cm2 \n' % (min_chi_sq_H*dist**2.))
		f.write('Radius = %0.2e cm \n'% (np.sqrt(min_chi_sq_H*dist**2./np.pi)))
		f.write('\n')
		f.write('OBJECT DERIVED PARAMETERS:\n')
		mass_siess,age_siess = isochrone_interp(np.log10(Teff),logLstar_input,model='Siess',PATH = PATH)
		mass_bara,age_bara = isochrone_interp(np.log10(Teff),logLstar_input,model='Baraffe',PATH = PATH)
		mass_palla,age_palla = isochrone_interp(np.log10(Teff),logLstar_input,model='Palla',PATH = PATH)
		mass_danto,age_danto = isochrone_interp(np.log10(Teff),logLstar_input,model='Dantona',PATH = PATH)
		mass_b15,age_b15 = isochrone_interp(np.log10(Teff),logLstar_input,model='B15',PATH = PATH)
		mass_Feiden,age_Feiden = isochrone_interp(np.log10(Teff),logLstar_input,model='Feiden',PATH = PATH)
		mstar,macc_siess = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Siess',PATH = PATH)
		mstar,macc_bara = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Baraffe',PATH = PATH)
		mstar,macc_palla = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Palla',PATH = PATH)
		mstar,macc_danto = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Dantona',PATH = PATH)
		mstar,macc_b15 = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='B15',PATH = PATH)
		mstar,macc_Feiden = macc_calc(Teff,Lstar_input,Lacc_Lsun,model='Feiden',PATH = PATH)
		f.write('M = %0.2f Msun Age = %.2f (BARAFFE+15)\n' % (mass_b15,10.**(age_b15)/1e6))
		f.write('Macc = %0.2e Msun/yr (BARAFFE+15)\n' % macc_b15)
		f.write('M = %0.2f Msun Age = %.2f (SIESS)\n' % (mass_siess,10.**(age_siess)/1e6))
		f.write('Macc = %0.2e Msun/yr (SIESS)\n' % macc_siess)
		f.write('M = %0.2f Msun Age = %.2f (BARAFFE)\n' % (mass_bara,10.**(age_bara)/1e6))
		f.write('Macc = %0.2e Msun/yr (BARAFFE)\n' % macc_bara)
		f.write('M = %0.2f Msun Age = %.2f (PALLA)\n' % (mass_palla,10.**(age_palla)/1e6))
		f.write('Macc = %0.2e Msun/yr (PALLA)\n' % macc_palla)
		f.write('M = %0.2f Msun Age = %.2f (DANTONA)\n' % (mass_danto,10.**(age_danto)/1e6))
		f.write('Macc = %0.2e Msun/yr (DANTONA)\n' % macc_danto)
		f.write('M = %0.2f Msun Age = %.2f (Feiden)\n' % (mass_Feiden,10.**(age_Feiden)/1e6))
		f.write('Macc = %0.2e Msun/yr (Feiden)\n' % macc_Feiden)
		f.write('\n')
		f.write('\n')
		f.write('Other best chi2 results:\n')
		f.write('%s' % best_chi_sq)
		f.write('\n')
		f.write('\n')
		f.write('FITTED PARAMETERS:\n')
		f.write('Class IIIs: %s \n' % cl3_in)
		f.write('Av: %s \n' % Av_list)
		#f.write('Execution time: %i seconds\n' % (time.time() - time_init))
		f.write('\n')
		f.write('\n')
		f.write('These values were computed using the grit found at: \n')
		f.write(gridFile +'\n')
		f.write('this grid contains the following features: \n')
		f.write(str(self.usedFeatures)+'\n')
		f.write('and is normalized at: \n')
		f.write(str(self.normWLandWidth[0])+'nm')
		f.write('\n')
		f.write('\n')
		f.close()

	def addEntryToTable(self, FileOut = None):
		best_chi_sq = self.best_chi_sq
		min_chi_sq = self.min_chi_sq
		min_chi_sq_cl3 = self.min_chi_sq_cl3
		min_chi_sq_Av = self.min_chi_sq_Av
		min_chi_sq_T = self.min_chi_sq_T
		min_chi_sq_Ne = self.min_chi_sq_Ne
		min_chi_sq_tau = self.min_chi_sq_tau
		min_chi_sq_H = self.min_chi_sq_H
		min_chi_sq_K = self.min_chi_sq_K
		Av_list =self.Av_list
		PATH_OUT =self.PATH_OUT
		obj_in  = self.obj_in
		dist_pc = self.dist_pc
		wl_VIS = self.wl_VIS
		fl_VIS = self.fl_VIS_in/cardelli_extinction(wl_VIS*10.,min_chi_sq_Av, Rv=self.Rv)
		filename_VIS =self.filename_VIS
		# ------------------------------
		# 6) CALCULATE Lacc from the BEST FIT slab
		# ------------------------------
		# remember to use the whole slab spectra, not the resampled ones!!!!
		# check if it has already been calculated this model, in which case you should read it
		if os.path.isfile(PATH_SLAB+'continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out'):
			wl_slab,fl_slab = readcol_py3(PATH_SLAB+'continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out',2,format='F,F')
		else:
		# otherwise, create it and then read it
			# first, write the input file
			f = open(PATH_ACC+'in.slab', 'w')
			outLine = min_chi_sq_T+'   '+min_chi_sq_Ne+'   '+min_chi_sq_tau
			f.write(outLine)
			#f.write(string.join([min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau],'   '))
			f.close()
			# run the C++ slab model program using the best fit parameters to calculate the slab model from 50 nm to 2477 nm (whole range)
			os.chdir(PATH_ACC)
			os.system('./hydrogen_slab')
			os.chdir(PATH)
			# read the result of the C++ program
			wl_slab,fl_slab = readcol_py3(PATH_ACC+'results/continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out',2,format='F,F')


		# GET THE Lacc FROM THE FITTING SLAB MODEL
		# fl_slab is in erg s-1 cm-2 nm-1 sr-1
		# H = Area_slab/D^2 (solid angle) [sr]

		#    - integrate over all the wavelengths
		F_acc = int_tabulated(wl_slab,fl_slab*min_chi_sq_H)  #  [erg s-1 cm-2]

		#    - multiply times 4pi*D^2 [cm2]
		dist = dist_pc*3.1e18	# [cm]
		lum_acc = F_acc*4.*np.pi*(dist**2.) #  [erg s-1]

		#    - in solar luminosity
		Lacc_Lsun = lum_acc/3.84e33 # in L_sun


		# ------------------------------
		# 7) CALCULATE Lstar from the BEST FIT classIII luminosity
		# ------------------------------
		# get distance and luminosity of the best fit class III
		#name_cl3,SpT_cl3,dist_cl3_pc,logL_cl3 = readcol_py3(PATH_CLASSIII+'data_classIII.txt',4,format='A,X,A,X,I,F',skipline=1)
		#dist_cl3_pc_fin = dist_cl3_pc[np.where(name_cl3 == min_chi_sq_cl3)]

		#Lstar_cl3_fin = 10.**(logL_cl3[np.where(name_cl3 == min_chi_sq_cl3)])
		# read the sav file with the correctly sampled slab model
		s = readsav(PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,self.cl3_in_toSelectModel))
		wl_slab_UVB_c,fl_slab_UVB_c = s['w'],s['f']
		s = readsav(PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,self.cl3_in_toSelectModel))
		wl_slab_VIS_c,fl_slab_VIS_c = s['w'],s['f']

		wl_slab = np.concatenate([wl_slab_UVB_c[wl_slab_UVB_c<550],wl_slab_VIS_c[wl_slab_VIS_c>550]])
		fl_slab = np.concatenate([fl_slab_UVB_c[wl_slab_UVB_c<550],fl_slab_VIS_c[wl_slab_VIS_c>550]])

		# Here I load the SpT -Teff relation of HH14
		relDir = PATH+'/models_grid/SpT_Teff_relation_hh14_short_codes.dat'
		relation = np.genfromtxt(relDir,usecols=(1,2),skip_header=1,dtype=[('sptCode',float),('Teff',float)])

		### This is still the HH14 bolometric correction with my changes made to it
		BolCorrDir = PATH+'/models_grid/BolCorr_hh14_myCorr_wInteroplationOverVO.txt'
		BolCorr = np.genfromtxt(BolCorrDir,usecols=(0,1),skip_header=1,dtype=[('Tphot',float),('f751Fbol',float)])


		# here T eff is obtained from the Spt
		Teff = np.array([np.interp(float(min_chi_sq_cl3),relation['sptCode'],relation['Teff'])])#[0]



		#Here the bolometric correction is aplied to the slab model subtracted observed spectrum
		f751_obs = np.median(fl_VIS[(wl_VIS>747)&(wl_VIS<755)])
		#####
		#THIS bit is the interpolation over the VO feature.
		wlScaling = 751

		if Teff < 3500:
			xlow = 730
			xhi =758
			flow = np.nanmedian(fl_VIS[(wl_VIS>xlow-0.4)&(wl_VIS<xlow+0.4)]) - (min_chi_sq_H*np.nanmedian(fl_slab[(wl_slab>xlow-0.4)&(wl_slab<xlow+0.4)]))
			fhigh = np.nanmedian(fl_VIS[(wl_VIS>xhi-0.4)&(wl_VIS<xhi+0.4)]) - (min_chi_sq_H*np.nanmedian(fl_slab[(wl_slab>xhi-0.4)&(wl_slab<xhi+0.4)]))
			f751_obs = ((flow*(xhi - wlScaling) )+ ((wlScaling - xlow)*fhigh))/(xhi-xlow)
			fl_photosphere751 = f751_obs
		else:
			f751_obs = np.median(fl_VIS[(wl_VIS>747)&(wl_VIS<755)])
			fl_slab751_forLum = np.median(fl_slab[(wl_slab>747)&(wl_slab<755)])
			fl_photosphere751 = f751_obs - (fl_slab751_forLum*min_chi_sq_H)

		fact = np.interp(Teff,BolCorr['Tphot'] ,BolCorr['f751Fbol'])#[0]
		#fbol = min_chi_sq_K/(10*fact)
		fbol = fl_photosphere751/(10*fact)
		CmPerPars = 3.08567758128e18
		Lsun = 3.826e33 #ergs/s
		# using the normalization constant of the Class III and the distances get the input object luminosity
		Lstar_input = 4*np.pi*((dist_pc*CmPerPars)**2)*fbol * (1/Lsun)

		logLstar_input = np.log10(Lstar_input)



		# ------------------------------
		# 8) PRINT SOME OUTPUTS
		# ------------------------------
		SpTBestFit = scod.convScodToSpTstring(float(min_chi_sq_cl3))

		path  = self.path

		mass_siess,age_siess = isochrone_interp(np.log10(Teff),logLstar_input,model='Siess',PATH = PATH)
		mass_bara,age_bara = isochrone_interp(np.log10(Teff),logLstar_input,model='Baraffe',PATH = PATH)
		mass_palla,age_palla = isochrone_interp(np.log10(Teff),logLstar_input,model='Palla',PATH = PATH)
		mass_danto,age_danto = isochrone_interp(np.log10(Teff),logLstar_input,model='Dantona',PATH = PATH)
		mass_b15,age_b15 = isochrone_interp(np.log10(Teff),logLstar_input,model='B15',PATH = PATH)
		mass_Feiden,age_Feiden = isochrone_interp(np.log10(Teff),logLstar_input,model='Feiden',PATH = PATH)
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

		#index of the input classIII
		f.write('%s \t %0.1f \t %s \t %0.2f \t %0.1f \t %0.2f \t %0.2f  \t %0.3e \t %0.3f \t %0.3f \t %0.4e \t %0.3f \t %0.4e \t %s \t %s \t %s \t %0.4e \t %0.4e \n'
		% (obj_in,dist_pc,SpTBestFit,float(min_chi_sq_cl3),Teff,min_chi_sq_Av,Rv,Lstar_input,np.log10(Lacc_Lsun),mass_b15,
		macc_b15,mass_siess,macc_siess,min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,min_chi_sq_H,min_chi_sq_K))
		#f.write('%i ' % dist_pc)
		f.write('\t')

		f.close()


	def plotRegFit(self,close = False):
		best_chi_sq = self.best_chi_sq
		min_chi_sq = self.min_chi_sq
		min_chi_sq_cl3 = self.min_chi_sq_cl3
		min_chi_sq_Av = self.min_chi_sq_Av
		min_chi_sq_T = self.min_chi_sq_T
		min_chi_sq_Ne = self.min_chi_sq_Ne
		min_chi_sq_tau = self.min_chi_sq_tau
		min_chi_sq_H = self.min_chi_sq_H
		min_chi_sq_K = self.min_chi_sq_K
		Av_list =self.Av_list
		PATH_OUT =self.PATH_OUT
		obj_in  = self.obj_in
		ind_uvb = self.ind_uvb
		usedFeatures =self.usedFeatures
		classIIIreadIn = self.classIIIreadIn
		normWLandWidth = self.normWLandWidth

		wl_UVB = self.wl_UVB
		fl_UVB = self.fl_UVB_in/cardelli_extinction(wl_UVB*10.,min_chi_sq_Av, Rv=self.Rv)
		wl_VIS = self.wl_VIS
		fl_VIS = self.fl_VIS_in/cardelli_extinction(wl_VIS*10.,min_chi_sq_Av, Rv=self.Rv)
		filename_VIS =self.filename_VIS

		# load NIR file!
		filename_NIR = self.filename_NIR
		fitsTab = self.fitsTab
		if filename_NIR == None:
			filename_NIR = path+'data_final/flux_%s_nir_tell.fits' % obj_in
		if os.path.isfile(filename_NIR):
			if fitsTab == False:
				wl_NIR,fl_NIR_in,hdr_NIR=spec_readspec(filename_NIR, 'hdr')	#the 'hdr' string is there to say that I want to save the header
			else:
				wl_NIR,fl_NIR_in,hdr_NIR=readspec_phase3(filename_NIR, hdr_out='YES')
			if self.perAA == True:
				fl_NIR_in = 10*fl_NIR_in
		else:
			print('No NIR spectrum')
			wl_NIR,fl_NIR_in = np.array([0 for x in range(100)]),np.array([0 for x in range(100)])
			#sys.exit(1)

		fl_NIR = fl_NIR_in/cardelli_extinction(wl_NIR*10.,min_chi_sq_Av, Rv=Rv)

		#Load Slab!!
		if os.path.isfile(PATH_SLAB+'continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out'):
			wl_slab,fl_slab = readcol_py3(PATH_SLAB+'continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out',2,format='F,F')
		else:
		# otherwise, create it and then read it
			# first, write the input file
			f = open(PATH_ACC+'in.slab', 'w')
			outLine = min_chi_sq_T+'   '+min_chi_sq_Ne+'   '+min_chi_sq_tau
			f.write(outLine)
			#f.write(string.join([min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau],'   '))
			f.close()
			# run the C++ slab model program using the best fit parameters to calculate the slab model from 50 nm to 2477 nm (whole range)
			os.chdir(PATH_ACC)
			os.system('./hydrogen_slab')
			os.chdir(PATH)
			# read the result of the C++ program
			wl_slab,fl_slab = readcol_py3(PATH_ACC+'results/continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out',2,format='F,F')



		p1 = pl.figure(figsize=(10,9))
		##############
		# Balmer jump region
		##############
		pl.subplot(211)

		### read in the features used for plotting only!!!
		gridFilePlot = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeUVB/interp.npz'
		classIIIFeatPlotting = pf.classIII(gridFilePlot)
		usedFeaturesPlotting = classIIIFeatPlotting.getUsedInterpFeat()
		featuresPlot,errorsPlot = classIIIFeatPlotting.getFeatsAtSpt_symetricErr(min_chi_sq_cl3)

		gridFilePlotVIS = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeVIS/interp.npz'
		classIIIFeatPlottingVIS = pf.classIII(gridFilePlotVIS)
		usedFeaturesPlottingVIS = classIIIFeatPlottingVIS.getUsedInterpFeat()
		featuresPlotVIS,errorsPlotVIS = classIIIFeatPlottingVIS.getFeatsAtSpt_symetricErr(min_chi_sq_cl3)

		usedFeaturesPlotting = np.concatenate((usedFeaturesPlotting,usedFeaturesPlottingVIS))
		featuresPlot = np.concatenate((featuresPlot,featuresPlotVIS))
		errorsPlot = np.concatenate((errorsPlot,errorsPlotVIS))


		if plot_smooth == False:
			pl.plot(wl_UVB[ind_uvb],fl_UVB[ind_uvb],'k',zorder = 1)#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
			print('best fit sptCode = '+min_chi_sq_cl3)

			# features used during fitting
			wlFeat = (usedFeatures[:,0]+usedFeatures[:,1])/2
			Xrange = np.abs(usedFeatures[:,0] - wlFeat)
			features,errors = classIIIreadIn.getFeatsAtSpt_symetricErr(min_chi_sq_cl3)



			fit_cont = np.zeros(len(usedFeatures))
			for i in range(len(usedFeatures)):
				fit_cont[i] = min_chi_sq_K*features[i]+min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeatures[i,0],usedFeatures[i,1])[0]
			stdTerm1 = (min_chi_sq_K*errors)**2
			stdTerm2 = 0#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not used for the plot, this is a hold over!!!
			fit_std = np.sqrt(stdTerm1 +stdTerm2)

			pl.errorbar(wlFeat,min_chi_sq_K*features,min_chi_sq_K*errors,xerr = Xrange,fmt='.',c=(213/255,94/255,0/255),zorder = 10)
			pl.errorbar(wlFeat,fit_cont,yerr=fit_std,xerr= Xrange,fmt='.',c=(204/255 ,121/255,167/255),zorder = 10)


			# features used only for plotting
			wlFeatPlot = (usedFeaturesPlotting[:,0]+usedFeaturesPlotting[:,1])/2
			XrangePlot = np.abs(usedFeaturesPlotting[:,0] - wlFeatPlot)


			fit_contPlot = np.zeros(len(usedFeaturesPlotting))
			for i in range(len(usedFeaturesPlotting[:,0])):
				fit_contPlot[i] = min_chi_sq_K*featuresPlot[i]+min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeaturesPlotting[i,0],usedFeaturesPlotting[i,1])[0]
			stdTerm1plot = (min_chi_sq_K*errorsPlot)**2
			stdTerm2plot = 0#(stddev_cont_dered**2)#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not taken into account!!!
			fit_stdPlot = np.sqrt(stdTerm1plot +stdTerm2plot)

			pl.plot(wlFeatPlot,min_chi_sq_K*featuresPlot ,c=(240/255,228/255,66/255),zorder = 10)
			pl.plot(wlFeatPlot,fit_contPlot ,c=(86/255,180/255,233/255),zorder = 10)

			pl.plot(wl_cl3_UVB[ind_uvb_3],min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],c= (0,158/255,115/255),zorder = 9)
		else:
			wl_UVB_smooth = wl_UVB[ind_uvb][::8]
			fl_UVB_smooth = spectrum_resample(fl_UVB,wl_UVB,wl_UVB_smooth)
			pl.plot(wl_UVB_smooth,fl_UVB_smooth,'k',zorder = 1)#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')

			wlFeat = (usedFeatures[:,0]+usedFeatures[:,1])/2
			Xrange = np.abs(usedFeatures[:,0] - wlFeat)
			features,errors = classIIIreadIn.getFeatsAtSpt_symetricErr(min_chi_sq_cl3)

			fit_cont = np.zeros(len(usedFeatures))
			for i in range(len(usedFeatures)):
				fit_cont[i] = min_chi_sq_K*features[i]+min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeatures[i,0],usedFeatures[i,1])[0]
			stdTerm1 = (min_chi_sq_K*errors)**2
			stdTerm2 = 0#(stddev_cont_dered**2)
			fit_std = np.sqrt(stdTerm1 +stdTerm2)

			pl.errorbar(wlFeat,min_chi_sq_K*features,min_chi_sq_K*errors,xerr = Xrange,fmt='.',c=(213/255,94/255,0/255),zorder = 10)
			pl.errorbar(wlFeat,fit_cont,yerr=fit_std,xerr= Xrange,fmt='.',c=(204/255,121/255,167/255),zorder = 10)

			# features used only for plotting
			wlFeatPlot = (usedFeaturesPlotting[:,0]+usedFeaturesPlotting[:,1])/2
			XrangePlot = np.abs(usedFeaturesPlotting[:,0] - wlFeatPlot)


			fit_contPlot = np.zeros(len(usedFeaturesPlotting))
			for i in range(len(usedFeaturesPlotting[:,0])):
				fit_contPlot[i] = min_chi_sq_K*featuresPlot[i]+min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeaturesPlotting[i,0],usedFeaturesPlotting[i,1])[0]
			stdTerm1plot = (min_chi_sq_K*errorsPlot)**2
			stdTerm2plot = 0#(stddev_cont_dered**2)#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not taken into account!!!
			fit_stdPlot = np.sqrt(stdTerm1plot +stdTerm2plot)

			pl.plot(wlFeatPlot,min_chi_sq_K*featuresPlot ,c=(240/255,228/255,66/255),zorder = 10)
			pl.plot(wlFeatPlot,fit_contPlot ,c=(86/255,180/255,233/255),zorder = 10)

			pl.plot(wl_slab,min_chi_sq_H*fl_slab,'c',zorder = 9)

		pl.title(obj_in)#+' - Chi$^2$= %0.3e' % (min_chi_sq))
		# pl.xlabel('Wavelength [nm]')
		pl.ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$]')
		pl.axis([330,470,-1e-16,3.5*np.mean(fl_UVB[(wl_UVB < 360) & (wl_UVB > 330)])])
		# pl.tight_layout()
		pl.savefig(PATH_OUT+'%s_clIII_%s.png'% (obj_in,min_chi_sq_cl3))
		#pl.show()




		##############
		# CaII 420 nm
		##############

		PATH_MY_CLASSIII = PATH+'/models_grid/RunnableGrid/'

		wl_cl3_UVB,fl_cl3_UVB,wl_cl3_VIS,fl_cl3_VIS,wl_cl3_NIR,fl_cl3_NIR,nameWls = pf.readMixClassIII(min_chi_sq_cl3,PATH_MY_CLASSIII,wlNorm =normWLandWidth[0])
		ind_uvb_3 = (wl_cl3_UVB <= max_wl_uvb)
		ind_vis_3 = (wl_cl3_VIS >= min_wl_vis)

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c= s['w'],s['f']
			#wl_slab,fl_slab = wl_slab_UVB_c,fl_slab_UVB_c
		else:
			#fluxcon = FluxConservingResampler()
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c = s['w'],s['f']
			#spec = Spectrum1D(spectral_axis=wl_slab_UVB_c, flux=fl_slab_UVB_c)
			fl_slab_UVB_c = np.interp(wl_cl3_UVB,wl_slab_UVB_c,fl_slab_UVB_c )
			wl_slab_UVB_c = wl_cl3_UVB
			#,fl_slab_UVB_c = np.array(resampledSpecUVB.spectral_axis/u.Unit('nm')), np.array(resampledSpecUVB.flux/u.Unit('erg cm-2 s-1 nm-1'))


		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c= s['w'],s['f']
		else:
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c = s['w'],s['f']
			fl_slab_VIS_c = np.interp(wl_cl3_VIS,wl_slab_VIS_c,fl_slab_VIS_c )
			wl_slab_VIS_c = wl_cl3_VIS



		# 1) normalize
		f_in_norm = np.nanmedian(fl_UVB[(wl_UVB > 417.5) & (wl_UVB < 419.5)])
		f_tot_norm = np.nanmedian(fl_cl3_UVB[(wl_cl3_UVB > 417.5)&(wl_cl3_UVB < 419.5)]*min_chi_sq_K+fl_slab_UVB_c[(wl_cl3_UVB > 417.5)&(wl_cl3_UVB < 419.5)]*min_chi_sq_H)
		f_clIII_norm = np.nanmedian(fl_cl3_UVB[(wl_cl3_UVB > 417.5)&(wl_cl3_UVB < 419.5)]*min_chi_sq_K)
		###For the features plotted but not FITTED
		f_ClIIIFeat_norm = min_chi_sq_K*featuresPlot[(wlFeatPlot > 417.5) & (wlFeatPlot < 419.5)]
		f_totFeat_norm = fit_contPlot[(wlFeatPlot > 417.5) & (wlFeatPlot < 419.5)]

		# 2) plot
		pl.subplot(224)
		pl.plot(wl_UVB[ind_uvb],fl_UVB[ind_uvb]/f_in_norm,'k')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
		pl.xlabel('Wavelength [nm]')
		# pl.ylabel('Flux')
		pl.axis([420,425,0,1.6])


		pl.plot(wl_cl3_UVB[ind_uvb_3],min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3]/f_tot_norm,'c')

		# Features used only for plotting
		pl.plot(wlFeatPlot,(min_chi_sq_K*featuresPlot)/f_ClIIIFeat_norm ,c=(240/255,228/255,66/255),zorder = 10)
		pl.plot(wlFeatPlot,(fit_contPlot)/f_totFeat_norm ,c=(86/255,180/255,233/255),zorder = 10)

		#pl.savefig(PATH_OUT+'%s_clIII_%s.png'% (obj_in,min_chi_sq_cl3))
		#pl.show()
		pl.plot(wl_cl3_UVB[ind_uvb_3],(min_chi_sq_K*fl_cl3_UVB[ind_uvb_3]+min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3])/f_tot_norm,c='b')
		pl.plot(wl_cl3_UVB[ind_uvb_3],min_chi_sq_K*fl_cl3_UVB[ind_uvb_3]/f_clIII_norm,c ='g',alpha =0.6)



		#TiO 710
		ind_vis = (wl_VIS >= min_wl_vis)
		pl.subplot(223)
		pl.plot(wl_VIS[ind_vis],fl_VIS[ind_vis],'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
		pl.xlabel('Wavelength [nm]')
		pl.ylabel('Flux')
		pl.axis([700,720,-1e-16,1.5*np.mean(fl_VIS[(wl_VIS < 720) & (wl_VIS > 700)])])
		#pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'b')
		pl.errorbar(wlFeat,fit_cont,yerr=fit_std,xerr= Xrange,fmt='.',c='b',zorder = 10)
		#pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3],'g')
		pl.errorbar(wlFeat,min_chi_sq_K*features,min_chi_sq_K*errors,xerr = Xrange,fmt='.',c='g',zorder = 10)
		#pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'c')
		pl.plot(wl_slab,min_chi_sq_H*fl_slab,'c',zorder = 9)

		pl.savefig(PATH_OUT+'%s_clIII_%s_%s.png'% (obj_in,min_chi_sq_cl3,nameWls))
		pl.show()
		if close:
			pl.close()

	def plotVeil(self,close =False):
		best_chi_sq = self.best_chi_sq
		min_chi_sq = self.min_chi_sq
		min_chi_sq_cl3 = self.min_chi_sq_cl3
		min_chi_sq_Av = self.min_chi_sq_Av
		min_chi_sq_T = self.min_chi_sq_T
		min_chi_sq_Ne = self.min_chi_sq_Ne
		min_chi_sq_tau = self.min_chi_sq_tau
		min_chi_sq_H = self.min_chi_sq_H
		min_chi_sq_K = self.min_chi_sq_K
		Av_list =self.Av_list
		PATH_OUT =self.PATH_OUT
		obj_in  = self.obj_in
		ind_uvb = self.ind_uvb
		ind_vis = self.ind_vis
		usedFeatures =self.usedFeatures
		classIIIreadIn = self.classIIIreadIn
		normWLandWidth = self.normWLandWidth

		wl_UVB = self.wl_UVB
		fl_UVB = self.fl_UVB_in/cardelli_extinction(wl_UVB*10.,min_chi_sq_Av, Rv=self.Rv)
		wl_VIS = self.wl_VIS
		fl_VIS = self.fl_VIS_in/cardelli_extinction(wl_VIS*10.,min_chi_sq_Av, Rv=self.Rv)
		filename_VIS =self.filename_VIS

		# load NIR file!
		filename_NIR = self.filename_NIR
		fitsTab = self.fitsTab
		if filename_NIR == None:
			filename_NIR = path+'data_final/flux_%s_nir_tell.fits' % obj_in
		if os.path.isfile(filename_NIR):
			if fitsTab == False:
				wl_NIR,fl_NIR_in,hdr_NIR=spec_readspec(filename_NIR, 'hdr')	#the 'hdr' string is there to say that I want to save the header
			else:
				wl_NIR,fl_NIR_in,hdr_NIR=readspec_phase3(filename_NIR, hdr_out='YES')
			if self.perAA == True:
				fl_NIR_in = 10*fl_NIR_in
		else:
			print('No NIR spectrum')
			wl_NIR,fl_NIR_in = np.array([0 for x in range(100)]),np.array([0 for x in range(100)])
			#sys.exit(1)

		fl_NIR = fl_NIR_in/cardelli_extinction(wl_NIR*10.,min_chi_sq_Av, Rv=Rv)

		#Load Slab!!
		if os.path.isfile(PATH_SLAB+'continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out'):
			wl_slab,fl_slab = readcol_py3(PATH_SLAB+'continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out',2,format='F,F')
		else:
		# otherwise, create it and then read it
			# first, write the input file
			f = open(PATH_ACC+'in.slab', 'w')
			outLine = min_chi_sq_T+'   '+min_chi_sq_Ne+'   '+min_chi_sq_tau
			f.write(outLine)
			#f.write(string.join([min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau],'   '))
			f.close()
			# run the C++ slab model program using the best fit parameters to calculate the slab model from 50 nm to 2477 nm (whole range)
			os.chdir(PATH_ACC)
			os.system('./hydrogen_slab')
			os.chdir(PATH)
			# read the result of the C++ program
			wl_slab,fl_slab = readcol_py3(PATH_ACC+'results/continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out',2,format='F,F')
		PATH_MY_CLASSIII = PATH+'/models_grid/RunnableGrid/'

		wl_cl3_UVB,fl_cl3_UVB,wl_cl3_VIS,fl_cl3_VIS,wl_cl3_NIR,fl_cl3_NIR,nameWls = pf.readMixClassIII(min_chi_sq_cl3,PATH_MY_CLASSIII,wlNorm =normWLandWidth[0])
		ind_uvb_3 = (wl_cl3_UVB <= max_wl_uvb)
		ind_vis_3 = (wl_cl3_VIS >= min_wl_vis)

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c= s['w'],s['f']
		else:
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c = s['w'],s['f']
			fl_slab_VIS_c = np.interp(wl_cl3_VIS,wl_slab_VIS_c,fl_slab_VIS_c )
			wl_slab_VIS_c = wl_cl3_VIS

		"""
		PLOT PHOTOSPHERIC FEATURES
		"""
		# 1) MATCH THE OBSERVED SPECTRUM AND THE CLASS III TO THE SAME WAVELENGTHS
		#			- do this by shifting the lithium line to the same position
		#print('HERE')
		# print(wl_cl3_VIS,fl_cl3_VIS)
		eqw, err_eqw, mode, fwhm, lline = eqw_auto(wl_VIS,fl_VIS,670.78,size_cont=1.,plot='NO',mode='gauss',fwhm_out='Ja',lline_out='Ja')
		# eqw, err_eqw, mode, fwhm, lline = eqw_auto(wl_VIS,fl_VIS,656.28,size_cont=2.,plot='NO',mode='gauss',fwhm_out='Ja',lline_out='Ja')
		if nameWls == 'LM601':
			lline3 = 670.8
		elif nameWls == 'Sz94':
			lline3 = 670.3
		else:
			eqw3, err_eqw3, mode3, fwhm3, lline3 = eqw_auto(wl_cl3_VIS,fl_cl3_VIS,670.78,size_cont=1.,plot='NO',mode='gauss',fwhm_out='Ja',lline_out='Ja')
			# eqw3, err_eqw3, mode3, fwhm3, lline3 = eqw_auto(wl_cl3_VIS,fl_cl3_VIS,656.28,size_cont=2.,plot='NO',mode='gauss',fwhm_out='Ja',lline_out='Ja')

		shift = lline - lline3
		# shift = 0.

		p1 = pl.figure(figsize=(10,9))
		##############
		# 1 - NaI ~820 nm
		##############
		# 1) normalize
		f_in_norm = np.mean(fl_VIS[(wl_VIS > 817.0) & (wl_VIS < 818.0)])
		f_tot_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 817.)&(wl_cl3_VIS < 818.)]*min_chi_sq_K+fl_slab_VIS_c[(wl_cl3_VIS > 817.)&(wl_cl3_VIS < 818.)]*min_chi_sq_H)
		f_clIII_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 817.)&(wl_cl3_VIS < 818.)]*min_chi_sq_K)

		# 2) plot
		pl.subplot(221)
		pl.plot(wl_VIS[ind_vis]-shift,fl_VIS[ind_vis]/f_in_norm,'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
		pl.title(obj_in)
		# pl.xlabel('Wavelength [nm]')
		pl.ylabel('Flux')
		pl.axis([816,822,0.1,1.3])
		pl.plot(wl_cl3_VIS[ind_vis_3],(min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+min_chi_sq_H*fl_slab_VIS_c[ind_vis_3])/f_tot_norm,'b')
		pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3]/f_clIII_norm,'g',alpha =0.6)
		pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_H*fl_slab_VIS_c[ind_vis_3]/f_tot_norm,'c')
		pl.text(821,0.4,'NaI')

		##############
		# 2 - KI ~770 nm
		##############
		# 1) normalize
		f_in_norm = np.mean(fl_VIS[(wl_VIS > 767.50) & (wl_VIS < 769.0)])
		f_tot_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 767.5)&(wl_cl3_VIS < 769.)]*min_chi_sq_K+fl_slab_VIS_c[(wl_cl3_VIS > 767.5)&(wl_cl3_VIS < 769.)]*min_chi_sq_H)
		f_clIII_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 767.5)&(wl_cl3_VIS < 769.)]*min_chi_sq_K)

		# 2) plot
		pl.subplot(222)
		pl.plot(wl_VIS[ind_vis]-shift,fl_VIS[ind_vis]/f_in_norm,'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
		# pl.xlabel('Wavelength [nm]')
		# pl.ylabel('Flux')
		pl.axis([764,774,0.1,1.3])
		pl.plot(wl_cl3_VIS[ind_vis_3],(min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+min_chi_sq_H*fl_slab_VIS_c[ind_vis_3])/f_tot_norm,'b')
		pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3]/f_clIII_norm,'g',alpha =0.6)
		pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_H*fl_slab_VIS_c[ind_vis_3]/f_tot_norm,'c')
		pl.text(772.5,0.4,'KI')

		##############
		# 3 - CaI ~616 nm
		##############
		# 1) normalize
		f_in_norm = np.mean(fl_VIS[(wl_VIS > 615.) & (wl_VIS < 615.8)])
		f_tot_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 615.)&(wl_cl3_VIS < 615.8)]*min_chi_sq_K+fl_slab_VIS_c[(wl_cl3_VIS > 615.)&(wl_cl3_VIS < 615.8)]*min_chi_sq_H)
		f_clIII_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 615.)&(wl_cl3_VIS < 615.8)]*min_chi_sq_K)

		# 2) plot
		pl.subplot(223)
		pl.plot(wl_VIS[ind_vis]-shift,fl_VIS[ind_vis]/f_in_norm,'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
		pl.xlabel('Wavelength [nm]')
		pl.ylabel('Flux')
		pl.axis([614,618,0.1,1.3])
		pl.xticks([614,615,616,617,618])
		pl.plot(wl_cl3_VIS[ind_vis_3],(min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+min_chi_sq_H*fl_slab_VIS_c[ind_vis_3])/f_tot_norm,'b')
		pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3]/f_clIII_norm,'g',alpha =0.6)
		pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_H*fl_slab_VIS_c[ind_vis_3]/f_tot_norm,'c')
		pl.text(617.3,0.4,'CaI')


		##############
		# 4 - TiO ~845 nm
		##############
		# 1) normalize
		f_in_norm = np.mean(fl_VIS[(wl_VIS > 845.50) & (wl_VIS < 846.50)])
		f_tot_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 845.5)&(wl_cl3_VIS < 846.5)]*min_chi_sq_K+fl_slab_VIS_c[(wl_cl3_VIS > 845.5)&(wl_cl3_VIS < 846.5)]*min_chi_sq_H)
		f_clIII_norm = np.mean(fl_cl3_VIS[(wl_cl3_VIS > 845.5)&(wl_cl3_VIS < 846.5)]*min_chi_sq_K)

		# 2) plot
		pl.subplot(224)
		pl.plot(wl_VIS[ind_vis]-shift,fl_VIS[ind_vis]/f_in_norm,'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
		pl.xlabel('Wavelength [nm]')
		# pl.ylabel('Flux')
		pl.axis([840,848,0.1,1.3])
		pl.plot(wl_cl3_VIS[ind_vis_3],(min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+min_chi_sq_H*fl_slab_VIS_c[ind_vis_3])/f_tot_norm,'b')
		pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3]/f_clIII_norm,'g',alpha =0.6)
		pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_H*fl_slab_VIS_c[ind_vis_3]/f_tot_norm,'c')
		pl.text(846,0.4,'TiO')



		# SAVE IT AND SHOW IT
		pl.tight_layout()
		pl.savefig(PATH_OUT+'%s_clIII_%s_photosph_%s.png'% (obj_in,min_chi_sq_cl3,nameWls))
		pl.show()
		if close:
			pl.close()

	def plotPaschen(self,CLIII =False,close = False):
		best_chi_sq = self.best_chi_sq
		min_chi_sq = self.min_chi_sq
		min_chi_sq_cl3 = self.min_chi_sq_cl3
		min_chi_sq_Av = self.min_chi_sq_Av
		min_chi_sq_T = self.min_chi_sq_T
		min_chi_sq_Ne = self.min_chi_sq_Ne
		min_chi_sq_tau = self.min_chi_sq_tau
		min_chi_sq_H = self.min_chi_sq_H
		min_chi_sq_K = self.min_chi_sq_K
		Av_list =self.Av_list
		PATH_OUT =self.PATH_OUT
		obj_in  = self.obj_in
		ind_uvb = self.ind_uvb
		ind_vis = self.ind_vis
		usedFeatures =self.usedFeatures
		classIIIreadIn = self.classIIIreadIn
		normWLandWidth = self.normWLandWidth

		wl_UVB = self.wl_UVB
		fl_UVB = self.fl_UVB_in/cardelli_extinction(wl_UVB*10.,min_chi_sq_Av, Rv=self.Rv)
		wl_VIS = self.wl_VIS
		fl_VIS = self.fl_VIS_in/cardelli_extinction(wl_VIS*10.,min_chi_sq_Av, Rv=self.Rv)
		filename_VIS =self.filename_VIS

		# load NIR file!
		filename_NIR = self.filename_NIR
		fitsTab = self.fitsTab
		if filename_NIR == None:
			filename_NIR = path+'data_final/flux_%s_nir_tell.fits' % obj_in
		if os.path.isfile(filename_NIR):
			if fitsTab == False:
				wl_NIR,fl_NIR_in,hdr_NIR=spec_readspec(filename_NIR, 'hdr')	#the 'hdr' string is there to say that I want to save the header
			else:
				wl_NIR,fl_NIR_in,hdr_NIR=readspec_phase3(filename_NIR, hdr_out='YES')
			if self.perAA == True:
				fl_NIR_in = 10*fl_NIR_in
		else:
			print('No NIR spectrum')
			wl_NIR,fl_NIR_in = np.array([0 for x in range(100)]),np.array([0 for x in range(100)])
			#sys.exit(1)

		fl_NIR = fl_NIR_in/cardelli_extinction(wl_NIR*10.,min_chi_sq_Av, Rv=Rv)

		#Load Slab!!
		if os.path.isfile(PATH_SLAB+'continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out'):
			wl_slab,fl_slab = readcol_py3(PATH_SLAB+'continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out',2,format='F,F')
		else:
		# otherwise, create it and then read it
			# first, write the input file
			f = open(PATH_ACC+'in.slab', 'w')
			outLine = min_chi_sq_T+'   '+min_chi_sq_Ne+'   '+min_chi_sq_tau
			f.write(outLine)
			#f.write(string.join([min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau],'   '))
			f.close()
			# run the C++ slab model program using the best fit parameters to calculate the slab model from 50 nm to 2477 nm (whole range)
			os.chdir(PATH_ACC)
			os.system('./hydrogen_slab')
			os.chdir(PATH)
			# read the result of the C++ program
			wl_slab,fl_slab = readcol_py3(PATH_ACC+'results/continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out',2,format='F,F')
		PATH_MY_CLASSIII = PATH+'/models_grid/RunnableGrid/'

		wl_cl3_UVB,fl_cl3_UVB,wl_cl3_VIS,fl_cl3_VIS,wl_cl3_NIR,fl_cl3_NIR,nameWls = pf.readMixClassIII(min_chi_sq_cl3,PATH_MY_CLASSIII,wlNorm =normWLandWidth[0])
		ind_uvb_3 = (wl_cl3_UVB <= max_wl_uvb)
		ind_vis_3 = (wl_cl3_VIS >= min_wl_vis)

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c= s['w'],s['f']
		else:
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c = s['w'],s['f']
			fl_slab_VIS_c = np.interp(wl_cl3_VIS,wl_slab_VIS_c,fl_slab_VIS_c )
			wl_slab_VIS_c = wl_cl3_VIS

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c= s['w'],s['f']
			#wl_slab,fl_slab = wl_slab_UVB_c,fl_slab_UVB_c
		else:
			#fluxcon = FluxConservingResampler()
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c = s['w'],s['f']
			#spec = Spectrum1D(spectral_axis=wl_slab_UVB_c, flux=fl_slab_UVB_c)
			fl_slab_UVB_c = np.interp(wl_cl3_UVB,wl_slab_UVB_c,fl_slab_UVB_c )
			wl_slab_UVB_c = wl_cl3_UVB
			#,fl_slab_UVB_c = np.array(resampledSpecUVB.spectral_axis/u.Unit('nm')), np.array(resampledSpecUVB.flux/u.Unit('erg cm-2 s-1 nm-1'))

		if CLIII == True:
			p1 = pl.figure(figsize=(10,9))
			##############
			#
			##############
			pl.subplot(211)
			pl.plot(wl_UVB[ind_uvb],fl_UVB[ind_uvb],'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
			pl.title(obj_in)#+' - Chi$^2$= %0.3e' % (min_chi_sq))
			# pl.xlabel('Wavelength [nm]')
			pl.ylabel('Flux')
			pl.axis([460,550,-1e-16,3.5*np.mean(fl_UVB[(wl_UVB < 460) & (wl_UVB > 440)])])
			pl.plot(wl_cl3_UVB[ind_uvb_3],min_chi_sq_K*fl_cl3_UVB[ind_uvb_3]+min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'b')
			pl.plot(wl_cl3_UVB[ind_uvb_3],min_chi_sq_K*fl_cl3_UVB[ind_uvb_3],'g',alpha =0.6)
			pl.plot(wl_cl3_UVB[ind_uvb_3],min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'c')

			##############
			#
			##############
			pl.subplot(212)
			pl.plot(wl_VIS[ind_vis],fl_VIS[ind_vis],'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
			pl.xlabel('Wavelength [nm]')
			pl.ylabel('Flux')
			pl.axis([550,700,-1e-16,1.5*np.mean(fl_VIS[(wl_VIS < 650) & (wl_VIS > 600)])])
			pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'b')
			pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3],'g',alpha =0.6)
			pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'c')

			pl.tight_layout()
			pl.savefig(PATH_OUT+'%s_clIII_%s_other_%s.png'% (obj_in,min_chi_sq_cl3,nameWls))
			pl.show()

		else:
			wlFeat = (usedFeatures[:,0]+usedFeatures[:,1])/2
			Xrange = np.abs(usedFeatures[:,0] - wlFeat)
			features,errors = classIIIreadIn.getFeatsAtSpt_symetricErr(min_chi_sq_cl3)
			p1 = pl.figure(figsize=(10,9))
			fit_cont = np.zeros(len(usedFeatures))
			for i in range(len(usedFeatures)):
				fit_cont[i] = min_chi_sq_K*features[i]+min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeatures[i,0],usedFeatures[i,1])[0]
			stdTerm1 = (min_chi_sq_K*errors)**2

			gridFilePlot = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeUVB/interp.npz'
			classIIIFeatPlotting = pf.classIII(gridFilePlot)
			usedFeaturesPlotting = classIIIFeatPlotting.getUsedInterpFeat()
			featuresPlot,errorsPlot = classIIIFeatPlotting.getFeatsAtSpt_symetricErr(min_chi_sq_cl3)




			gridFilePlotVIS = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeVIS/interp.npz'
			classIIIFeatPlottingVIS = pf.classIII(gridFilePlotVIS)
			usedFeaturesPlottingVIS = classIIIFeatPlottingVIS.getUsedInterpFeat()
			featuresPlotVIS,errorsPlotVIS = classIIIFeatPlottingVIS.getFeatsAtSpt_symetricErr(min_chi_sq_cl3)

			usedFeaturesPlotting = np.concatenate((usedFeaturesPlotting,usedFeaturesPlottingVIS))
			featuresPlot = np.concatenate((featuresPlot,featuresPlotVIS))
			errorsPlot = np.concatenate((errorsPlot,errorsPlotVIS))

			wlFeatPlot = (usedFeaturesPlotting[:,0]+usedFeaturesPlotting[:,1])/2
			XrangePlot = np.abs(usedFeaturesPlotting[:,0] - wlFeatPlot)

			fit_contPlot = np.zeros(len(usedFeaturesPlotting))
			for i in range(len(usedFeaturesPlotting[:,0])):
				fit_contPlot[i] = min_chi_sq_K*featuresPlot[i]+min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeaturesPlotting[i,0],usedFeaturesPlotting[i,1])[0]
			stdTerm1plot = (min_chi_sq_K*errorsPlot)**2
			stdTerm2plot = 0#(stddev_cont_dered**2)#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not taken into account!!!
			fit_stdPlot = np.sqrt(stdTerm1plot +stdTerm2plot)

			stdTerm2 = 0#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not used for the plot, this is a hold over!!!
			fit_std = np.sqrt(stdTerm1 +stdTerm2)
			##############
			#
			##############
			pl.subplot(211)
			pl.plot(wl_UVB[ind_uvb],fl_UVB[ind_uvb],'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
			pl.title(obj_in)#+' - Chi$^2$= %0.3e' % (min_chi_sq))
			# pl.xlabel('Wavelength [nm]')
			pl.ylabel('Flux')
			pl.axis([460,550,-1e-16,3.5*np.mean(fl_UVB[(wl_UVB < 460) & (wl_UVB > 440)])])
			#pl.plot(wl_cl3_UVB[ind_uvb_3],min_chi_sq_K*fl_cl3_UVB[ind_uvb_3]+min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'b')
			#pl.plot(wl_cl3_UVB[ind_uvb_3],min_chi_sq_K*fl_cl3_UVB[ind_uvb_3],'g')
			pl.plot(wl_cl3_UVB[ind_uvb_3],min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'c')
			pl.errorbar(wlFeat,min_chi_sq_K*features,min_chi_sq_K*errors,xerr = Xrange,fmt='.',c='g',zorder = 10)
			pl.errorbar(wlFeat,fit_cont,yerr=fit_std,xerr= Xrange,fmt='.',c='b',zorder = 10)

			#Additional features
			pl.plot(wlFeatPlot,min_chi_sq_K*featuresPlot ,c='tab:olive',zorder = 10)
			pl.plot(wlFeatPlot,fit_contPlot ,c='tab:cyan',zorder = 10)


			##############
			#
			##############
			pl.subplot(212)
			pl.plot(wl_VIS[ind_vis],fl_VIS[ind_vis],'r')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
			pl.xlabel('Wavelength [nm]')
			pl.ylabel('Flux')
			pl.axis([550,700,-1e-16,1.5*np.mean(fl_VIS[(wl_VIS < 650) & (wl_VIS > 600)])])
			#pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'b')
			#pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3],'g')
			pl.errorbar(wlFeat,min_chi_sq_K*features,min_chi_sq_K*errors,xerr = Xrange,fmt='.',c='g',zorder = 10)
			pl.errorbar(wlFeat,fit_cont,yerr=fit_std,xerr= Xrange,fmt='.',c='b',zorder = 10)

			#Additional features
			pl.plot(wlFeatPlot,min_chi_sq_K*featuresPlot ,c='tab:olive',zorder = 10)
			pl.plot(wlFeatPlot,fit_contPlot ,c='tab:cyan',zorder = 10)


			pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'c')

			pl.tight_layout()
			pl.savefig(PATH_OUT+'%s_clIII_%s_other_InterpFeat.png'% (obj_in,min_chi_sq_cl3))
			pl.show()
		if close:
			pl.close()


	def plotAll(self,CLIII = False,smooth = False, close = False):
		best_chi_sq = self.best_chi_sq
		min_chi_sq = self.min_chi_sq
		min_chi_sq_cl3 = self.min_chi_sq_cl3
		min_chi_sq_Av = self.min_chi_sq_Av
		min_chi_sq_T = self.min_chi_sq_T
		min_chi_sq_Ne = self.min_chi_sq_Ne
		min_chi_sq_tau = self.min_chi_sq_tau
		min_chi_sq_H = self.min_chi_sq_H
		min_chi_sq_K = self.min_chi_sq_K
		Av_list =self.Av_list
		PATH_OUT =self.PATH_OUT
		obj_in  = self.obj_in
		ind_uvb = self.ind_uvb
		ind_vis = self.ind_vis
		usedFeatures =self.usedFeatures
		classIIIreadIn = self.classIIIreadIn
		normWLandWidth = self.normWLandWidth

		wl_UVB = self.wl_UVB
		fl_UVB = self.fl_UVB_in/cardelli_extinction(wl_UVB*10.,min_chi_sq_Av, Rv=self.Rv)
		wl_VIS = self.wl_VIS
		fl_VIS = self.fl_VIS_in/cardelli_extinction(wl_VIS*10.,min_chi_sq_Av, Rv=self.Rv)
		filename_VIS =self.filename_VIS

		# load NIR file!
		filename_NIR = self.filename_NIR
		fitsTab = self.fitsTab
		if filename_NIR == None:
			filename_NIR = path+'data_final/flux_%s_nir_tell.fits' % obj_in
		if os.path.isfile(filename_NIR):
			if fitsTab == False:
				wl_NIR,fl_NIR_in,hdr_NIR=spec_readspec(filename_NIR, 'hdr')	#the 'hdr' string is there to say that I want to save the header
			else:
				wl_NIR,fl_NIR_in,hdr_NIR=readspec_phase3(filename_NIR, hdr_out='YES')
			if self.perAA == True:
				fl_NIR_in = 10*fl_NIR_in
		else:
			print('No NIR spectrum')
			wl_NIR,fl_NIR_in = np.array([0 for x in range(100)]),np.array([0 for x in range(100)])
			#sys.exit(1)

		fl_NIR = fl_NIR_in/cardelli_extinction(wl_NIR*10.,min_chi_sq_Av, Rv=Rv)

		#Load Slab!!
		if os.path.isfile(PATH_SLAB+'continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out'):
			wl_slab,fl_slab = readcol_py3(PATH_SLAB+'continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out',2,format='F,F')
		else:
		# otherwise, create it and then read it
			# first, write the input file
			f = open(PATH_ACC+'in.slab', 'w')
			outLine = min_chi_sq_T+'   '+min_chi_sq_Ne+'   '+min_chi_sq_tau
			f.write(outLine)
			#f.write(string.join([min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau],'   '))
			f.close()
			# run the C++ slab model program using the best fit parameters to calculate the slab model from 50 nm to 2477 nm (whole range)
			os.chdir(PATH_ACC)
			os.system('./hydrogen_slab')
			os.chdir(PATH)
			# read the result of the C++ program
			wl_slab,fl_slab = readcol_py3(PATH_ACC+'results/continuum_tot_T'+min_chi_sq_T+'_ne'+min_chi_sq_Ne+'tau'+min_chi_sq_tau+'.out',2,format='F,F')
		PATH_MY_CLASSIII = PATH+'/models_grid/RunnableGrid/'

		wl_cl3_UVB,fl_cl3_UVB,wl_cl3_VIS,fl_cl3_VIS,wl_cl3_NIR,fl_cl3_NIR,nameWls = pf.readMixClassIII(min_chi_sq_cl3,PATH_MY_CLASSIII,wlNorm =normWLandWidth[0])
		ind_uvb_3 = (wl_cl3_UVB <= max_wl_uvb)
		ind_vis_3 = (wl_cl3_VIS >= min_wl_vis)

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c= s['w'],s['f']
		else:
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_VIS.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_VIS_c,fl_slab_VIS_c = s['w'],s['f']
			fl_slab_VIS_c = np.interp(wl_cl3_VIS,wl_slab_VIS_c,fl_slab_VIS_c )
			wl_slab_VIS_c = wl_cl3_VIS

		pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,nameWls)
		if os.path.isfile(pathSlab):
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c= s['w'],s['f']
			#wl_slab,fl_slab = wl_slab_UVB_c,fl_slab_UVB_c
		else:
			#fluxcon = FluxConservingResampler()
			pathSlab = PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_UVB.sav' % (min_chi_sq_T,min_chi_sq_Ne,min_chi_sq_tau,'TWA7')
			s = readsav(pathSlab)
			wl_slab_UVB_c,fl_slab_UVB_c = s['w'],s['f']
			#spec = Spectrum1D(spectral_axis=wl_slab_UVB_c, flux=fl_slab_UVB_c)
			fl_slab_UVB_c = np.interp(wl_cl3_UVB,wl_slab_UVB_c,fl_slab_UVB_c )
			wl_slab_UVB_c = wl_cl3_UVB
			#,fl_slab_UVB_c = np.array(resampledSpecUVB.spectral_axis/u.Unit('nm')), np.array(resampledSpecUVB.flux/u.Unit('erg cm-2 s-1 nm-1'))

		p1 = pl.figure(figsize=(10,7))
		#pl.rcParams.update({'font.size': 13})
		if smooth == False:
			pl.plot (wl_UVB[ind_uvb],fl_UVB[ind_uvb],'r')
			pl.plot(wl_VIS[ind_vis],fl_VIS[ind_vis],'r')
			pl.plot(wl_NIR,fl_NIR,'r')
		else:

			##############
			#
			##############
			wl_UVB_smooth = wl_UVB[ind_uvb][::12]
			fl_UVB_smooth = spectrum_resample(fl_UVB,wl_UVB,wl_UVB_smooth)
			wl_VIS_smooth = wl_VIS[ind_vis][::12]
			fl_VIS_smooth = spectrum_resample(fl_VIS,wl_VIS,wl_VIS_smooth)
			wl_NIR_smooth = wl_NIR[::12 ]
			fl_NIR_smooth = spectrum_resample(fl_NIR,wl_NIR,wl_NIR_smooth)
			pl.plot(wl_UVB_smooth,fl_UVB_smooth,'r')
			pl.plot(wl_VIS_smooth,fl_VIS_smooth,'r')
			pl.plot(wl_NIR_smooth,fl_NIR_smooth,'r')

		if CLIII == True:
			"""
			PLOT WHOLE SPECTRUM
			"""

			##############
			#
			##############

			pl.title(obj_in)#+' - Chi$^2$= %0.3e' % (min_chi_sq))
			pl.xlabel('Wavelength [nm]')
			pl.ylabel('Flux')
			pl.axis([330,2048,0.1*np.median(fl_UVB[(wl_UVB < 450) & (wl_UVB > 400)]),4*np.median(fl_VIS[(wl_VIS < 820) & (wl_VIS > 800)])])
			pl.plot(wl_cl3_UVB[ind_uvb_3],min_chi_sq_K*fl_cl3_UVB[ind_uvb_3]+min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'b',alpha =0.5)
			pl.plot(wl_cl3_UVB[ind_uvb_3],min_chi_sq_K*fl_cl3_UVB[ind_uvb_3],'g',alpha =0.6)
			pl.plot(wl_cl3_UVB[ind_uvb_3],min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'c')

			pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'b',alpha =0.5)
			print(min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+min_chi_sq_H*fl_slab_VIS_c[ind_vis_3])
			pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3],'g',alpha =0.6)
			pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'c')

			pl.plot(wl_cl3_NIR,min_chi_sq_K*fl_cl3_NIR,'g',alpha =0.6)

			pl.xscale('log')
			pl.yscale('log')

			pl.tight_layout()
			pl.savefig(PATH_OUT+'%s_clIII_%s_ALL_%s.png'% (obj_in,min_chi_sq_cl3,nameWls))
			pl.show()

		else:
			"""
			PLOT WHOLE SPECTRUM WITH INTERP FEAT
			"""
			wlFeat = (usedFeatures[:,0]+usedFeatures[:,1])/2
			Xrange = np.abs(usedFeatures[:,0] - wlFeat)
			features,errors = classIIIreadIn.getFeatsAtSpt_symetricErr(min_chi_sq_cl3)
			#p1 = pl.figure(5,figsize=(10,9))
			fit_cont = np.zeros(len(usedFeatures))
			for i in range(len(usedFeatures)):
				fit_cont[i] = min_chi_sq_K*features[i]+min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeatures[i,0],usedFeatures[i,1])[0]
			stdTerm1 = (min_chi_sq_K*errors)**2

			gridFilePlot = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeUVB/interp.npz'
			classIIIFeatPlotting = pf.classIII(gridFilePlot)
			usedFeaturesPlotting = classIIIFeatPlotting.getUsedInterpFeat()
			featuresPlot,errorsPlot = classIIIFeatPlotting.getFeatsAtSpt_symetricErr(min_chi_sq_cl3)




			gridFilePlotVIS = PATH+'/models_grid/Interpolations/earlyK_norm731_200p_1000iter_rad2.5_WholeVIS/interp.npz'
			classIIIFeatPlottingVIS = pf.classIII(gridFilePlotVIS)
			usedFeaturesPlottingVIS = classIIIFeatPlottingVIS.getUsedInterpFeat()
			featuresPlotVIS,errorsPlotVIS = classIIIFeatPlottingVIS.getFeatsAtSpt_symetricErr(min_chi_sq_cl3)

			usedFeaturesPlotting = np.concatenate((usedFeaturesPlotting,usedFeaturesPlottingVIS))
			featuresPlot = np.concatenate((featuresPlot,featuresPlotVIS))
			errorsPlot = np.concatenate((errorsPlot,errorsPlotVIS))

			wlFeatPlot = (usedFeaturesPlotting[:,0]+usedFeaturesPlotting[:,1])/2
			XrangePlot = np.abs(usedFeaturesPlotting[:,0] - wlFeatPlot)

			fit_contPlot = np.zeros(len(usedFeaturesPlotting))
			for i in range(len(usedFeaturesPlotting[:,0])):
				fit_contPlot[i] = min_chi_sq_K*featuresPlot[i]+min_chi_sq_H*compute_flux_inRange(wl_slab,fl_slab,usedFeaturesPlotting[i,0],usedFeaturesPlotting[i,1])[0]
			stdTerm1plot = (min_chi_sq_K*errorsPlot)**2
			stdTerm2plot = 0#(stddev_cont_dered**2)#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not taken into account!!!
			fit_stdPlot = np.sqrt(stdTerm1plot +stdTerm2plot)

			stdTerm2 = 0#(stddev_cont_dered**2) # the STD on the deredenned spectrum is not used for the plot, this is a hold over!!!
			fit_std = np.sqrt(stdTerm1 +stdTerm2)
			#p1 = pl.figure(figsize=(10,7))
			##############
			#
			##############

			pl.title(obj_in)#+' - Chi$^2$= %0.3e' % (min_chi_sq))
			pl.xlabel('Wavelength [nm]')
			pl.ylabel('Flux')
			pl.axis([330,2048,0.1*np.median(fl_UVB[(wl_UVB < 450) & (wl_UVB > 400)]),4*np.median(fl_VIS[(wl_VIS < 820) & (wl_VIS > 800)])])
			#pl.loglog(wl_cl3_UVB[ind_uvb_3],min_chi_sq_K*fl_cl3_UVB[ind_uvb_3]+min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'b')
			#pl.loglog(wl_cl3_UVB[ind_uvb_3],min_chi_sq_K*fl_cl3_UVB[ind_uvb_3],'g')
			pl.plot(wl_cl3_UVB[ind_uvb_3],min_chi_sq_H*fl_slab_UVB_c[ind_uvb_3],'c')

			#pl.loglog(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3]+min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'b')
			#pl.loglog(wl_cl3_VIS[ind_vis_3],min_chi_sq_K*fl_cl3_VIS[ind_vis_3],'g')
			pl.plot(wl_cl3_VIS[ind_vis_3],min_chi_sq_H*fl_slab_VIS_c[ind_vis_3],'c')

			#pl.loglog(wl_cl3_NIR,min_chi_sq_K*fl_cl3_NIR,'g')

			#Used feat
			pl.errorbar(wlFeat,min_chi_sq_K*features,min_chi_sq_K*errors,xerr = Xrange,fmt='.',c='g',alpha =0.6,zorder = 10)
			pl.errorbar(wlFeat,fit_cont,yerr=fit_std,xerr= Xrange,fmt='.',c='b',zorder = 10)

			#Additional features
			pl.plot(wlFeatPlot,min_chi_sq_K*featuresPlot ,c='tab:olive',zorder = 10)
			pl.plot(wlFeatPlot,fit_contPlot ,c='tab:cyan',zorder = 10)


			#pl.errorbar(wlFeatPlot_NIR,min_chi_sq_K*featuresPlot_NIR,min_chi_sq_K*errorsPlot_NIR,xerr = XrangePlot_NIR,fmt='.',c='tab:olive',zorder = 10)
			#pl.errorbar(wlFeatPlot_NIR,fit_contPlot_NIR,yerr=fit_stdPlot_NIR,xerr= XrangePlot_NIR,fmt='.',c='tab:cyan',zorder = 10)

			pl.xscale('log')
			pl.yscale('log')

			pl.tight_layout()
			pl.show()
			pl.savefig(PATH_OUT+'%s_clIII_%s_ALL_INTERP_FEAT.png'% (obj_in,min_chi_sq_cl3))
		if close:
			pl.close()

	"""
	CHI2 ANALYSIS
	"""
	def Chi2SpT(self,close =False):
		chi_sq = self.chi_sq
		H_fin = self.H_fin
		K_fin = self.K_fin
		cl3_in_list = self.cl3_in_list
		obj_in =self.obj_in
		min_chi_sq_cl3 = self.min_chi_sq_cl3
	# DeltaChi2 vs SpT
		if len(cl3_in_list) > 1:
			chi_sq_distr_cl3(chi_sq,H_fin,K_fin,cl3_in_list,PATH)
			tit = '%s - %s' % (obj_in,min_chi_sq_cl3)
			pl.title(tit)
			pl.tight_layout()
			pl.savefig(self.PATH_OUT+'%s_clIII_%s_cl3_chi2.png'% (obj_in,min_chi_sq_cl3))
			pl.show()
		if close:
			pl.close()

	# DeltaChi2 vs Av
	def Chi2Av(self,close =False):
		chi_sq = self.chi_sq
		H_fin = self.H_fin
		K_fin = self.K_fin
		Av_list = self.Av_list
		min_chi_sq_Av = self.min_chi_sq_Av
		min_chi_sq_cl3 = self.min_chi_sq_cl3
		obj_in =self.obj_in
		if len(Av_list) > 1:
			chi_sq_distr_Av(chi_sq,H_fin,K_fin,Av_list,min_chi_sq_Av)
			tit = '%s - %s' % (obj_in,min_chi_sq_cl3)
			pl.title(tit)
			pl.tight_layout()
			pl.savefig(self.PATH_OUT+'%s_clIII_%s_Av_chi2.png'% (obj_in,min_chi_sq_cl3))
			pl.show()
		if close:
			pl.close()

	# DeltaChi2 surface vs Av and Teff
	def Chi2AvAndSpT(self,close =False):
		chi_sq = self.chi_sq
		H_fin = self.H_fin
		K_fin = self.K_fin
		cl3_in_list = self.cl3_in_list
		Av_list = self.Av_list
		obj_in =self.obj_in
		min_chi_sq_cl3 = self.min_chi_sq_cl3
		#min_chi_sq_Av = self.min_chi_sq_Av
		if len(cl3_in_list) > 2 and len(Av_list) > 2:
			chi_sq_distr_cl3_Av(chi_sq,H_fin,K_fin,cl3_in_list,Av_list,PATH)
			# pl.plot([min_chi_sq_Av],[4350],'ro')
			tit = '%s - %s' % (obj_in,min_chi_sq_cl3)
			pl.title(tit)
			pl.savefig(self.PATH_OUT+'%s_clIII_%s_Av_SpT_chi2.png'% (obj_in,min_chi_sq_cl3))
			#pl.show()
		if close:
			pl.close()

	def Chi2AvAndSpT_Posterior(self,close =False):
		chi_sq = self.chi_sq
		H_fin = self.H_fin
		K_fin = self.K_fin
		cl3_in_list = self.cl3_in_list
		Av_list = self.Av_list
		obj_in =self.obj_in
		min_chi_sq_cl3 = self.min_chi_sq_cl3
		#min_chi_sq_Av = self.min_chi_sq_Av
		if len(cl3_in_list) > 2 and len(Av_list) > 2:
			posterior_distr_cl3_Av(chi_sq,H_fin,K_fin,cl3_in_list,Av_list,PATH)
			# pl.plot([min_chi_sq_Av],[4350],'ro')
			tit = '%s - %s' % (obj_in,min_chi_sq_cl3)
			pl.title(tit)
			pl.savefig(self.PATH_OUT+'%s_clIII_%s_Av_SpT_chi2_posterior.png'% (obj_in,min_chi_sq_cl3))
			#pl.show()
		if close:
			pl.close()
