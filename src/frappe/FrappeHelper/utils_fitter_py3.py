#!/usr/bin/env python

import numpy as np
import matplotlib as mp
import pylab as pl
from spec_readspec import *
from spectrum_resample import *
import time
import sys
import string
import os
#from linfit import *
from scipy.io.idl import readsav
from readcol_py3 import *




# FUNCTION to read slab models produced with the C++ program AND RESAMPLED according to a particular ClassIII
# ---- THIS ONE CHECKES FIRST IF THERE IS THE SAV FILE, OTHERWISE IT LOOKS FOR THE DAT ONE, but there is no possibility to do a new model
def read_slab_check(T_in,Ne_in,tau_in,cl3_inn,ARM_in,PATH_SLAB_RESAMPLED_SAV,PATH_SLAB_RESAMPLED):
#read a slab model spectrum already resampled to match the Class 3 wavelength scale
	if os.path.isfile(PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_%s.sav' % (T_in,Ne_in,tau_in,cl3_inn,ARM_in)):
		print('READ SAV FILE - %s' % ARM_in)
		s = readsav(PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_%s.sav' % (T_in,Ne_in,tau_in,cl3_inn,ARM_in))
		# print(w)
		wl_slab,fl_slab = s['w'],s['f']
	elif os.path.isfile(PATH_SLAB_RESAMPLED+'slab_T%s_ne%s_tau%s_clIII_%s_%s.dat' % (T_in,Ne_in,tau_in,cl3_inn,ARM_in)):
		#print('Leggiamo il file')
		#read the UVB and VIS files of the resampled slab model
		wl_slab,fl_slab = read_resampled_slab(T_in,Ne_in,tau_in,cl3_inn,ARM_in,PATH_SLAB_RESAMPLED)
		#print('Letto')
	else:
		sys.exit('slab model not available.')
	return np.array(wl_slab,dtype=np.float64),np.array(fl_slab,dtype=np.float64)
# ------------------------------------------------------------


# FUNCTION to read slab models produced with the C++ program AND RESAMPLED according to a particular ClassIII
# ---- THIS ONE CHECKES FIRST IF THERE IS THE SAV FILE, OTHERWISE IT LOOKS FOR THE DAT ONE
def read_slab(T_in,Ne_in,tau_in,cl3_inn,ARM_in,PATH_SLAB_RESAMPLED_SAV,PATH_SLAB_RESAMPLED,PATH_ACC,wl_cl3):
#read a slab model spectrum already resampled to match the Class 3 wavelength scale
	if os.path.isfile(PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_%s.sav' % (T_in,Ne_in,tau_in,cl3_inn,ARM_in)):
		print('READ SAV FILE - %s' % ARM_in)
		s = readsav(PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_%s.sav' % (T_in,Ne_in,tau_in,cl3_inn,ARM_in))
		# print(w)
		wl_slab,fl_slab = s['w'],s['f']
	elif os.path.isfile(PATH_SLAB_RESAMPLED+'slab_T%s_ne%s_tau%s_clIII_%s_%s.dat' % (T_in,Ne_in,tau_in,cl3_inn,ARM_in)):
		#print('Leggiamo il file')
		#read the UVB and VIS files of the resampled slab model
		wl_slab,fl_slab = read_resampled_slab(T_in,Ne_in,tau_in,cl3_inn,ARM_in,PATH_SLAB_RESAMPLED)
		#print('Letto')
	else:
		# if it's not there, you have to create it, but this is slower!!!
		time_init = time.time()
		ARM = ARM_in
		# first, write the input file
		f = open(PATH_ACC+'in.slab', 'w')
		f.write('   '.join([T_in,Ne_in,tau_in]))
		f.close()
		# second, write the wavelengths file
		f = open(PATH_ACC+'wavel_user_def.dat', 'w')
		for wl in wl_cl3:
			f.write(str(wl)+'\n')
		f.close()
		# run the C++ slab model program using the best fit parameters to calculate the slab model from 50 nm to 2477 nm (whole range)
		os.chdir(PATH_ACC)
		os.system('./hydrogen_slab_wvl')
		# os.chdir(PATH)
		# copy the slab model to a file called slab_T####_ne#e##_tau#.#_clIII_[name]_[ARM].dat
		print('COPY FILE'	)
		os.system('cp '+PATH_ACC+'/results/continuum_tot_T'+T_in+'_ne'+Ne_in+'tau'+tau_in+'.out '\
			+PATH_SLAB_RESAMPLED+'slab_T%s_ne%s_tau%s_clIII_%s_%s.dat' % (T_in,Ne_in,tau_in,cl3_inn,ARM))
		print(time.clock())
		# print('IT TOOK ',time.time()-time_init_res,'s TO CREATE IT')
		wl_slab,fl_slab = read_resampled_slab(T_in,Ne_in,tau_in,cl3_inn,ARM_in,PATH_SLAB_RESAMPLED)

	return np.array(wl_slab,dtype=np.float64),np.array(fl_slab,dtype=np.float64)
# ------------------------------------------------------------

# FUNCTION to read slab models produced with the C++ program AND RESAMPLED according to a particular ClassIII
# ---- THIS ONE ASSUMES THERE IS THE SAV FILE
def read_slab_sav_RC(T_in,Ne_in,tau_in,cl3_inn,ARM_in,PATH_SLAB_RESAMPLED_SAV):
#read a slab model spectrum already resampled to match the Class 3 wavelength scale
	s = readsav(PATH_SLAB_RESAMPLED_SAV+'slab_T%s_ne%s_tau%s_clIII_%s_%s.sav' % (T_in,Ne_in,tau_in,cl3_inn,ARM_in))
	wl_slab,fl_slab = s['w'],s['f']

	return np.array(wl_slab,dtype=np.float64),np.array(fl_slab,dtype=np.float64)
# ------------------------------------------------------------


# FUNCTION to read slab models produced with the C++ program AND RESAMPLED according to a particular ClassIII
def read_resampled_slab(T_in,Ne_in,tau_in,cl3_inn,ARM_in,PATH_IN):
	ntoskip = 9 # the first 9 lines should be skipped, they are only explanatory lines
	wl = np.array([])
	fl = np.array([],dtype=np.float64)
	nskip = 0
	nline = 0
	for line in open(PATH_IN+'slab_T%s_ne%s_tau%s_clIII_%s_%s.dat' % (T_in,Ne_in,tau_in,cl3_inn,ARM_in)):
		if nskip < ntoskip:
			line.split()
			nskip+=1
		else:
			wl = np.append(wl,line.split()[0])
			fl = np.append(fl,line.split()[1])
			nline+=1
	return np.array(wl,dtype=np.float64),np.array(fl,dtype=np.float64)
# ------------------------------------------------------------

# CHANGED ON 2015-12-22 exchanging the mean with the median (outlier resistant)
# # FUNCTION to compute the flux at a given wavelength of a spectrum (wl in nm!!!)
# # if not provided, the wl interval on which the mean should be computed is assumed to be 8 nm
# # centered on the nominal wavelength
# def compute_flux_at_wl_nostd(wl_in,fl_in,wl0,interval=8):
# 	ind = (wl_in >= (wl0-interval*0.5)) & (wl_in <= (wl0+interval*0.5))
# 	flux_at = np.mean(fl_in[ind], dtype=np.float64)
# 	return flux_at
# # ------------------------------------------------------------

# # FUNCTION to compute the flux at a given wavelength of a spectrum (wl in nm!!!)
# # if not provided, the wl interval on which the mean should be computed is assumed to be 8 nm
# # centered on the nominal wavelength
# def compute_flux_at_wl_std(wl_in,fl_in,wl0,interval=8):
# 	ind = (wl_in >= (wl0-interval*0.5)) & (wl_in <= (wl0+interval*0.5))
# 	flux_at = np.mean(fl_in[ind], dtype=np.float64)
# 	stddev_at = np.std(fl_in[ind], dtype=np.float64)
# 	return flux_at,stddev_at
# # ------------------------------------------------------------

# # FUNCTION to compute the flux at a given wavelength of a spectrum (wl in nm!!!)
# # if not provided, the wl interval on which the mean should be computed is assumed to be 8 nm
# # centered on the nominal wavelength
# def compute_flux_at_wl(wl_in,fl_in,wl0,interval=8,stddev='NO'):
# 	ind = (wl_in >= (wl0-interval*0.5)) & (wl_in <= (wl0+interval*0.5))
# 	flux_at = np.mean(fl_in[ind], dtype=np.float64)
# 	if stddev != 'NO':
# 		stddev_at = np.std(fl_in[ind], dtype=np.float64)
# 		return flux_at,stddev_at
# 	else:
# 		return flux_at
# # ------------------------------------------------------------

# FUNCTION to compute the flux at a given wavelength of a spectrum (wl in nm!!!)
# if not provided, the wl interval on which the mean should be computed is assumed to be 8 nm
# centered on the nominal wavelength
def compute_flux_at_wl_nostd(wl_in,fl_in,wl0,interval=8):
	ind = (wl_in >= (wl0-interval*0.5)) & (wl_in <= (wl0+interval*0.5))
	flux_at = np.median(fl_in[ind])
	return flux_at
# ------------------------------------------------------------

# FUNCTION to compute the flux at a given wavelength of a spectrum (wl in nm!!!)
# if not provided, the wl interval on which the median should be computed is assumed to be 8 nm
# centered on the nominal wavelength
def compute_flux_at_wl_std(wl_in,fl_in,wl0,interval=8):
	ind = (wl_in >= (wl0-interval*0.5)) & (wl_in <= (wl0+interval*0.5))
	flux_at = np.nanmedian(fl_in[ind])
	stddev_at = np.std(fl_in[ind], dtype=np.float64)
	return flux_at,stddev_at

def compute_flux_inRange(wl_in,fl_in,wl1,wl2):
	#wl_in= np.vstack([wl_in]*len(wl1))
	#print('bla')
	ind = (wl_in >= (wl1)) & (wl_in <= (wl2))
	flux_at = np.median(fl_in[ind])
	stddev_at = np.std(fl_in[ind], dtype=np.float64)
	return flux_at,stddev_at
# ------------------------------------------------------------

# FUNCTION to compute the flux at a given wavelength of a spectrum (wl in nm!!!)
# if not provided, the wl interval on which the median should be computed is assumed to be 8 nm
# centered on the nominal wavelength
def compute_flux_at_wl(wl_in,fl_in,wl0,interval=8,stddev='NO'):
	ind = (wl_in >= (wl0-interval*0.5)) & (wl_in <= (wl0+interval*0.5))
	flux_at = np.median(fl_in[ind])
	if stddev != 'NO':
		stddev_at = np.std(fl_in[ind], dtype=np.float64)
		return flux_at,stddev_at
	else:
		return flux_at
# ------------------------------------------------------------

# FUNCTION to compute the flux at ~360 nm
def compute_cont_360_nostd(wl_in,fl_in):
	flux_at_355 = compute_flux_at_wl_nostd(wl_in,fl_in,355,interval=6)
	return flux_at_355
# ------------------------------------------------------------

# FUNCTION to compute the flux at ~360 nm
def compute_cont_360_std(wl_in,fl_in):
	flux_at_355,stddev_at_355 = compute_flux_at_wl_std(wl_in,fl_in,355,interval=6)
	return flux_at_355,stddev_at_355
# ------------------------------------------------------------

# FUNCTION to compute the flux at ~360 nm
def compute_cont_360(wl_in,fl_in,stddev='NO'):
	if stddev != 'NO':
		flux_at_355,stddev_at_355 = compute_flux_at_wl(wl_in,fl_in,355,interval=6,stddev='YES')
		return flux_at_355,stddev_at_355
	else:
		flux_at_355 = compute_flux_at_wl(wl_in,fl_in,355,interval=6)
		return flux_at_355
# ------------------------------------------------------------

# FUNCTION to compute the flux at ~460 nm
def compute_cont_460_nostd(wl_in,fl_in):
	flux_at_461 = compute_flux_at_wl_nostd(wl_in,fl_in,461,interval=3)
	return flux_at_461
# ------------------------------------------------------------

# FUNCTION to compute the flux at ~460 nm
def compute_cont_460_std(wl_in,fl_in):
	flux_at_461,stddev_at_461 = compute_flux_at_wl_std(wl_in,fl_in,461,interval=3)
	return flux_at_461,stddev_at_461
# ------------------------------------------------------------

# FUNCTION to compute the flux at ~460 nm
def compute_cont_460(wl_in,fl_in,stddev='NO'):
	if stddev != 'NO':
		flux_at_461,stddev_at_461 = compute_flux_at_wl(wl_in,fl_in,461,interval=3,stddev='YES')
		return flux_at_461,stddev_at_461
	else:
		flux_at_461 = compute_flux_at_wl(wl_in,fl_in,461,interval=3)
		return flux_at_461
# ------------------------------------------------------------

# # OLD FUNCTION to compute the flux at ~710 nm ---- NOT IN USE ANYMORE
# def compute_cont_710(wl_in,fl_in,stddev='NO'):
# 	ind = ((wl_in >= 702) & (wl_in <= 704)) | ((wl_in >= 714) & (wl_in <= 718))
# 	# flux_at_703 = compute_flux_at_wl(wl_in,fl_in,703,interval=2)
# 	# flux_at_716 = compute_flux_at_wl(wl_in,fl_in,716,interval=4)
# 	# mean_flux = np.mean([flux_at_703,flux_at_716],dtype=np.float64)
# 	mean_flux = np.mean(fl_in[ind],dtype=np.float64)
# 	if stddev != 'NO':
# 		stddev_at = np.std(fl_in[ind], dtype=np.float64)
# 		return mean_flux,stddev_at
# 	else:
# 		return mean_flux
# # ------------------------------------------------------------


# FUNCTION to compute the Balmer jump ratio
def compute_BJ_nostd(wl_in,fl_in):
	flux_at_360,stddev_at_360 = compute_flux_at_wl_std(wl_in,fl_in,357.5,interval=5)
	flux_at_400,stddev_at_400 = compute_flux_at_wl_std(wl_in,fl_in,400,interval=4)
	BJ_calc = flux_at_360/flux_at_400
	return BJ_calc
# ------------------------------------------------------------

# FUNCTION to compute the Balmer jump ratio
def compute_BJ_std(wl_in,fl_in):
	flux_at_360,stddev_at_360 = compute_flux_at_wl_std(wl_in,fl_in,357.5,interval=5)
	flux_at_400,stddev_at_400 = compute_flux_at_wl_std(wl_in,fl_in,400,interval=4)
	BJ_calc = flux_at_360/flux_at_400
	stddev_at = BJ_calc * np.sqrt((stddev_at_360/flux_at_360)**2 + (stddev_at_400/flux_at_400)**2)
	return BJ_calc,stddev_at
# ------------------------------------------------------------

# FUNCTION to compute the Balmer jump ratio
def compute_BJ(wl_in,fl_in,stddev='NO'):
	flux_at_360,stddev_at_360 = compute_flux_at_wl(wl_in,fl_in,357.5,interval=5,stddev='yes')
	flux_at_400,stddev_at_400 = compute_flux_at_wl(wl_in,fl_in,400,interval=4,stddev='yes')
	BJ_calc = flux_at_360/flux_at_400
	if stddev != 'NO':
		stddev_at = BJ_calc * np.sqrt((stddev_at_360/flux_at_360)**2 + (stddev_at_400/flux_at_400)**2)
		return BJ_calc,stddev_at
	else:
		return BJ_calc
# ------------------------------------------------------------

# ------------------------------------------------------------

# # FUNCTION to compute the Balmer continuum slope
# def compute_balmer_cont(wl_in,fl_in,stddev='NO'):
# 	wl = wl_in[np.where( (wl_in > 330.) & (wl_in < 360.) )]
# 	fl = fl_in[np.where( (wl_in > 330.) & (wl_in < 360.) )]
# 	# assume an error on the flux of 5%
# 	BC_slope_calc,BC_slope_err = linfit(wl,fl,fl*0.1,sigma='YES')
# 	if stddev != 'NO':
# 		return BC_slope_calc[0], BC_slope_err[0]
# 	else:
# 		return BC_slope_calc[0]
# # ------------------------------------------------------------


# # FUNCTION to compute the Paschen continuum slope
# def compute_paschen_cont(wl_in,fl_in,stddev='NO'):
# 	wl = wl_in[np.where( (wl_in > 400.) & (wl_in < 476.) )]
# 	fl = fl_in[np.where( (wl_in > 400.) & (wl_in < 476.) )]
# 	# assume an error on the flux of 5%
# 	PC_slope_calc,PC_slope_err = linfit(wl,fl,fl*0.1,sigma='YES')
# 	if stddev != 'NO':
# 		return PC_slope_calc[0], PC_slope_err[0]
# 	else:
# 		return PC_slope_calc[0]
# # ------------------------------------------------------------

# # FUNCTION to compute the Chi2 of the fit ---- OLD VERSION
# def chi_squared(BJ_obs_in,BC_obs_in,PC_obs_in,c360_obs_in,c710_obs_in,BJ_stddev,BC_stddev,PC_stddev,c360_stddev,c710_stddev,\
# 	BJ_fit_in,BC_fit_in,PC_fit_in,c360_fit_in,c710_fit_in):
# 	chi_sq = ((BJ_obs_in - BJ_fit_in)/BJ_stddev)**2 + ((BC_obs_in - BC_fit_in)/BC_stddev)**2 + ((PC_obs_in - PC_fit_in)/PC_stddev)**2 + \
# 			((c360_obs_in - c360_fit_in)/c360_stddev)**2 + ((c710_obs_in - c710_fit_in)/c710_stddev)**2
# 	return chi_sq
# # ------------------------------------------------------------

# # FUNCTION to compute the Chi2 of the fit using also cont @ 460 nm ---- OLD VERSION
# def chi_squared_bis(BJ_obs_in,BC_obs_in,PC_obs_in,c360_obs_in,c460_obs_in,c710_obs_in,BJ_stddev,BC_stddev,PC_stddev,c360_stddev,c460_stddev,c710_stddev,\
# 	BJ_fit_in,BC_fit_in,PC_fit_in,c360_fit_in,c460_fit_in,c710_fit_in):
# 	chi_sq = ((BJ_obs_in - BJ_fit_in)/BJ_stddev)**2 + ((BC_obs_in - BC_fit_in)/BC_stddev)**2 + ((PC_obs_in - PC_fit_in)/PC_stddev)**2 + \
# 			((c360_obs_in - c360_fit_in)/c360_stddev)**2 + ((c460_obs_in - c460_fit_in)/c460_stddev)**2 + ((c710_obs_in - c710_fit_in)/c710_stddev)**2
# 	return chi_sq
# # ------------------------------------------------------------

# FUNCTION to compute the Chi2 of the fit
def chi_squared(BJ_obs_in,BC_obs_in,PC_obs_in,c360_obs_in,c703_obs_in,c707_obs_in,c710_obs_in,c715_obs_in,\
	BJ_stddev,BC_stddev,PC_stddev,c360_stddev,c703_stddev,c707_stddev,c710_stddev,c715_stddev,\
	BJ_fit_in,BC_fit_in,PC_fit_in,c360_fit_in,c703_fit_in,c707_fit_in,c710_fit_in,c715_fit_in):
	chi_sq = ((BJ_obs_in - BJ_fit_in)/BJ_stddev)**2 + ((BC_obs_in - BC_fit_in)/BC_stddev)**2 + ((PC_obs_in - PC_fit_in)/PC_stddev)**2 + \
			((c360_obs_in - c360_fit_in)/c360_stddev)**2 + ((c703_obs_in - c703_fit_in)/c703_stddev)**2 + ((c707_obs_in - c707_fit_in)/c707_stddev)**2 \
			+ ((c710_obs_in - c710_fit_in)/c710_stddev)**2 + ((c715_obs_in - c715_fit_in)/c715_stddev)**2
	return chi_sq
# ------------------------------------------------------------

# FUNCTION to compute the Chi2 of the fit using also cont @ 460 nm
def chi_squared_bis(BJ_obs_in,BC_obs_in,PC_obs_in,c360_obs_in,c460_obs_in,c703_obs_in,c707_obs_in,c710_obs_in,c715_obs_in,\
	BJ_stddev,BC_stddev,PC_stddev,c360_stddev,c460_stddev,c703_stddev,c707_stddev,c710_stddev,c715_stddev,\
	BJ_fit_in,BC_fit_in,PC_fit_in,c360_fit_in,c460_fit_in,c703_fit_in,c707_fit_in,c710_fit_in,c715_fit_in):
	chi_sq = ((BJ_obs_in - BJ_fit_in)/BJ_stddev)**2 + ((BC_obs_in - BC_fit_in)/BC_stddev)**2 + ((PC_obs_in - PC_fit_in)/PC_stddev)**2 + \
			((c360_obs_in - c360_fit_in)/c360_stddev)**2 + ((c460_obs_in - c460_fit_in)/c460_stddev)**2 + ((c703_obs_in - c703_fit_in)/c703_stddev)**2 + \
			((c707_obs_in - c707_fit_in)/c707_stddev)**2 + ((c710_obs_in - c710_fit_in)/c710_stddev)**2 + ((c715_obs_in - c715_fit_in)/c715_stddev)**2
	return chi_sq
# ------------------------------------------------------------

# FUNCTION to compute the Chi2 of the fit using also cont @ 460 nm
def chi_squared_additions(BJ_obs_in,BC_obs_in,PC_obs_in,c360_obs_in,c460_obs_in,c703_obs_in,c707_obs_in,c710_obs_in,c715_obs_in,\
	BJ_stddev,BC_stddev,PC_stddev,c360_stddev,c460_stddev,c703_stddev,c707_stddev,c710_stddev,c715_stddev,\
	BJ_fit_in,BC_fit_in,PC_fit_in,c360_fit_in,c460_fit_in,c703_fit_in,c707_fit_in,c710_fit_in,c715_fit_in):
	chi_sq = ((BJ_obs_in - BJ_fit_in)/BJ_stddev)**2 + ((BC_obs_in - BC_fit_in)/BC_stddev)**2 + ((PC_obs_in - PC_fit_in)/PC_stddev)**2 + \
			((c360_obs_in - c360_fit_in)/c360_stddev)**2 + ((c460_obs_in - c460_fit_in)/c460_stddev)**2 + ((c703_obs_in - c703_fit_in)/c703_stddev)**2 + \
			((c707_obs_in - c707_fit_in)/c707_stddev)**2 + ((c710_obs_in - c710_fit_in)/c710_stddev)**2 + ((c715_obs_in - c715_fit_in)/c715_stddev)**2
	return chi_sq
# ------------------------------------------------------------


# FUNCTION to compute the Chi2 with a variable number of input parameters
def chi2_general(*argv):
	# the argument must be passed in sub-arrays of 3 elements: [VALUE_OBJ, VALUE_MODEL, STD_OBJ]
	# Example: chi2 = chi2_general([12.3,12.2,0.2], [14.2,14.5,0.3], [198.3,184.2,1.7])
    chi2=0.
    for arg in argv:
        chi2 = chi2+( (arg[0]-arg[1]) / arg[2] )**2
    return chi2

# ------------------------------------------------------------

# FUNCTION to plot the distribution of delta chi2 with respect to SpT
def chi_sq_distr_cl3(chi_sq,H_fin,K_fin,cl3_in_list,PATH):
	""" PLOT THE DISTRIBUTION OF DELTA CHI2 WITH RESPECT TO SpT """
	# get the SpT and Teff of the cl3 targets
	#name_cl3,SpT_cl3,Teff_cl3 = readcol_py3(PATH_CLASSIII+'summary_classIII.txt',3,format='A,X,A,I',skipline=1)

	# estimate the min value of the chi2
	min_chi2 = np.min(list(chi_sq.values()))

	# create array with len(chi_sq) rows and 8 columns (cl3,Av,T,Ne,tau,chi**2,H,K)
	data = np.empty([len(chi_sq),8],dtype=('S20'))
	relDir = PATH + '/FrappeHelper/SpT_Teff_relation_hh14_short_codes.dat'
	relation = np.genfromtxt(relDir,usecols=(1,2),skip_header=1,dtype=[('sptCode',float),('Teff',float)])
	# read the chi square dictionary and put it in the array
	i = 0
	for k in chi_sq:
		data[i,0] = k.split('/')[0] # Cl3
		data[i,1] = k.split('/')[1]	# Av
		data[i,2] = k.split('/')[2]	# T
		data[i,3] = k.split('/')[3]	# Ne
		data[i,4] = k.split('/')[4]	# tau
		data[i,5] = chi_sq[k]	# chi_sq
		#print(H_fin[k][0])
		data[i,6] = H_fin[k][0]	# H   --- if it creates problem I should add a [0] at the end ??? 2015-02-02 CFM
		data[i,7] = K_fin[k][0]	# K
		i+=1

	# prepare the plot
	pl.figure(figsize=(6,5))

	# plot the DeltaChi2 values
	for el in cl3_in_list:
		if str(el).encode("utf-8") in data[:,0]:
			chi_x = np.array(data[np.where(data[:,0] == str(el).encode("utf-8")),5], dtype=float)
			chi_x = np.reshape(chi_x,(len(chi_x[0,:])))
			min_chi_x = np.min(chi_x)
			Teff = np.array([np.interp(el,relation['sptCode'],relation['Teff'])])#[0]
			pl.plot(Teff,[min_chi_x-min_chi2],'ro',ms=5)
			#print('here')
			#print(Teff,[min_chi_x-min_chi2])
	# pl.yscale('log')
	pl.axis([5900,2320,-1,10])
	pl.plot([5900,2320],[0,0],'--', lw=3)
	pl.plot([5900,2320],[1,1],'r--', lw=3)
	pl.plot([5900,2320],[4,4],'g--', lw=3)
	pl.xlabel(r'T$_{\rm eff}$',fontsize=20)
	pl.ylabel(r'$\Delta\chi ^2$',fontsize=20)

	pass
# ------------------------------------------------------------


# FUNCTION to plot the distribution of delta chi2 with respect to Av
def chi_sq_distr_Av(chi_sq,H_fin,K_fin,Av_list,best_av):
	""" PLOT THE DISTRIBUTION OF DELTA CHI2 WITH RESPECT TO Av """
	# estimate the min value of the chi2
	min_chi2 = np.min(list(chi_sq.values()))

	# create array with len(chi_sq) rows and 8 columns (cl3,Av,T,Ne,tau,chi**2,H,K)
	data = np.empty([len(chi_sq),8],dtype=('S20'))
	#print(Av_list)
	# read the chi square dictionary and put it in the array
	i = 0
	for k in chi_sq:
		data[i,0] = k.split('/')[0] # Cl3
		data[i,1] = k.split('/')[1]	# Av
		#print(k.split('/')[1])
		data[i,2] = k.split('/')[2]	# T
		data[i,3] = k.split('/')[3]	# Ne
		data[i,4] = k.split('/')[4]	# tau
		data[i,5] = chi_sq[k]	# chi_sq
		data[i,6] = H_fin[k][0]	# H
		data[i,7] = K_fin[k][0]	# K
		i+=1

	# prepare the plot
	pl.figure(figsize=(6,5))

	# plot the DeltaChi2 values
	for el in Av_list:
		if str(el).encode("utf-8") in data[:,1]:
			chi_x = np.array(data[np.where(data[:,1] == str(el).encode("utf-8")),5], dtype=float)
			chi_x = np.reshape(chi_x,(len(chi_x[0,:])))
			min_chi_x = np.min(chi_x)
			pl.plot([el],[min_chi_x-min_chi2],'ro',ms=5)
	pl.axis([np.max([0,best_av-3.]),best_av+3.,-1,10])
	# pl.axis([5,8,-1,10])
	pl.plot([0,8],[0,0],'--', lw=3)
	pl.plot([0,8],[1,1],'r--', lw=3)
	pl.plot([0,8],[4,4],'g--', lw=3)
	pl.xlabel(r'A$_V$',fontsize=20)
	pl.ylabel(r'$\Delta\chi ^2$',fontsize=20)

	pass
# ------------------------------------------------------------


# FUNCTION to plot the distribution of delta chi2 with respect to SpT
def chi_sq_distr_cl3_Av(chi_sq,H_fin,K_fin,cl3_in_list,Av_list,PATH):
	""" PLOT THE DISTRIBUTION OF DELTA CHI2 WITH RESPECT TO SpT and Av """
	# get the SpT and Teff of the cl3 targets

	# estimate the min value of the chi2
	min_chi2 = np.min(list(chi_sq.values()))

	# create array with len(chi_sq) rows and 8 columns (cl3,Av,T,Ne,tau,chi**2,H,K)
	data = np.empty([len(chi_sq),8],dtype=('S20'))

	# read the chi square dictionary and put it in the array
	i = 0
	relDir = PATH + '/FrappeHelper/SpT_Teff_relation_hh14_short_codes.dat'
	relation = np.genfromtxt(relDir,usecols=(1,2),skip_header=1,dtype=[('sptCode',float),('Teff',float)])
	for k in chi_sq:
		data[i,0] = k.split('/')[0] # Cl3
		data[i,1] = k.split('/')[1]	# Av
		data[i,2] = k.split('/')[2]	# T
		data[i,3] = k.split('/')[3]	# Ne
		data[i,4] = k.split('/')[4]	# tau
		data[i,5] = chi_sq[k]	# chi_sq
		data[i,6] = H_fin[k][0]	# H
		data[i,7] = K_fin[k][0]	# K
		i+=1

	# assign to each Cl3 a Teff value
	Teff_all = np.ones(len(data[:,0]))
	for ite in range(len(data[:,0])):
		Teff = np.array([np.interp(data[ite,0],relation['sptCode'],relation['Teff'])])[0]
		Teff_all[ite] = Teff
	# get the Av in float
	Av_all = np.array(data[:,1],dtype=np.float32,copy=True)
	# get the chi2 in float
	Chi2_all = np.array(data[:,5],dtype=np.float32,copy=True)
	# get the DeltaChi2 values
	#print('Chi2_all')
	#print(Chi2_all)
	#print('min_chi2')
	#print(min_chi2)
	DChi2_all = Chi2_all - min_chi2

	# for each Av and Teff get the minimum DeltaChi2
	Teff_in = np.empty(len(cl3_in_list),dtype=float)
	DChi2 = np.zeros([len(cl3_in_list),len(Av_list)], dtype=float)
	# arrays for displaying where the Av in Av_list and Teff in  are not in the fitter output
	NotInAv_AllButExplored =np.array([])
	NotInTeff_AllButExplored =np.array([])
	for it in range(len(cl3_in_list)):
		Teff = np.array([np.interp(cl3_in_list[it],relation['sptCode'],relation['Teff'])])[0]
		Teff_in[it] = Teff
		#Teff_in[it] = Teff_cl3[np.where(name_cl3 == cl3_in_list[it])]
		for ine in range(len(Av_list)):
			#print(*Av_all,Av_list[ine])
			#print(*Teff_all,Teff_in[it])
			ValuesAtAvAndTeff = Chi2_all[(Av_all == Av_list[ine])&(Teff_all==Teff_in[it])]
			if len(ValuesAtAvAndTeff) == 0:
				DChi2[it,ine] = np.nan-min_chi2
				NotInAv_AllButExplored =  np.append(NotInAv_AllButExplored,Av_list[ine])
				NotInTeff_AllButExplored =  np.append(NotInTeff_AllButExplored,Teff_in[it])
			else:
				DChi2[it,ine] = np.min(ValuesAtAvAndTeff)-min_chi2
			# if data[np.nonzero((data[:,0] == cl3_in_list[it]) & (np.array(data[:,1],dtype=float) == Av_list[ine])),5].size == 0:
			# 	DChi2[it,ine] = None
			# else:
			# 	DChi2[it,ine] = np.min(np.array(data[((data[:,0] == str(cl3_in_list[it])) & \
			# 		(np.array(data[:,1],dtype=float) == Av_list[ine])),5],dtype=float)) - min_chi2


	# prepare the plot
	fig = pl.figure()

	pl.plot(NotInAv_AllButExplored,NotInTeff_AllButExplored,'kx',ms=3,mfc='None',mec='k',alpha=0.6)
	pl.plot(Av_all,Teff_all,'ko',ms=3,mfc='None',mec='k',alpha=0.1)
	pl.plot(Av_all[Chi2_all==min_chi2],Teff_all[Chi2_all==min_chi2],'ro',ms=5,mfc='None',mec='r')
	pl.contour(Av_list,Teff_in, DChi2, levels=[0.,0.1,0.5,1.,1.5,2.30,4.61,6.17], linewidths=0.5, colors='k')
	cntr2 = pl.contourf(Av_list,Teff_in, DChi2, levels=[0.,0.1,0.5,1.,1.5,2.30,4.61,6.17], cmap='plasma')
	cbar = fig.colorbar(cntr2)
	cbar.ax.set_ylabel(r'$\Delta\chi ^2$')
	pl.axis([np.min(Av_list)-0.3,np.max(Av_list)+0.3,np.min(Teff_in)-50,np.max(Teff_in)+50])
	pl.ylabel(r'T$_{\rm eff}$ [K]', fontsize=24)
	pl.xlabel(r'A$_V$ [mag]', fontsize=24)
	pl.gca().minorticks_on()
	pl.tight_layout()
	pl.show()

def posterior_distr_cl3_Av(chi_sq,H_fin,K_fin,cl3_in_list,Av_list,PATH):
	""" PLOT THE DISTRIBUTION OF DELTA CHI2 WITH RESPECT TO SpT and Av """
	# get the SpT and Teff of the cl3 targets
	#name_cl3,SpT_cl3,Teff_cl3 = readcol_py3(PATH_CLASSIII+'summary_classIII.txt',3,format='A,X,A,I',skipline=1)

	# estimate the min value of the chi2
	#min_chi2 = np.min(list(chi_sq.values()))
	#max_chi2 = np.min(list(chi_sq.values()))
	# create array with len(chi_sq) rows and 8 columns (cl3,Av,T,Ne,tau,chi**2,H,K)
	data = np.empty([len(chi_sq),8],dtype=('S20'))

	# read the chi square dictionary and put it in the array
	i = 0
	relDir = PATH + '/FrappeHelper/SpT_Teff_relation_hh14_short_codes.dat'
	relation = np.genfromtxt(relDir,usecols=(1,2),skip_header=1,dtype=[('sptCode',float),('Teff',float)])
	for k in chi_sq:
		data[i,0] = k.split('/')[0] # Cl3
		data[i,1] = k.split('/')[1]	# Av
		data[i,2] = k.split('/')[2]	# T
		data[i,3] = k.split('/')[3]	# Ne
		data[i,4] = k.split('/')[4]	# tau
		data[i,5] = chi_sq[k]	# chi_sq
		data[i,6] = H_fin[k][0]	# H
		data[i,7] = K_fin[k][0]	# K
		i+=1

	# assign to each Cl3 a Teff value
	Teff_all = np.ones(len(data[:,0]))
	for ite in range(len(data[:,0])):
		Teff = np.array([np.interp(data[ite,0],relation['sptCode'],relation['Teff'])])[0]
		Teff_all[ite] = Teff
	# get the Av in float
	Av_all = np.array(data[:,1],dtype=np.float32,copy=True)
	# get the chi2 in float
	Chi2_all = np.array(data[:,5],dtype=np.float32,copy=True)
	# get the DeltaChi2 values
	#print('Chi2_all')
	#print(Chi2_all)
	#print('min_chi2')
	#print(min_chi2)
	DChi2_all = Chi2_all #- min_chi2

	# for each Av and Teff get the minimum DeltaChi2
	Teff_in = np.empty(len(cl3_in_list),dtype=float)
	DChi2 = np.zeros([len(cl3_in_list),len(Av_list)], dtype=float)
	# arrays for displaying where the Av in Av_list and Teff in  are not in the fitter output
	NotInAv_AllButExplored =np.array([])
	NotInTeff_AllButExplored =np.array([])
	for it in range(len(cl3_in_list)):
		Teff = np.array([np.interp(cl3_in_list[it],relation['sptCode'],relation['Teff'])])[0]
		Teff_in[it] = Teff
		#Teff_in[it] = Teff_cl3[np.where(name_cl3 == cl3_in_list[it])]
		for ine in range(len(Av_list)):
			#print(*Av_all,Av_list[ine])
			#print(*Teff_all,Teff_in[it])
			ValuesAtAvAndTeff = Chi2_all[(Av_all == Av_list[ine])&(Teff_all==Teff_in[it])]
			if len(ValuesAtAvAndTeff) == 0:
				DChi2[it,ine] = np.nan#-min_chi2
				NotInAv_AllButExplored =  np.append(NotInAv_AllButExplored,Av_list[ine])
				NotInTeff_AllButExplored =  np.append(NotInTeff_AllButExplored,Teff_in[it])
			else:
				DChi2[it,ine] = np.nansum(np.exp(-ValuesAtAvAndTeff))#-min_chi2
			# if data[np.nonzero((data[:,0] == cl3_in_list[it]) & (np.array(data[:,1],dtype=float) == Av_list[ine])),5].size == 0:
			# 	DChi2[it,ine] = None
			# else:
			# 	DChi2[it,ine] = np.min(np.array(data[((data[:,0] == str(cl3_in_list[it])) & \
			# 		(np.array(data[:,1],dtype=float) == Av_list[ine])),5],dtype=float)) - min_chi2


	# prepare the plot
	DChi2_norm = DChi2/np.nansum(DChi2)
	fig = pl.figure()

	pl.plot(NotInAv_AllButExplored,NotInTeff_AllButExplored,'kx',ms=3,mfc='None',mec='k',alpha=0.6)
	pl.plot(Av_all,Teff_all,'ko',ms=3,mfc='None',mec='k',alpha=0.1)
	#pl.plot(Av_all[Chi2_all==min_chi2],Teff_all[Chi2_all==min_chi2],'ro',ms=5,mfc='None',mec='r')
	pl.contour(Av_list,Teff_in, DChi2_norm, linewidths=0.5, colors='k')
	cntr2 = pl.contourf(Av_list,Teff_in, DChi2_norm, cmap='plasma')
	cbar = fig.colorbar(cntr2)
	cbar.ax.set_ylabel(r'$\Delta\chi ^2$')
	pl.axis([np.min(Av_list)-0.3,np.max(Av_list)+0.3,np.min(Teff_in)-50,np.max(Teff_in)+50])
	pl.ylabel(r'T$_{\rm eff}$ [K]', fontsize=24)
	pl.xlabel(r'A$_V$ [mag]', fontsize=24)
	pl.gca().minorticks_on()
	pl.tight_layout()
	pl.show()



# FUNCTION to plot the distribution of delta chi2 with respect to SpT
def chi_sq_distr_cl3_Av_Marginalise(chi_sq,H_fin,K_fin,cl3_in_list,Av_list,PATH):
	""" PLOT THE DISTRIBUTION OF DELTA CHI2 WITH RESPECT TO SpT and Av """
	# get the SpT and Teff of the cl3 targets
	#name_cl3,SpT_cl3,Teff_cl3 = readcol_py3(PATH_CLASSIII+'summary_classIII.txt',3,format='A,X,A,I',skipline=1)

	# estimate the min value of the chi2
	min_chi2 = np.min(list(chi_sq.values()))

	# create array with len(chi_sq) rows and 8 columns (cl3,Av,T,Ne,tau,chi**2,H,K)
	data = np.empty([len(chi_sq),8],dtype=('S20'))

	# read the chi square dictionary and put it in the array
	i = 0
	relDir = PATH + '/FrappeHelper/SpT_Teff_relation_hh14_short_codes.dat'
	relation = np.genfromtxt(relDir,usecols=(1,2),skip_header=1,dtype=[('sptCode',float),('Teff',float)])
	for k in chi_sq:
		data[i,0] = k.split('/')[0] # Cl3
		data[i,1] = k.split('/')[1]	# Av
		data[i,2] = k.split('/')[2]	# T
		data[i,3] = k.split('/')[3]	# Ne
		data[i,4] = k.split('/')[4]	# tau
		data[i,5] = chi_sq[k]	# chi_sq
		data[i,6] = H_fin[k][0]	# H
		data[i,7] = K_fin[k][0]	# K
		i+=1

	# assign to each Cl3 a Teff value
	Teff_all = np.ones(len(data[:,0]))
	for ite in range(len(data[:,0])):
		Teff = np.array([np.interp(data[ite,0],relation['sptCode'],relation['Teff'])])[0]
		Teff_all[ite] = Teff
	# get the Av in float
	Av_all = np.array(data[:,1],dtype=np.float32,copy=True)
	# get the chi2 in float
	Chi2_all = np.array(data[:,5],dtype=np.float32,copy=True)
	# get the DeltaChi2 values
	#print('Chi2_all')
	#print(Chi2_all)
	#print('min_chi2')
	#print(min_chi2)
	DChi2_all = Chi2_all - min_chi2

	# for each Av and Teff get the minimum DeltaChi2
	Teff_in = np.empty(len(cl3_in_list),dtype=float)
	DChi2 = np.zeros([len(cl3_in_list),len(Av_list)], dtype=float)
	# arrays for displaying where the Av in Av_list and Teff in  are not in the fitter output
	NotInAv_AllButExplored =np.array([])
	NotInTeff_AllButExplored =np.array([])
	for it in range(len(cl3_in_list)):
		Teff = np.array([np.interp(cl3_in_list[it],relation['sptCode'],relation['Teff'])])[0]
		Teff_in[it] = Teff
		#Teff_in[it] = Teff_cl3[np.where(name_cl3 == cl3_in_list[it])]
		for ine in range(len(Av_list)):
			#print(*Av_all,Av_list[ine])
			#print(*Teff_all,Teff_in[it])
			ValuesAtAvAndTeff = Chi2_all[(Av_all == Av_list[ine])&(Teff_all==Teff_in[it])]
			if len(ValuesAtAvAndTeff) == 0:
				DChi2[it,ine] = np.nan-min_chi2
				NotInAv_AllButExplored =  np.append(NotInAv_AllButExplored,Av_list[ine])
				NotInTeff_AllButExplored =  np.append(NotInTeff_AllButExplored,Teff_in[it])
			else:
				DChi2[it,ine] = np.sum(ValuesAtAvAndTeff)
			# if data[np.nonzero((data[:,0] == cl3_in_list[it]) & (np.array(data[:,1],dtype=float) == Av_list[ine])),5].size == 0:
			# 	DChi2[it,ine] = None
			# else:
			# 	DChi2[it,ine] = np.min(np.array(data[((data[:,0] == str(cl3_in_list[it])) & \
			# 		(np.array(data[:,1],dtype=float) == Av_list[ine])),5],dtype=float)) - min_chi2


	# prepare the plot
	fig = pl.figure()

	pl.plot(NotInAv_AllButExplored,NotInTeff_AllButExplored,'kx',ms=3,mfc='None',mec='k',alpha=0.6)
	pl.plot(Av_all,Teff_all,'ko',ms=3,mfc='None',mec='k',alpha=0.1)
	pl.plot(Av_all[Chi2_all==min_chi2],Teff_all[Chi2_all==min_chi2],'ro',ms=5,mfc='None',mec='r')
	prob = np.exp(DChi2)/sum(np.exp(DChi2))
	maxProb = np.max(prob)
	ValAt1std = 0.2419
	pl.contour(Av_list,Teff_in, prob, levels=[0.5 *ValAt1std*maxProb, ValAt1std*maxProb,2*ValAt1std*maxProb ,3*ValAt1std*maxProb], linewidths=0.5, colors='k')
	cntr2 = pl.contourf(Av_list,Teff_in, prob, levels=[0.5 *ValAt1std*maxProb, ValAt1std*maxProb,2*ValAt1std*maxProb ,3*ValAt1std*maxProb], cmap='plasma')
	cbar = fig.colorbar(cntr2)
	cbar.ax.set_ylabel(r'$\Delta\chi ^2$')
	pl.axis([np.min(Av_list)-0.3,np.max(Av_list)+0.3,np.min(Teff_in)-50,np.max(Teff_in)+50])
	pl.ylabel(r'T$_{\rm eff}$ [K]', fontsize=24)
	pl.xlabel(r'A$_V$ [mag]', fontsize=24)
	pl.gca().minorticks_on()
	pl.tight_layout()
	pl.show()







	# PLOT

	# fig = pl.figure()
	# pl.plot(Av_all,Teff_all,'ko',ms=3,mfc='None',mec='k',alpha=0.1)
	# pl.plot(Av_all[Chi2_all==min_chi2],Teff_all[Chi2_all==min_chi2],'ro',ms=5,mfc='None',mec='r')
	# pl.contourf(Av_all,Teff_all, DChi2, levels=[-0.1,0.,0.1,0.5,1.,1.5,2.30,4.61,6.17], linewidths=0.5, colors='k')
	# cntr2 = pl.tricontourf(Av_all,Teff_all, DChi2, levels=[-0.1,0.,0.1,0.5,1.,1.5,2.30,4.61,6.17], cmap="plasma")

	# fig.colorbar(cntr2)
	# pl.axis([np.min(Av_all)-0.3,np.max(Av_all)+0.3,np.min(Teff_all)-50,np.max(Teff_all)+50])
	# pl.ylabel(r'T$_{\rm eff}$ [K]', fontsize=24)
	# pl.xlabel(r'A$_V$ [mag]', fontsize=24)
	# pl.gca().minorticks_on()
	# pl.tight_layout()

	pass
# ------------------------------------------------------------










"""
# FUNCTION to compute the Chi2 of the fit using also cont @ 460 nm
def chi_squared_bis(BJ_obs_in,BC_obs_in,PC_obs_in,c360_obs_in,c460_obs_in,c703_obs_in,c707_obs_in,c710_obs_in,c715_obs_in,\
	BJ_stddev,BC_stddev,PC_stddev,c360_stddev,c460_stddev,c703_stddev,c707_stddev,c710_stddev,c715_stddev,\
	BJ_fit_in,BC_fit_in,PC_fit_in,c360_fit_in,c460_fit_in,c703_fit_in,c707_fit_in,c710_fit_in,c715_fit_in):
	chi_sq = ((BJ_obs_in - BJ_fit_in)/BJ_stddev)**2 + ((BC_obs_in - BC_fit_in)/BC_stddev)**2 + ((PC_obs_in - PC_fit_in)/PC_stddev)**2 + \
			((c360_obs_in - c360_fit_in)/c360_stddev)**2 + ((c460_obs_in - c460_fit_in)/c460_stddev)**2 + 0.1*(((c703_obs_in - c703_fit_in)/c703_stddev)**2 + \
			((c707_obs_in - c707_fit_in)/c707_stddev)**2 + ((c710_obs_in - c710_fit_in)/c710_stddev)**2 + ((c715_obs_in - c715_fit_in)/c715_stddev)**2)
	return chi_sq
# ------------------------------------------------------------
"""
# FUNCTION to plot two spectra altogether
def plotter(x1,y1,tit,xtit,ytit,xmin,xmax,ymin,ymax,x2=np.zeros(1),y2=np.zeros(1),x3=np.zeros(1),y3=np.zeros(1),out='SHOW'):
	# rc('font',family='Futura Lt AT')
	rc('text',usetex=True)
	#plot the spectrum
	p1 = pl.figure(1)
	pl.plot(x1,y1,'k')#,title=hdr['OBJECT'])#,xtitle='Wavelength [nm]',ytitle='Flux')
	pl.title(tit)
	pl.xlabel(xtit)
	pl.ylabel(ytit)
	pl.axis([xmin,xmax,ymin,ymax])
	if x2.any() != 0 and y2.any() != 0:
		pl.plot(x2,y2,'r')
	if x3.any() != 0 and y3.any() != 0:
		pl.plot(x3,y3,'w')
	if out == 'SHOW':
		pl.show()
	elif (out.split('.')[-1] == 'eps') or (out.split('.')[-1] == 'png') or (out.split('.')[-1] == 'jpg') or (out.split('.')[-1] == 'ps'):
		pl.savefig(out)
	elif out == 'next':
		pl.show()
		pl.cla()
	else:
		return
	return p1
# ------------------------------------------------------------

"""
# FUNCTION to read slab models produced with the C++ program
def read_slab(T,Ne,tau,PATH_IN):
	nelements = 3972 #the last line is empty!
	wl = np.zeros(nelements)
	fl = np.zeros(nelements)
	nskip = 0
	nline = 0
	for line in open(PATH_IN+'continuum_tot_T'+T+'_ne'+Ne+'tau'+tau+'.out'):
		if nskip <= 2:
			line.split()
			nskip+=1
		else:
			wl[nline] = line.split()[0]
			fl[nline] = line.split()[1]
			nline+=1
	return wl,fl
# ------------------------------------------------------------
"""
