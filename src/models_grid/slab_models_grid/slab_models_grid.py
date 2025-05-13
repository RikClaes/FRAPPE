#!/usr/bin/env python

			###############################################
			# 				slab_models_grid.py 	      #
			# prepare a grid of slab models using C++ pro #
			###############################################


# Created by:
#		CFM, on Jan 2nd-3rd 2013 @Arsago+Munich
# Corrected by CFM on Feb, 21st 2013 in Santiago to get the right output file (*_tot_*)
#
# ---------------------------------------------------------------------------------------

import os
import time

start_time = time.time()

PATH = '/Users/cmanara/Accretion/ultimate_fitter/models_grid/slab_models_grid/'
PATH_ACC = '/Users/cmanara/Accretion/slab_model_cfm/'

#1) decide the set of parameters (T,N_e,tau) of the slab models to be used in the grid --> in.slab
T_slab = ['5000','5500','6000','6500','7000','7500','7750','8000','8250','8500','8750','9000','9250','9500','9750',\
'10000','10500','11000']
Ne_slab = ['1e+11','1e+12','1e+13','3e+13','5e+13','7e+13','1e+14','5e+14','1e+15','1e+16']
tau_slab = ['0.01', '0.05','0.1','0.3','0.5','0.75','1','3','5']


for T_iter in T_slab:
	for Ne_iter in Ne_slab:
		for tau_iter in tau_slab:
			f = open(PATH_ACC+'in.slab','w')
			f.write(T_iter+'   '+Ne_iter+'   '+tau_iter)
			f.close()
			#change directory to where the C++ program is
			os.chdir(PATH_ACC)
			print T_iter, Ne_iter, tau_iter
			if os.path.isfile('hydrogen_slab'):
			#2) run the c++ program for each parameter of the slab model
				os.system('./hydrogen_slab')
			#3) move the results to the folder 'slab_models_grid'
				if os.path.isfile('results/continuum_tot_T'+T_iter+'_ne'+Ne_iter+'tau'+tau_iter+'.out'):
					os.system('cp results/continuum_tot_T'+T_iter+'_ne'+Ne_iter+'tau'+tau_iter+'.out '+PATH)
				else:
					print 'ERROR! output file not found'
					break	
			else:
				print 'ERROR! C++ file not found!'
				break
					

os.chdir(PATH)				

print 'Execution time:', time.time() - start_time, " seconds, or"
print 'Execution time:', (time.time() - start_time)/60., " minutes, or"
print 'Execution time:', (time.time() - start_time)/3600., " hours"


# FOR T_slab = ['5000','5500','6000','6500','7000','7500','7750','8000','8250','8500','8750','9000','9250','9500','9750',\
# '10000','10500','11000']
# Ne_slab = ['1e+11','1e+12','1e+13','3e+13','5e+13','7e+13','1e+14','5e+14','1e+15','1e+16']
# tau_slab = ['0.01', '0.05','0.1','0.3','0.5','0.75','1','3','5']
# it took:
# Execution time: 291.211688042  seconds, or
# Execution time: 4.85352853537  minutes
# Jan, 3rd 2013 - CFM @ Munich


#4) save them all in a smart way, easy to be read ---> another program? Maybe better
#IDL:
# ;read the result of the C++ program
# readcol, path_acc+'results/continuum_tot_T'+temp[0]+'_ne'+n_e[0]+'tau'+tau[0]+'.out', wave_in, flux_acc, format='d,d', /silent
# flux_acc_res = spectrum_resample(flux_acc, wave_in, lambda[0:find_lambda(2470,lambda[*,0]),0])
# l = lambda[0:find_lambda(2470,lambda[*,0]),0]
# f = flux[0:find_lambda(2470,lambda[*,0]),0]
#



