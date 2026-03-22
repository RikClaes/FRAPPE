#!/usr/bin/env python

import numpy as np
from scipy.io.idl import readsav
import sys
import pylab as pl

def isochrone_plot(ages=[2,10,30,100],masses=[0.02,0.05,0.1,0.2,0.4,0.6,0.8,1.,1.2, 1.5, 2],model='Siess',plot_sun = 'NO',PATH =None):
	""" Plot isochrones using the requested evolutionary model for the requested ages and masses

	Example:
		isochrone_plot(ages=[2,10,30,100],masses=[0.02,0.05,0.1,0.2,0.4,0.6,0.8,1.,1.2],model='Siess')

	Available models: Siess, Baraffe (Baraffe+98), Dantona, Palla, B15 (Baraffe+15), Feiden
	"""
	# get evolutionary model
	PATH_MODELS_SAV = PATH + '/models_grid/evolutionary_tracks/'

	if model == 'Siess':
		filename = PATH_MODELS_SAV+'tracks_siess00c_dense_no-overshoot.sav'
	elif model == 'Baraffe':
		filename = PATH_MODELS_SAV+'tracks_bcah98_dense.sav'
	elif model == 'Dantona':
		filename = PATH_MODELS_SAV+'tracks_dantona98_dense.sav'
	elif model == 'Palla':
		filename = PATH_MODELS_SAV+'tracks_palla99_dense.sav'
	elif model == 'B15':
		filename = PATH_MODELS_SAV+'tracks_bhac15_dense_lowmass.sav'
	elif model == 'B15_OLD':
		filename = PATH_MODELS_SAV+'tracks_bhac15_dense_nolowmass.sav'
	elif model == 'Feiden':
		filename = PATH_MODELS_SAV+'tracks_feiden_dense.sav' # Feiden+2016
	else:
		sys.exit('Possible models are: Siess, Baraffe, Dantona, Palla, B15, Feiden. Please select one of them or use the default one (Siess)')

	s = readsav(filename)
	l,t,m,a = s['l'],s['t'],s['m'],s['a']
	s.clear()

	# definitions
	ages = np.array(ages)
	strages = np.array(ages, dtype='S10')
	logages=np.log10(ages*1.e6)
	tsun = 5780.
	gsun=27400.
	mym = np.array(masses)

	# plot masses
	for i in range(len(mym)):
		sel = np.where(m == mym[i])
		if len(t[sel]) > 0:
			pl.plot(t[sel], l[sel], 'k--',alpha=0.5)
			if mym[i]<0.2:
				label = r'%.2f M$_\odot$' % mym[i]
			else:
				label = r'%.1f M$_\odot$' % mym[i]
			# ax = pl.gca()
			pl.text(t[sel][0]-0.015,(l[sel][0]+0.05),label,fontsize=11,ha='center',clip_on=True)#,\
			# pl.text(t[sel][len(t[sel])/5],(l[sel][len(t[sel])/5]),label,fontsize=10,ha='center',clip_on=True)#,\
				# bbox=(dict(facecolor='white',edgecolor='white',alpha=0.5,clip_on=True)))
		else:
			pass
		# pl.annotate(label,xy=(t[sel[0]],l[sel[0]]), xytext=(t[sel[0]],l[sel[0]]+0.1))
		# t1.rotate,180,yaxis=1 & endfor

	# plot ages
	for k in range(len(ages)):
		myage = logages[k]
		mvalues=np.concatenate( (mym,np.unique(m)[::15]) )	# put together all the input values of M and some of the model's one
		mvalues = np.unique(mvalues)	# now use only the unique values and sort them
		myt = np.zeros(len(mvalues))
		myl = np.zeros(len(mvalues))

		for i in range(len(mvalues)):
			thistrack = np.where(m == mvalues[i])
			if len(m[thistrack]) > 0:
				if myage >= np.min(a[thistrack]) and myage <= np.max(a[thistrack]):
					if np.all(np.diff(a[thistrack]) > 0):
						a_sort = a[thistrack]
						t_sort = t[thistrack]
						l_sort = l[thistrack]
					else:
						ind = np.argsort(a[thistrack])
						a_sort = a[thistrack][ind]
						t_sort = t[thistrack][ind]
						l_sort = l[thistrack][ind]
					myt[i]=np.interp(myage,a_sort,t_sort)
					myl[i]=np.interp(myage,a_sort,l_sort)
				else:
					myt[i]=np.nan
					myl[i]=np.nan
			else:
				myt[i]=np.nan
				myl[i]=np.nan
# myr=sqrt(10^myl/(10^myt/tsun)^4)
# myg=alog10(mym*((10^myt/tsun)^4.)/10^myl*gsun)
		pl.plot(myt[0:], myl[0:], 'k',alpha=0.5)
		# print(myt[0:],myl[0:])

	pass


def isochrone_interp(logT,logL,model='Siess',tol=0.016,PATH =None):
	""" Derive mass and ages using the input values for logL and logT and the requested evolutionary model

	Example:
		mass_bara,logage_bara = isochrone_interp(logT,logL,model='Siess',tol=0.016)

	Available models: Siess, Baraffe (Baraffe+98), Dantona, Palla, B15 (Baraffe+15), Feiden
	"""
	# get evolutionary model
	PATH_MODELS_SAV = PATH + '/models_grid/evolutionary_tracks/'
	if model == 'Siess':
		filename = PATH_MODELS_SAV+'tracks_siess00c_dense_no-overshoot.sav'
	elif model == 'Baraffe':
		filename = PATH_MODELS_SAV+'tracks_bcah98_dense.sav'
	elif model == 'Dantona':
		filename = PATH_MODELS_SAV+'tracks_dantona98_dense.sav'
	elif model == 'Palla':
		filename = PATH_MODELS_SAV+'tracks_palla99_dense.sav'
	elif model == 'B15':
		filename = PATH_MODELS_SAV+'tracks_bhac15_dense_lowmass.sav'
	elif model == 'B15_OLD':
		filename = PATH_MODELS_SAV+'tracks_bhac15_dense_nolowmass.sav'
	elif model == 'Feiden':
		filename = PATH_MODELS_SAV+'tracks_feiden_dense.sav' # Feiden+2016
	else:
		sys.exit('Possible models are: Siess, Baraffe, Dantona, Palla, B15, Feiden. Please select one of them or use the default one (Siess)')

	s = readsav(filename)
	l,t,m,a = s['l'],s['t'],s['m'],s['a']
	s.clear()

	# definitions
	Nsource = np.size(logL) # Jan,13th 2015 - changed len to np.size to be able to treat properly also floats
	mass = np.zeros(Nsource)
	logage = np.zeros(Nsource)

	for i in range(Nsource):
		distance = np.sqrt((logT[i]-t)**2+(logL[i]-l)**2)
		best = np.where(distance == np.min(distance))
		if np.size(best[0]) > 1:	# this happens if there are two solutions which are the same
			best = best[0][0]	# in this case I select one of the two (they are the same)
		if distance[best] <= tol:
			mass[i] = np.squeeze(m[best])
			logage[i] = np.squeeze(a[best])
		else:
			#print(distance[best])
			mass[i] = np.nan
			logage[i] = np.nan

	return mass,logage


def get_isochrone(mstar,age,model='Siess',PATH =None):
	"""
	# output = loglstar,logteff
	# returns the values of Lstar and Teff for a given isochrone at the SINGLE input age [Myr]
	# and for the set of (MULTIPLE) Mstar [Msun] given as input for the requested model
	"""
	# get evolutionary model
	PATH_MODELS_SAV = PATH + '/models_grid/evolutionary_tracks/'

	if model == 'Siess':
		filename = PATH_MODELS_SAV+'tracks_siess00c_dense_no-overshoot.sav'
	elif model == 'Baraffe':
		filename = PATH_MODELS_SAV+'tracks_bcah98_dense.sav'
	elif model == 'Dantona':
		filename = PATH_MODELS_SAV+'tracks_dantona98_dense.sav'
	elif model == 'Palla':
		filename = PATH_MODELS_SAV+'tracks_palla99_dense.sav'
	elif model == 'B15':
		filename = PATH_MODELS_SAV+'tracks_bhac15_dense_lowmass.sav'
	elif model == 'B15_OLD':
		filename = PATH_MODELS_SAV+'tracks_bhac15_dense_nolowmass.sav'
	elif model == 'Feiden':
		filename = PATH_MODELS_SAV+'tracks_feiden_dense.sav' # Feiden+2016
	else:
		sys.exit('Possible models are: Siess, Baraffe, Dantona, Palla, B15, Feiden. Please select one of them or use the default one (Siess)')

	s = readsav(filename)
	l,t,m,a = s['l'],s['t'],s['m'],s['a']
	s.clear()

	# select the correct isochrone (the closer to the correct one)
	# isoc_ind = np.where(np.abs(a-np.log10(age*1e6))==np.min(np.abs(a-np.log10(age*1e6))))
	isoc_ind = np.where(np.abs(a-np.log10(age*1e6))<=1e-2)

	# now interpolate the isochrone at the given masses
	# first order for increasing mass
	m_ord_ind = np.argsort(m[isoc_ind])
	# get the Teff
	tout = np.interp(mstar,m[isoc_ind][m_ord_ind],t[isoc_ind][m_ord_ind])
	# get the Lstar
	lout = np.interp(mstar,m[isoc_ind][m_ord_ind],l[isoc_ind][m_ord_ind])

	return lout,tout



# # TO BE DONE
# def isochrone_get(logT_in='none',logL_in='none',age_out='none',mass_out='none',model='Siess',verbose=False):
# 	"""
# 	Gives the parameters needed (logT,logL) to have an object with a given age or mass according to the requested evolutionary model

# 	Inputs:
# 	logT_in = input temperature in logarithm [unit = K]
# 	logL_in = input stellar luminosity in logarithm [unit = Lsun]
# 	age_out = requested age of the object [unit = Myr]
# 	mass_out = requested mass of the object [unit = Msun]

# 	- If none of logT and logL is given, than the function wants a given age and mass to give both logT and logL needed
# 	- TO BE IMPLEMENTED: If logT or logL is given in input, the function gives back the other parameter to get the requested age or mass
# 	- if you have both logT and logL, then the function to be used is isochrone_interp, because you know where you are on the HRD

# 	Example:
# 		logt,logl = isochrone_get(logT_in='none',logL='none',age_out=1.2,mass_out=0.5,model='Siess')

# 	Available models: Siess, Baraffe, Dantona, Palla
# 	"""
# 	# get evolutionary model
# 	PATH_MODELS_SAV = '/Users/cmanara/work/utilities/evolutionary_tracks/'

# 	if model == 'Siess':
# 		filename = PATH_MODELS_SAV+'tracks_siess00c_dense_no-overshoot.sav'
# 	elif model == 'Baraffe':
# 		filename = PATH_MODELS_SAV+'tracks_bcah98_dense.sav'
# 	elif model == 'Dantona':
# 		filename = PATH_MODELS_SAV+'tracks_dantona98_dense.sav'
# 	elif model == 'Palla':
# 		filename = PATH_MODELS_SAV+'tracks_palla99_dense.sav'
# 	elif model == 'B15':
# 		filename = PATH_MODELS_SAV+'tracks_bhac15_dense.sav'
# 	else:
# 		sys.exit('Possible models are: Siess, Baraffe, Dantona, Palla, B15. Please select one of them or use the default one (Siess)')

# 	s = readsav(filename)
# 	l,t,m,a = s['l'],s['t'],s['m'],s['a']
# 	s.clear()

# 	if logT_in != 'none' and logL_in == 'none':
# 		sys.exit('NOT IMPLEMENTED YET')
# 		# I have T input, I have to get L to obtain the requested ...
# 		if mass_out != 'none' and age_out == 'none':
# 			# ... stellar mass
# 			# get the points where the temperature is the requested one (or close)
# 			sel_t = np.where(np.abs(t-logT_in)<0.001)
# 	elif (logT_in == 'none' and logL_in == 'none') and (logT_in != 'none' and logL_in == 'none')






# # 			# get the points in the array where you have the requested mass
# # 			sel_m = np.where(m == mass_out)
# # 			# check that this mass is present in the models
# # 			if np.size(sel_m)==0:
# # 				# in this case it is not present - select the closer one
# # 				mass_out_ad = unique(m[m.sort()])[np.where(np.abs(mass_out-unique(m[m.sort()]))==np.min(np.abs(mass_out-unique(m[m.sort()]))))][0]
# # 				if verbose:
# # 					print('Mstar = %f Msun is not available, I give you the results for Mstar = %f Msun' % (mass_out,mass_out_ad))
# # 				sel_m = np.where(m == mass_out_ad)
# # 			# now get the point among the sel_m array positions where logT is closer to the requested
# # 			sel_tm = np.where(np.abs(t[sel_m]-logT_in)==np.min(np.abs(t[sel_m]-logT_in)))
# # 			if verbose:
# # 				print('We found a solution for logT = %f instead of the logT = %s you requested' % (t[sel_m][sel_tm][0],logT_in))
# # 			# then return the logL where this solution is found
# # 			return l[sel_m][sel_tm]
# # 		elif mass_out == 'none' and age_out != 'none':
# # 			# ... age
# # 			# convert the requested age in logage
# # 			logage_out = np.log10(age_out*1e6) # the input age is in Myr
# # 			# get the points in the array where you have the requested age
# # 			sel_a = np.where(np.abs(a - logage_out)<0.01)
# # 			# check that this logage is present in the models
# # 			if np.size(sel_a)==0:
# # 				# in this case it is not present - select the closer one
# # 				logage_out_ad = unique(a[a.sort()])[np.where(np.abs(logage_out-unique(a[a.sort()]))==np.min(np.abs(logage_out-unique(a[a.sort()]))))][0]
# # 				if verbose:
# # 					print('logage = %f  is not available, I give you the results for logage = %f ' % (logage_out,logage_out_ad))
# # 				sel_a = np.where(a == logage_out_ad)
# # 			# now get the point among the sel_a array positions where logT is closer to the requested
# # 			sel_ta = np.where(np.abs(t[sel_a]-logT_in)==np.min(np.abs(t[sel_a]-logT_in)))
# # 			if verbose:
# # 				print('We found a solution for logT = %f instead of the logT = %s you requested' % (t[sel_a][sel_ta][0],logT_in))
# # 			# then return the logL where this solution is found
# # 			return l[sel_a][sel_ta]
