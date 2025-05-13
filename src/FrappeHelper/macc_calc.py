#!/usr/bin/env python

##################################
#		macc_calc.py   		   	 #
#  script to calculate macc 	 #
##################################

# -------------------------------------------------------------------------------
#  From Lstar and Teff gets Mstar (using ev.models) and Rstar. With also Lacc
#	calculates Macc using the relation Macc = (Lacc*Rstar)/(0.8*G*Mstar)
#
#  Syntax: macc = macc_calc(Teff,Lstar,Lacc,model='Siess')
#
#	uses the  function isochrone_interp
#
# -------------------------------------------------------------------------------

import numpy as np
from isochrone_alpha import *



def macc_calc(Teff,Lstar,Lacc='none',model='Siess',PATH = None):
	"""
	#  From Lstar and Teff gets Mstar (using ev.models) and Rstar. With also Lacc
	#	calculates Macc using the relation Macc = (Lacc*Rstar)/(0.8*G*Mstar)
	#
	#  Syntax: mstar[,macc] = macc_calc(Teff,Lstar[,Lacc,model='Siess'])
	#
	#	uses the  function isochrone_interp
	"""

	# constants
	Msun = 1.989e33
	Lsun = 3.839e33
	Rsun = 6.955e10
	sig_Bolt = 5.67e-5
	G = 6.67e-8
	yr = np.pi*1e7

	Nsource = len(np.atleast_1d(Lstar))

	if Nsource == 1:
		Rstar = np.sqrt(Lstar*Lsun / (4.*np.pi*sig_Bolt*float(Teff)**4)) / Rsun
		Mstar,logage = isochrone_interp([np.log10(Teff)],[np.log10(Lstar)],model=model,PATH =PATH)
		if Lacc != 'none':
			Macc = Lacc*Lsun * Rstar*Rsun / ( 0.8*G*Mstar*Msun ) * yr/Msun

	else:
		# definitions
		Rstar = np.zeros(Nsource)
		Mstar = np.zeros(Nsource)
		Macc = np.zeros(Nsource)

		for i in range(Nsource):
			Rstar[i] = np.sqrt(Lstar[i]*Lsun / (4.*np.pi*sig_Bolt*float(Teff[i])**4)) / Rsun
			Mstar[i],logage = isochrone_interp([np.log10(Teff[i])],[np.log10(Lstar[i])],model=model,PATH =PATH)
			if Lacc[i] != 'none':
				Macc[i] = Lacc[i]*Lsun * Rstar[i]*Rsun / ( 0.8*G*Mstar[i]*Msun ) * yr/Msun

	if Nsource == 1:
		if Lacc != 'none':
			return Mstar,Macc
		else:
			return Mstar
	else:
		if np.all(Lacc != 'none'):
			return Mstar,Macc
		else:
			return Mstar




def macc_calc_mstar(Teff,Lstar,Mstar,Lacc):
	"""
	#  From Lstar and Teff gets Rstar. Mstar and Lacc are inputs.
	#	calculates Macc using the relation Macc = (Lacc*Rstar)/(0.8*G*Mstar)
	#
	#  Syntax: macc = macc_calc_mstar(Teff,Lstar,Mstar,Lacc)
	"""

	# constants
	Msun = 1.989e33
	Lsun = 3.839e33
	Rsun = 6.955e10
	sig_Bolt = 5.67e-5
	G = 6.67e-8
	yr = np.pi*1e7

	Nsource = len(np.atleast_1d(Lstar))

	if Nsource == 1:
		Rstar = np.sqrt(Lstar*Lsun / (4.*np.pi*sig_Bolt*np.float(Teff)**4)) / Rsun
		Macc = Lacc*Lsun * Rstar*Rsun / ( 0.8*G*Mstar*Msun ) * yr/Msun

	else:
		# definitions
		Rstar = np.zeros(Nsource)
		Macc = np.zeros(Nsource)

		for i in range(Nsource):
			Rstar[i] = np.sqrt(Lstar[i]*Lsun / (4.*np.pi*sig_Bolt*np.float(Teff[i])**4)) / Rsun
			Macc[i] = Lacc[i]*Lsun * Rstar[i]*Rsun / ( 0.8*G*Mstar[i]*Msun ) * yr/Msun

	return Macc
