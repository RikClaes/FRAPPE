#!/usr/bin/env python

import pylab as pl
import numpy as np
import sys
from int_tabulated import *

# import utilities (various functions)
from eqw_auto_utils import *

def eqw_auto(wl,fl,wave_line,size_cont=1.,plot='YES',mode='gauss',name_out=None,cont_out='NO',err_cont_out='NO',fwhm_out='NO',lline_out='NO',synth_spec='NO',verbose='NO'):
	"""
	# ---------------------------------------------------------------------------------------
	# NAME: eqw_auto.py
	#
	# Purpose: automatically derive Equivalent Width for a given absorption or emission line
	#	
	# OUTPUT:
	# 	EQW, ERR_EQW, MODE[, CONT, ERR_CONT, FWHM, LLINE] = eqw_auto(wave,flux,wave_line,size_cont=1.,plot='YES',mode='gauss'[,name_out=name_out, \
	#						cont_out='NO',err_cont_out='NO',fwhm_out='NO',lline_out='NO',synth_spec='NO'])
	#
	# INPUT:
	# 	wave,flux = wavelength and flux of the input spectrum
	#	wave_line = reference wavelength of the line
	#
	# OPTIONAL INPUT:
	#
	# PROCEDURES USED:
	#
	# Example:
	#	see test_eqw_auto.py in the to_develop folder
	#		eqw,err_eqw,mode[,cont,err_cont,fwhm,wl_line] = eqw_auto(wl,fl,766.5,plot='save',mode='best',name_out='DMTau_KI.png')
	#
	# Created by:
	#		CFM, Mar 15th-20th 2013 @ESO-Garching -- vers 0.0
	#			 
	#
	# ---------------------------------------------------------------------------------------
	"""

	### I ASSUME INPUT WAVELENGTHS ARE IN NM!!! ###

	# PLOT THE LINE REGION
	flmin = 0.9*np.min(fl[ (wl>wave_line-size_cont) & (wl<wave_line+size_cont)])
	flmax = 1.1 * np.max(fl[ (wl>wave_line-size_cont) & (wl<wave_line+size_cont)])
	
	if plot == 'YES' or plot == 'stop':
		pl.figure(1)
		pl.subplot(211)
		pl.plot(wl,fl, 'k',drawstyle='steps-mid',lw=2) # spectrum
		pl.axis([wave_line-1.0,wave_line+1.0, flmin,flmax]) # limits of the plot

	"""
	Estimate continuum:
		do it in a region of 2 nm around the theoretical line position
		by iteratively exclude the points which are outside a given sigma
	"""
	# select wl and fl around the theoretical line position
	wl_cont = wl[(wl >= wave_line-size_cont) & (wl <= wave_line+size_cont)]
	fl_cont = fl[(wl >= wave_line-size_cont) & (wl <= wave_line+size_cont)]

	# fit the continuum in this region using all points
	if plot == 'YES' or plot == 'stop':
		par,res = poly_fit(wl_cont,fl_cont)
	else:
		par,res = poly_fit(wl_cont,fl_cont,plot='NO')

	std_res = np.std(res)

	sigma_cut = 2.

	if plot == 'YES' or plot == 'stop':
		pl.subplot(212)
		pl.plot(wl_cont,res,'gd')
		pl.plot(wl_cont,np.repeat(sigma_cut*std_res,len(wl_cont)),'r',lw=2)	# plot the 2 sigma of the residuals
		pl.plot(wl_cont,np.repeat(-sigma_cut*std_res,len(wl_cont)),'r',lw=2)	# plot the 2 sigma of the residuals
		pl.xlim(wave_line-1.0,wave_line+1.0)
		pl.show()
		if plot =='stop':
			raw_input('First guess')
		pl.close()

	# exclude points with res larger than 2sigma	- ITERATIONS
	niter = 0 
	npoints = len(wl_cont)
	npoints_old = 1e15
	while npoints_old-npoints > 0 and niter <= 20:
		niter+=1

		if verbose != 'NO':
			print('ITERATION #',niter)
		if plot == 'YES' or plot == 'stop':
			pl.figure(2)
			pl.subplot(211)
			pl.plot(wl,fl, 'k',drawstyle='steps-mid',lw=2) # spectrum
			pl.axis([wave_line-1.0,wave_line+1.0, flmin,flmax]) # limits of the plot
	
		wl_cont = wl_cont[np.abs(res) <= sigma_cut*std_res]
		fl_cont = fl_cont[np.abs(res) <= sigma_cut*std_res]

		npoints_old = npoints
		npoints = len(wl_cont)
		if verbose != 'NO':
			print('POINTS %i' % len(wl_cont))
		
		# fit the continuum in this region using all points
		if plot == 'YES' or plot == 'stop':
			par,res = poly_fit(wl_cont,fl_cont)
		else:
			par,res = poly_fit(wl_cont,fl_cont,plot='NO')
		
		# old_std_res = std_res
		std_res = np.std(res)
	
		if plot == 'YES' or plot == 'stop':
			pl.subplot(212)
			pl.plot(wl_cont,res,'gd')
			pl.plot(wl_cont,np.repeat(sigma_cut*std_res,len(wl_cont)),'r',lw=2)	# plot the 2 sigma of the residuals
			pl.plot(wl_cont,np.repeat(-sigma_cut*std_res,len(wl_cont)),'r',lw=2)	# plot the 2 sigma of the residuals
			pl.xlim(wave_line-1.0,wave_line+1.0)
			pl.show()
			if plot =='stop':
				raw_input('Iterating')
			pl.close()



	"""
	NORMALIZE THE SPECTRUM TO THE CONTINUUM
	"""
	Pol = lambda p, x: p[0]*x**2 + p[1]*x + p[2]
	# Pol = lambda p, x: p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3]
	cont = np.mean(Pol(par,np.array((wave_line-0.3,wave_line,wave_line+0.3))))
	cont_err_rob = std_res
	if verbose != 'NO':
		print('Continuum = %1.3e erg s-1 cm-2 nm-1' % cont)
		print('Final sigma_cont = %1.3e erg s-1 cm-2 nm-1' % cont_err_rob)
	fl_norm = fl/Pol(par,wl)


	if plot == 'YES' or plot == 'stop':
		pl.plot(wl,fl_norm, 'k',drawstyle='steps-mid',lw=2)
		pl.plot(wl,np.repeat(1.,len(wl)),'r:',lw=2)
		pl.axis([wave_line-1.0,wave_line+1.0,0.2,1.3])
		pl.show()
		if plot =='stop':
			raw_input('guarda')
		pl.close()


	"""
	IF IT IS A SYNTHETIC SPECTRUM, ASSUME THAT THE INPUT LINE WAVELENGTH IS CORRECT
	"""
	if synth_spec != 'NO':
		wl_zero = wave_line
	else:

		"""
		CALCULATE THE DERIVATIVES TO GET THE LINE POSITION
		"""
		der = deriv_lowres(wl,fl_norm)
		der2 = deriv2_lowres(wl,fl_norm)
		der3 = deriv3_lowres(wl,fl_norm)

		"""
		GET THE II DER MAX POSITION, AND THEN THE ZERO OF THE III DER AROUND THAT POINT
		"""
		# determine the maximum around the line (could be optional, I need the wavelength) - ATT!!! se e' emissione non funziona!
		max_2der = np.max(der2[(wl > (wave_line - size_cont)) & (wl < (wave_line + size_cont))])
		if verbose != 'NO':
			print('MAX II DER = %5.3f' % max_2der)
		# get the wavelength where the maximum is (by creating a fake array)
		fake = der2[(wl > (wave_line - size_cont)) & (wl < (wave_line + size_cont))]
		wl_fake = wl[(wl > (wave_line - size_cont)) & (wl < (wave_line + size_cont))]
		ind_fake = np.argmax(fake)
		wl_max_2der = wl_fake[np.argmax(fake)]
		if verbose != 'NO':
			print('WL OF THE MAX = %5.2f nm' % wl_max_2der)

		# look for a solution of the equation f'''(wl) = 0, i.e. where the III der is 0 in the 'fake' array
		d3_fake = der3[(wl > (wave_line - size_cont)) & (wl < (wave_line + size_cont))]
		# first look for the transition point in wavelength, where one point is positive and the next is negative (or viceversa)
		if d3_fake[ind_fake]*d3_fake[ind_fake+1] < 0.:
			if d3_fake[ind_fake] < d3_fake[ind_fake+1]:
				wl_zero = np.interp(0,[d3_fake[ind_fake],d3_fake[ind_fake+1]],[wl_fake[ind_fake],wl_fake[ind_fake+1]])
			else:
				wl_zero = np.interp(0,[d3_fake[ind_fake+1],d3_fake[ind_fake]],[wl_fake[ind_fake+1],wl_fake[ind_fake]])	
		elif d3_fake[ind_fake]*d3_fake[ind_fake-1] < 0.:
			if d3_fake[ind_fake] > d3_fake[ind_fake-1]:
				wl_zero = np.interp(0,[d3_fake[ind_fake-1],d3_fake[ind_fake]],[wl_fake[ind_fake-1],wl_fake[ind_fake]])
			else:
				wl_zero = np.interp(0,[d3_fake[ind_fake],d3_fake[ind_fake-1]],[wl_fake[ind_fake],wl_fake[ind_fake-1]])
		else:
			sys.exit('Cannot find zero third derivative. Please check!') 
		# using linear interpolation between the points before and after the maximum
		if verbose != 'NO':
			print('PRECISE WL OF THE MAX = %5.3f nm' % wl_zero)

		"""
		PLOT DERIVATIVES
		"""
		if plot == 'YES' or plot == 'stop':
			pl.figure(1)
			pl.subplot(221)
			pl.plot(wl,fl_norm, 'k',lw=2)
			pl.plot([wl_zero,wl_zero],[-10,10],'r')
			pl.plot(wl,np.repeat(1.,len(wl)),'k:')
			pl.axis([wave_line-1.0,wave_line+1.0,0.2,1.3])
			pl.title('INPUT SPECTRUM')

			pl.subplot(222)
			pl.plot(wl,der, 'k',lw=2)
			pl.plot([wl_zero,wl_zero],[-10,10],'r')
			pl.axis([wave_line-1.0,wave_line+1.0,-10,10])
			pl.title('FIRST DERIVATIVE')

			pl.subplot(223)
			pl.plot(wl,der2, 'k',lw=2)
			pl.plot([wl_zero,wl_zero],[-1e10,1e10],'r')
			pl.axis([wave_line-1.0,wave_line+1.0,np.min(fake),np.max(fake)])
			pl.title('SECOND DERIVATIVE')

			pl.subplot(224)
			pl.plot(wl,der3, 'k',lw=2)
			pl.plot([wl_zero,wl_zero],[-1e10,1e10],'r')
			pl.axis([wave_line-1.0,wave_line+1.0,np.min(d3_fake),np.max(d3_fake)])
			pl.title('THIRD DERIVATIVE')

			pl.show()
			if plot =='stop':
				raw_input('guarda')
			pl.close()


	"""
	LINE FITTING
	"""
	int_fit = (wl >= wl_zero-size_cont) & (wl <= wl_zero+size_cont)
	wl_fit = wl[int_fit]
	fl_fit = fl_norm[int_fit]

	if plot == 'YES' or plot == 'stop' or plot=='save':
		pl.figure(1)
		pl.plot(wl,fl_norm, 'k-',lw=2) # spectrum
		# pl.plot(wl_fit,fl_fit,'k-', lw=2)
		pl.axis([wl_zero-1.0, wl_zero+1.0, -0.3, 1.3])

	if mode == 'gauss':
		Gf = lambda p, x: p[0]*np.exp(-(x-p[1])**2/(2*p[2]**2)) #1d Gaussian func
		fit_par,res = gauss_fit(wl_fit,fl_fit,wl_zero,plot)#,init_guess_in=[np.min(fl_fit)-1.,wl_zero,0.2])
		fwhm = 2.53*fit_par[2]
		if plot == 'YES' or plot == 'stop':
			pl.show()
		elif plot == 'save':
			pl.savefig(name_out)
			pl.show()
			pl.clf()
	elif mode == 'lor':	
		Lf = lambda p, x: p[0] * ((0.5*p[2])/((x-p[1])**2+(0.5*p[2])**2)) #1d Lorentzian func
		fit_par,res = lor_fit(wl_fit,fl_fit,wl_zero,plot)
		fwhm = fit_par[2]
		if plot == 'YES' or plot == 'stop':
			pl.show()
		elif plot == 'save':
			pl.savefig(name_out)
			pl.show()
			pl.clf()
	elif mode == 'best':
		fit_par_g,res_g = gauss_fit(wl_fit,fl_fit,wl_zero,plot)
		ind_line = (wl_fit > wl_zero-2.8*fit_par_g[2]) & (wl_fit < wl_zero+2.8*fit_par_g[2])
		std_res_g = np.std(res_g[ind_line])
		if plot == 'YES' or plot == 'stop':
			pl.plot([wl_zero-2.8*fit_par_g[2],wl_zero-2.8*fit_par_g[2]],[-0.2,0.2],'g')
			pl.plot([wl_zero+2.8*fit_par_g[2],wl_zero+2.8*fit_par_g[2]],[-0.2,0.2],'g')
		fit_par_l,res_l = lor_fit(wl_fit,fl_fit,wl_zero)
		# std_res_l = np.std(res_l[(wl_fit > wl_zero-0.93*fit_par_l[2]) & (wl_fit < wl_zero+0.93*fit_par_l[2])])
		std_res_l = np.std(res_l[ind_line])
		if plot == 'YES' or plot == 'stop':
			pl.plot([wl_zero-1.5*fit_par_l[2],wl_zero-1.5*fit_par_l[2]],[-0.2,0.2],'b')
			pl.plot([wl_zero+1.5*fit_par_l[2],wl_zero+1.5*fit_par_l[2]],[-0.2,0.2],'b')
			pl.plot(wl_fit,np.repeat(0,len(wl_fit)),'k:')
			pl.show()
		elif plot == 'save':
			pl.plot([wl_zero-1.5*fit_par_l[2],wl_zero-1.5*fit_par_l[2]],[-0.2,0.2],'b')
			pl.plot([wl_zero+1.5*fit_par_l[2],wl_zero+1.5*fit_par_l[2]],[-0.2,0.2],'b')
			pl.plot(wl_fit,np.repeat(0,len(wl_fit)),'k:')
			pl.savefig(name_out)
			pl.show()
			pl.clf()
		if verbose != 'NO':
			print('Std_res_gauss = %5.2e' % std_res_g)
			print('Std_res_lor = %5.2e' % std_res_l)
		if std_res_g <= std_res_l:
			mode = 'gauss'
			Gf = lambda p, x: p[0]*np.exp(-(x-p[1])**2/(2*p[2]**2)) #1d Gaussian func
			fit_par,res = fit_par_g,res_g
			fwhm = 2.53*fit_par[2]
		else:
			mode = 'lor'
			Lf = lambda p, x: p[0] * ((0.5*p[2])/((x-p[1])**2+(0.5*p[2])**2)) #1d Lorentzian func
			fit_par,res = fit_par_l,res_l
			fwhm = fit_par[2]
	else:
		sys.exit('This fitting method is not defined.')

	# ESTIMATE THE ERROR ON THE CONTINUUM, WHICH IS IDENTIFIED IN THE FIT
	ind_cont = (wl_fit < wl_zero-1.5*fwhm) | (wl_fit > wl_zero+1.5*fwhm)
	cont_err = np.std(fl_fit[ind_cont])*cont


	# raw_input('FUNZIONA????')	


	"""
	Equivalent width estimate using the fit
	"""

	if mode == 'gauss':
		eqw = int_tabulated(wl_fit,-Gf(fit_par,wl_fit))
	elif mode == 'lor':	
		eqw = int_tabulated(wl_fit,-Lf(fit_par,wl_fit))
	else:
		sys.exit('This fitting method is not defined.')

	err_eqw = eqw * (cont_err/cont)

	"""
	OUTPUTS
	"""
	# declare a tuple with the normal output of the function
	result_out = (eqw,err_eqw,mode)
	# add to the tuple the other optional results selected by the user (in the correct order)
	if cont_out != 'NO':	
		result_out = result_out + (cont,)
	if err_cont_out != 'NO':
		result_out = result_out + (cont_err,)
	if fwhm_out != 'NO':
		result_out = result_out + (fwhm,)
	if lline_out != 'NO':
		result_out = result_out + (fit_par[1],)

	# return (eqw,1)
	return result_out
