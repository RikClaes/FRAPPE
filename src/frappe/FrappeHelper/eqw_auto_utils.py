#!/usr/bin/env python
"""
FUNCTIONS NEEDED IN eqw_auto.py
"""

import pylab as pl
import numpy as np
from scipy.optimize import leastsq
from int_tabulated import *
#from smooth import *



def poly_fit(wl_fit,fl_fit,plot='YES'):
	"""
	Fit the given points with a second degree polynomial
	"""
	Pol = lambda p, x: p[0]*x**2 + p[1]*x + p[2]
	res_fit = lambda p, x, y: (Pol(p,x) - y)

	out = leastsq(res_fit, [1.,1.,1.], args=(wl_fit,fl_fit), full_output=1)

	fit_par = out[0] #fit parameters out
	covar = out[1] #covariance matrix output

	# print 'p[0], a: ', fit_par[0]
	# print 'p[1], b: ', fit_par[1]
	# print 'p[2], c: ', fit_par[2]

	if plot == 'YES' or plot == 'save':
		pl.plot(wl_fit,Pol(fit_par,wl_fit),'g--',lw=2)
		pl.show()

	return fit_par, res_fit(fit_par,wl_fit,fl_fit)

# def poly3_fit(wl_fit,fl_fit,plot='YES'):
# 	"""
# 	Fit the given points with a third degree polynomial
# 	"""
# 	Pol = lambda p, x: p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3]
# 	res_fit = lambda p, x, y: (Pol(p,x) - y)

# 	out = leastsq(res_fit, [1.,1.,1.,1.], args=(wl_fit,fl_fit), full_output=1)

# 	fit_par = out[0] #fit parameters out
# 	covar = out[1] #covariance matrix output

# 	# print 'p[0], a: ', fit_par[0]
# 	# print 'p[1], b: ', fit_par[1]
# 	# print 'p[2], c: ', fit_par[2]

# 	if plot == 'YES' or plot == 'save':
# 		pl.plot(wl_fit,Pol(fit_par,wl_fit),'g--',lw=2)
# 		pl.show()

# 	return fit_par, res_fit(fit_par,wl_fit,fl_fit)

def deriv(x,a):
	# First derivative of vector using 2-point central difference.
	#  T. C. O'Haver, 1988.
	n=len(a)
	d=np.ones(n)
	d[0]=(a[1]-a[0])/(x[1]-x[0])
	d[-1]=(a[-2]-a[-1])/(x[-1]-x[-2])
	for j in range(1,n-1):
		d[j]=(a[j+1]-a[j-1])/ (2.*(x[j+1]-x[j]))
	d = smooth(d,8,smoothtype='hanning',downsample=False)
	return d

def deriv_lowres(x,a):
	# First derivative of vector using 2-point central difference.
	#  T. C. O'Haver, 1988.
	n=len(a)
	d=np.ones(n)
	d[0]=(a[1]-a[0])/(x[1]-x[0])
	d[-1]=(a[-2]-a[-1])/(x[-1]-x[-2])
	for j in range(1,n-1):
		d[j]=(a[j+1]-a[j-1])/ (2.*(x[j+1]-x[j]))
	# d = smooth(d,3,smoothtype='hanning',downsample=False)
	return d

def deriv_hires(x,a):
	# First derivative of vector using 2-point central difference.
	#  T. C. O'Haver, 1988.
	n=len(a)
	d=np.ones(n)
	d[0]=(a[1]-a[0])/(x[1]-x[0])
	d[-1]=(a[-2]-a[-1])/(x[-1]-x[-2])
	for j in range(1,n-1):
		d[j]=(a[j+1]-a[j-1])/ (2.*(x[j+1]-x[j]))
	d = smooth(d,15,smoothtype='hanning',downsample=False)
	return d

def deriv2(x,a):
	# Second derivative of vector using 3-point central difference.
	#  T. C. O'Haver, 2006.
	n=len(a)
	d2 = np.ones(n)
	for j in range(1,n-1):
		d2[j]= (a[j+1] - 2.*a[j] + a[j-1]) / ((x[j+1]-x[j])**2)
	d2[0]=d2[1]
	d2[-1]=d2[-2]
	d2 = smooth(d2,8,smoothtype='hanning',downsample=False)
	return d2

def deriv2_lowres(x,a):
	# Second derivative of vector using 3-point central difference.
	#  T. C. O'Haver, 2006.
	n=len(a)
	d2 = np.ones(n)
	for j in range(1,n-1):
		d2[j]= (a[j+1] - 2.*a[j] + a[j-1]) / ((x[j+1]-x[j])**2)
	d2[0]=d2[1]
	d2[-1]=d2[-2]
	# d2 = smooth(d2,3,smoothtype='hanning',downsample=False)
	return d2

def deriv2_hires(x,a):
	# Second derivative of vector using 3-point central difference.
	#  T. C. O'Haver, 2006.
	n=len(a)
	d2 = np.ones(n)
	for j in range(1,n-1):
		d2[j]= (a[j+1] - 2.*a[j] + a[j-1]) / ((x[j+1]-x[j])**2)
	d2[0]=d2[1]
	d2[-1]=d2[-2]
	d2 = smooth(d2,15,smoothtype='hanning',downsample=False)
	return d2

def deriv2_vhires(x,a):
	# Second derivative of vector using 3-point central difference.
	#  T. C. O'Haver, 2006.
	n=len(a)
	d2 = np.ones(n)
	for j in range(1,n-1):
		d2[j]= (a[j+1] - 2.*a[j] + a[j-1]) / ((x[j+1]-x[j])**2)
	d2[0]=d2[1]
	d2[-1]=d2[-2]
	d2 = smooth(d2,31,smoothtype='hanning',downsample=False)
	return d2

def deriv3(x,a):
	# Third derivative of vector a
	#  T. C. O'Haver, 2008.
	n=len(a)
	d3 = np.ones(n)
	for j in range(2,n-2):
		d3[j]= (a[j+2] - 2.*a[j+1] + 2.*a[j-1] - a[j-2]) / (2.*(x[j+1]-x[j])**3)
	d3[0:1]=d3[2]
	d3[-2:-1]=d3[-3]
	d3 = smooth(d3,8,smoothtype='hanning',downsample=False)
	return d3

def deriv3_lowres(x,a):
	# Third derivative of vector a
	#  T. C. O'Haver, 2008.
	n=len(a)
	d3 = np.ones(n)
	for j in range(2,n-2):
		d3[j]= (a[j+2] - 2.*a[j+1] + 2.*a[j-1] - a[j-2]) / (2.*(x[j+1]-x[j])**3)
	d3[0:1]=d3[2]
	d3[-2:-1]=d3[-3]
	# d3 = smooth(d3,3,smoothtype='hanning',downsample=False)
	return d3

def deriv3_hires(x,a):
	# Third derivative of vector a
	#  T. C. O'Haver, 2008.
	n=len(a)
	d3 = np.ones(n)
	for j in range(2,n-2):
		d3[j]= (a[j+2] - 2.*a[j+1] + 2.*a[j-1] - a[j-2]) / (2.*(x[j+1]-x[j])**3)
	d3[0:1]=d3[2]
	d3[-2:-1]=d3[-3]
	d3 = smooth(d3,15,smoothtype='hanning',downsample=False)
	return d3

def deriv3_vhires(x,a):
	# Third derivative of vector a
	#  T. C. O'Haver, 2008.
	n=len(a)
	d3 = np.ones(n)
	for j in range(2,n-2):
		d3[j]= (a[j+2] - 2.*a[j+1] + 2.*a[j-1] - a[j-2]) / (2.*(x[j+1]-x[j])**3)
	d3[0:1]=d3[2]
	d3[-2:-1]=d3[-3]
	d3 = smooth(d3,31,smoothtype='hanning',downsample=False)
	return d3

def gauss_fit_old(wl_fit,fl_fit,wave_line,plot='YES'):
	"""
	Fit with a gaussian the line
	"""
	Gf = lambda p, x: p[0]*np.exp(-(x-p[1])**2/(2*p[2]**2)) #1d Gaussian func
	e_gauss_fit = lambda p, x, y: (Gf(p,x) - y) #1d Gaussian fit

	init_guess = [np.min(fl_fit)-1.,wave_line,0.05]  #inital guesses for Gaussian Fit

	out = leastsq(e_gauss_fit, init_guess, args=(wl_fit,fl_fit-1.), full_output=1) #Gauss Fit, maxfev=100000

	fit_par = out[0] #fit parameters out
	covar = out[1] #covariance matrix output

	# print 'p[0], A: ', fit_par[0]
	# print 'p[1], x0: ', fit_par[1]
	# print 'p[2], sigma: ', fit_par[2]

	if plot == 'YES' or plot == 'save':
		pl.plot(wl_fit,Gf(fit_par,wl_fit)+1.,'g--',lw=2)
		pl.plot(wl_fit,e_gauss_fit(fit_par,wl_fit,fl_fit-1.),'g',drawstyle='steps-mid',lw=2)
		pl.show()

	return fit_par, e_gauss_fit(fit_par,wl_fit,fl_fit-1.)

def gauss_fit(wl_fit,fl_fit,wave_line,plot='YES',init_guess_in='none'):
	"""
	Fit with a gaussian the line
	"""
	Gf = lambda p, x: p[0]*np.exp(-(x-p[1])**2/(2*p[2]**2)) #1d Gaussian func
	e_gauss_fit = lambda p, x, y: (Gf(p,x) - y) #1d Gaussian fit

	if init_guess_in=='none':	# 2015-07-13 to give different initial guesses
		init_guess = [np.min(fl_fit)-1.,wave_line,0.05]  #inital guesses for Gaussian Fit
	else:
		init_guess = init_guess_in

	out = leastsq(e_gauss_fit, init_guess, args=(wl_fit,fl_fit-1.), full_output=1) #Gauss Fit, maxfev=100000

	fit_par = out[0] #fit parameters out
	covar = out[1] #covariance matrix output

	# print 'p[0], A: ', fit_par[0]
	# print 'p[1], x0: ', fit_par[1]
	# print 'p[2], sigma: ', fit_par[2]

	if plot == 'YES' or plot == 'save' or plot =='stop':
		pl.plot(wl_fit,Gf(fit_par,wl_fit)+1.,'g--',lw=2)
		pl.plot(wl_fit,e_gauss_fit(fit_par,wl_fit,fl_fit-1.),'g',drawstyle='steps-mid',lw=2)
		pl.show()

	return fit_par, e_gauss_fit(fit_par,wl_fit,fl_fit-1.)


def lor_fit(wl_fit,fl_fit,wave_line,plot='YES'):
	"""
	Fit with a lorentzian the line
	"""
	Lf = lambda p, x: p[0] * ((0.5*p[2])/((x-p[1])**2+(0.5*p[2])**2)) #1d Lorentzian func
	e_lor_fit = lambda p, x, y: (Lf(p,x) - y) #1d lorentzian fit

	init_guess = [np.min(fl_fit)-1.,wave_line,0.05]  #inital guesses for lorentzian Fit

	out = leastsq(e_lor_fit, init_guess, args=(wl_fit,fl_fit-1.), full_output=1) #lor Fit, maxfev=100000

	fit_par = out[0] #fit parameters out
	covar = out[1] #covariance matrix output

	# print 'p[0], A: ', fit_par[0]
	# print 'p[1], x0: ', fit_par[1]
	# print 'p[2], FWHM: ', fit_par[2]

	if plot == 'YES' or plot == 'save':
		pl.plot(wl_fit,Lf(fit_par,wl_fit)+1.,'b--',lw=2)
		pl.plot(wl_fit,e_lor_fit(fit_par,wl_fit,fl_fit-1.),'b',drawstyle='steps-mid',lw=2)
		pl.show()

	return fit_par, e_lor_fit(fit_par,wl_fit,fl_fit-1.)
