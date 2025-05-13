#!/usr/bin/env python

# Script by C.F. Manara

############################################
#			  spec_readspec.py 			   #
# script to read a spectrum from a 1D fits #
############################################

# import pyfits
from astropy.io import fits as pyfits
import numpy as np

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

def readsize2d(spec_im, hdu_sp):
	# determine the X and Y axes coordinate systems from the header of the 2D spectrum
	# created by CFM on Jan, 27th 2015 at ESTEC to read both axes of a 2D spectrum
	# spec_im is a 2D spectrum image, and hdu_sp its hdu read by pyfits
	# returns pos (y-axis) and wave (x-axis)

	#########
	# X-Axis - wavelengths
	#########
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
	n_lambda = len(spec_im[0,:])
	wave = np.zeros(n_lambda)
	for l  in range(n_lambda):
		wave[l] = (l+1.0-crpix1)*cd1_1+crval1
	#Use pixel number starting with 0 if no lambda information is found.
	if (np.min(wave)+np.max(wave) == 0.0):
		print('No lambda information found: used pixel number starting with 0')
		for l  in range(n_lambda):
			wave[l] = l
	#########
	# Y-Axis - spatial
	#########
	crpix2 = hdu_sp[0].header['crpix2']
	#/ value of ref pixel
	crval2 = hdu_sp[0].header['crval2']
	#/ delta per pixel
	if 'cd2_2' in hdu_sp[0].header:
		cd2_2 = hdu_sp[0].header['cd2_2']
	#cd2_2 is sometimes called cdelt2.
	else:
		cd2_2 = hdu_sp[0].header['cdelt2']
	if cd2_2 == 0:
		print("NOT WORKING")
		return
	n_pos = len(spec_im[:,0])
	pos = np.zeros(n_pos)
	for l  in range(n_pos):
		pos[l] = (l+1.0-crpix2)*cd2_2+crval2
	#Use pixel number starting with 0 if no position information is found.
	if (np.min(pos)+np.max(pos) == 0.0):
		print('No position information found: used pixel number starting with 0')
		for l  in range(n_pos):
			pos[l] = l

	return pos,wave
