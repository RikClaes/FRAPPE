#!/usr/bin/env python

############################################
#			  spec_readspec.py 			   #
# script to read a spectrum from a 1D fits #
############################################

# import pyfits
from astropy.io import fits as pyfits
import numpy as np

def readspec_phase3(file,err_out='NO',hdr_out='NO'):
	# USAGE:
	# wl,fl[,err,hdr] = readspec_GES(filename[,err_out='YES',hdr_out='YES'])
	hdu = pyfits.open(file)
	hdr = hdu[0].header
	if 'FLUX' in hdu[1].columns.names:
		flux = np.array(hdu[1].data['FLUX'],dtype=np.float64)
	elif 'FLUX_REDUCED' in hdu[1].columns.names:
		flux = np.array(hdu[1].data['FLUX_REDUCED'],dtype=np.float64)
	wave = np.array(hdu[1].data['WAVE'],dtype=np.float64)
	if 'ERR' in hdu[1].columns.names:
		err = np.array(hdu[1].data['ERR'],dtype=np.float64)
	elif 'ERR_REDUCED' in hdu[1].columns.names:
		err = np.array(hdu[1].data['ERR_REDUCED'],dtype=np.float64)
	if len(flux) == 1:
		# in this case all the wavelengths are in one line, and you have to get the array in the array
		flux = flux[0]
		wave = wave[0]
		err = err[0]
	hdu.close	
	if err_out=='NO' and hdr_out=='NO':
		return wave,flux
	elif err_out!='NO' and hdr_out=='NO':
		return wave,flux,err
	elif err_out=='NO' and hdr_out!='NO':
		return wave,flux,hdr
	else:
		return wave,flux,err,hdr

def readspec_phase3_tac(file,err_out='NO',hdr_out='NO'):
	# USAGE:
	# wl,fl[,err,hdr] = readspec_GES(filename[,err_out='YES',hdr_out='YES'])
	hdu = pyfits.open(file)
	hdr = hdu[0].header
	flux = np.array(hdu[1].data['tacflux'],dtype=np.float64)
	wave = np.array(hdu[1].data['WAVE'],dtype=np.float64)
	err = np.array(hdu[1].data['ERR'],dtype=np.float64)
	if len(flux) == 1:
		# in this case all the wavelengths are in one line, and you have to get the array in the array
		flux = flux[0]
		wave = wave[0]
		err = err[0]
	hdu.close	
	if err_out=='NO' and hdr_out=='NO':
		return wave,flux
	elif err_out!='NO' and hdr_out=='NO':
		return wave,flux,err
	elif err_out=='NO' and hdr_out!='NO':
		return wave,flux,hdr
	else:
		return wave,flux,err,hdr


# def convert_phase3(file_in,file_out):
# 	wl,flux,hdr = readspec_phase3(file_in,hdr_out='YES')
# 	hdu = pyfits.PrimaryHDU(data=flux,header=hdr)
# 	hdu.writeto(file_out, clobber=True)	# this overwrites if the file is already there
# 	pass

# def convert_phase3_tac(file_in,file_out):
# 	wl,flux,hdr = readspec_phase3_tac(file_in,hdr_out='YES')
# 	hdu = pyfits.PrimaryHDU(data=flux,header=hdr)
# 	hdu.writeto(file_out, clobber=True)	# this overwrites if the file is already there
# 	pass
