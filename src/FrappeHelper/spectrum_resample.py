#!/usr/bin/env python

			############################################
			#		spectrum_resample.py			   #
			# RESAMPLE THE INPUT SPECTRUM TO NEW WAVEL #
			############################################

# Written on Jan, 3rd 2013 by CFM using the IDL procedure spectrum_resample.pro written by Nicola Da Rio			

# from scipy.interpolate import interp1d
import numpy as np

def spectrum_resample(old_flux,old_wl,new_wl):
	old_wl_first_offset = np.mean(abs(old_wl-np.roll(old_wl,1))[1:])

# adding 2 points equal zero at each end of the old spectrum.
# just to make sure the eventual interpolation outside the original wavelenght range
# doesn't do anything evil but keeps everything equal zero.
	# old_wl_2=reform(old_wl)
	# old_wl_2 = old_wl[:]
	arr1 = np.array([old_wl[0]-old_wl_first_offset*2,old_wl[0]-old_wl_first_offset])
	arr2 = np.array([old_wl[-1]+old_wl_first_offset,old_wl[-1]+old_wl_first_offset*2])
	old_wl_2 = np.concatenate([arr1,old_wl,arr2])
	old_flux_2 = np.concatenate([np.zeros(2),old_flux,np.zeros(2)])

# I define the medianpoints as the average lambda between two consecutive points of the new_wl
	medianpoints = ((new_wl+np.roll(new_wl,-1))/2.)[:-1]

# the "domain" of each point of the new_wl is defined as the range beween its "left" and right "point". These are 
# the points at half way to the previous and next point of new_wl
	leftpoints = np.concatenate([np.array([new_wl[0] - (medianpoints[0]-new_wl[0])]), medianpoints])
	rightpoints = np.concatenate([medianpoints, np.array([new_wl[-1] + (new_wl[-1]-medianpoints[-1])])])

# I interpolate the old spectrum in a new scale that does not remove any of the previous values of old_wl but instead 
# adds all the left and right points of new_wl. Given that we are just adding points, linear interpolation is correct.
	biggerscale = np.concatenate([old_wl_2, leftpoints, rightpoints])
	biggerscale = np.unique(biggerscale) # removing duplicates
	biggerspec = np.interp(biggerscale,old_wl_2,old_flux_2)

# more or less, what I do here is the following:
# for each point of the new_wl I have a range as mentioned above
# if the old scale was much denser here than the new one, I'll have many points in my i-th range
# if the old scale was less denser here than the new one, at least I'll have 2 points in the range
# that map exaclty the old spectrum, i.e. the "left" and "right" points of this i-th wavelength point.
# Either way, I compute my value for the spectrum in the i-th point in a rigorous way that preserves the flux:
# I compute the exact integral of the old spectrum within the range and I divide it by the width of the range
	new_flux = np.zeros(len(new_wl),np.float64)
	for i in range(len(new_wl)):
		Range = (biggerscale >= leftpoints[i]) & (biggerscale <= rightpoints[i])
		widths = (biggerscale[Range]-np.roll(biggerscale[Range],1))[1:]
		totwidth = widths.sum()
		meanfluxes = ((biggerspec[Range]+np.roll(biggerspec[Range],1))[1:])/2.
		flux_int = (meanfluxes*widths)/totwidth
		new_flux[i] = flux_int.sum()


	return new_flux




