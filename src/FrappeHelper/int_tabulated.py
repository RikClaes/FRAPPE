#!/usr/bin/env python

			############################################
			#			  int_tabulated.py			   #
			# function to integrate a set of points    #
			############################################

# ---------------------------------------------------------------------------------------
# Purpose:
#	
# OUTPUT:
# 	Result = integral of int
#
# INPUT:
# 	
#
# OPTIONAL INPUT:
# 	
#
# Example:
#	
#
# Created by:
#		CFM, on Dec 30th 2012 @Il Ciocco
#
# ---------------------------------------------------------------------------------------

import numpy as np

def int_tabulated(x,y):
	if len(x) != len(y):
		print('ERROR! x and y should have the same dimensions!')
		return False
	width = np.zeros(len(x)-1)
	ff = np.zeros(len(x)-1)
	for i in range(len(x)-1):
		k = i+1
		width[i] = (x[k] - x[i])
		ff[i] = (y[i]+y[k])/2.

	return np.sum(width*ff)

