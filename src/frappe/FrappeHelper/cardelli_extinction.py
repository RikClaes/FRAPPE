#!/usr/bin/env python

##############################################
#		cardelli_extinction.py 			   	 #
#  script to apply a reddening correction 	 #
##############################################

# -------------------------------------------------------------------------------
#  adapted for python from cardelli_extinction.pro, which was adapted from CCM_UNRED.PRO
#  function returns the output flux fraction [from 0 to 1] applying an A_V amount
#  of extinction
# 
#  Syntax: flux_fraction = cardelli_extinction(wavelengths, A_V, [ R_V = ])
# 
#  The reddening curve is that of Cardelli, Clayton, and Mathis (1989 ApJ.
#      345, 245), including the update for the near-UV given by O'Donnell 
#      (1994, ApJ, 422, 158).   Parameterization is valid from the IR to the 
#      far-UV (3.5 microns to 0.1 microns).
# 
#  The input wavelengths are considered in Angstrom
#  This is valid in the wavelength range 900 - 33330 AA
#
#  If you use it to apply a reddening to a spectrum, multiply it for the result of
#  this function, while you should divide by it in the case you want to deredden it.
#
# -------------------------------------------------------------------------------

import numpy as np 


def cardelli_extinction(wave,Av,Rv=3.1):

  ebv = Av/Rv

  # print 'Av = ',Av, 'Rv = ',Rv

  x = 10000./ wave                # Convert to inverse microns 
  npts = len(x)
  a = np.zeros(npts)  
  b = np.zeros(npts)
#******************************

  good = (x > 0.3) & (x  < 1.1)	       #Infrared
  Ngood = np.count_nonzero(good == True)	
  if Ngood > 0:
    a[good] =  0.574 * x[good]**(1.61)
    b[good] = -0.527 * x[good]**(1.61)

#******************************

  good = (x >= 1.1) & (x < 3.3)            #Optical/NIR
  Ngood = np.count_nonzero(good == True)	
  if Ngood > 0:           #Use new constants from O'Donnell (1994)
    y = x[good] - 1.82
    c1 = [-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1.0]  #New coefficients 
    c2 = [3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0.0] #from O'Donnell (1994)
#   c1 = [ 1. , 0.17699, -0.50447, -0.02427,  0.72085,    $ #Original
#                 0.01979, -0.77530,  0.32999 ]               #coefficients
#   c2 = [ 0.,  1.41338,  2.28305,  1.07233, -5.38434,    $ #from CCM89
#                -0.62251,  5.30260, -2.09002 ]   # If you use them remember to revert them
               
    a[good] = np.polyval(c1,y)
    b[good] = np.polyval(c2,y)


#******************************

  good = (x >= 3.3) & (x < 8)            #Mid-UV
  Ngood = np.count_nonzero(good == True)	
  if Ngood > 0:
    y = x[good]
    F_a = np.zeros(Ngood)    
    F_b = np.zeros(Ngood)
    good1 = (y > 5.9)
    Ngood1 = len(good1)	
    if Ngood1 > 0:
    	y1 = y[good1] - 5.9
    	F_a[good1] = -0.04473 * y1**2 - 0.009779 * y1**3
    	F_b[good1] =   0.2130 * y1**2  +  0.1207 * y1**3
    a[good] =  1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a
    b[good] = -3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b
  

#   *******************************

  good = (x >= 8) & (x <= 11)         #Far-UV
  Ngood = np.count_nonzero(good == True)	
  if Ngood > 0:
    y = x[good] - 8.
    c1 = [-0.07, 0.137, -0.628, -1.073]
    c2 = [0.374, -0.42, 4.257, 13.67]
    a[good] = np.polyval(c1,y)
    b[good] = np.polyval(c2,y)

#   *******************************

#=======



  A_lambda = Av * (a + b/Rv)
  # print A_lambda

  ratio =  10.**(-0.4*A_lambda)


# I substitute zero for all the extreme UV wavelenghts (not covered by the cardelli law)
  good = x > 11
  Ngood = np.count_nonzero(good == True)
  if Ngood > 0:
  	ratio[good]=0.

# I extrapolate linearly the law for Mid-IR wavelenghts (not covered by the cardelli law)
# Right now it does not extrapolate outside the validity range --- TO BE DONE
  lasttwo= (x > 0.3)
  # lasttwosort=reverse(sort(lasttwo))
  # xlasttwosort=x[lasttwosort]
  # llasttwosort=wave[lasttwosort]
  # ratiolasttwosort=ratio[lasttwosort]
  xlasttwosort=x[lasttwo][::-1]
  llasttwosort=wave[lasttwo][::-1]
  ratiolasttwosort=ratio[lasttwo][::-1]
  
  mir = x<=0.3
  Nmir = np.count_nonzero(mir == True)
  if Nmir > 0:
    ratio[mir]=np.interp(x[mir],xlasttwosort,ratiolasttwosort)
  bad= ratio > 1
  nbad = np.count_nonzero(bad == True) 
  if nbad > 0:
    ratio[bad]=1


  return ratio




def cardelli_extinction_verbose(wave,Av,Rv=3.1):

  ebv = Av/Rv

  print('Av = ',Av, 'Rv = ',Rv)

  x = 10000./ wave                # Convert to inverse microns 
  npts = len(x)
  a = np.zeros(npts)  
  b = np.zeros(npts)
#******************************

  good = (x > 0.3) & (x  < 1.1)        #Infrared
  Ngood = np.count_nonzero(good == True)  
  if Ngood > 0:
    a[good] =  0.574 * x[good]**(1.61)
    b[good] = -0.527 * x[good]**(1.61)

#******************************

  good = (x >= 1.1) & (x < 3.3)            #Optical/NIR
  Ngood = np.count_nonzero(good == True)  
  if Ngood > 0:           #Use new constants from O'Donnell (1994)
    y = x[good] - 1.82
    c1 = [-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1.0]  #New coefficients 
    c2 = [3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0.0] #from O'Donnell (1994)
#   c1 = [ 1. , 0.17699, -0.50447, -0.02427,  0.72085,    $ #Original
#                 0.01979, -0.77530,  0.32999 ]               #coefficients
#   c2 = [ 0.,  1.41338,  2.28305,  1.07233, -5.38434,    $ #from CCM89
#                -0.62251,  5.30260, -2.09002 ]   # If you use them remember to revert them
               
    a[good] = np.polyval(c1,y)
    b[good] = np.polyval(c2,y)


#******************************

  good = (x >= 3.3) & (x < 8)            #Mid-UV
  Ngood = np.count_nonzero(good == True)  
  if Ngood > 0:
    y = x[good]
    F_a = np.zeros(Ngood)    
    F_b = np.zeros(Ngood)
    good1 = (y > 5.9)
    Ngood1 = len(good1) 
    if Ngood1 > 0:
      y1 = y[good1] - 5.9
      F_a[good1] = -0.04473 * y1**2 - 0.009779 * y1**3
      F_b[good1] =   0.2130 * y1**2  +  0.1207 * y1**3
    a[good] =  1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a
    b[good] = -3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b
  

#   *******************************

  good = (x >= 8) & (x <= 11)         #Far-UV
  Ngood = np.count_nonzero(good == True)  
  if Ngood > 0:
    y = x[good] - 8.
    c1 = [-0.07, 0.137, -0.628, -1.073]
    c2 = [0.374, -0.42, 4.257, 13.67]
    a[good] = np.polyval(c1,y)
    b[good] = np.polyval(c2,y)

#   *******************************

#=======



  A_lambda = Av * (a + b/Rv)
  # print A_lambda

  ratio =  10.**(-0.4*A_lambda)


# I substitute zero for all the extreme UV wavelenghts (not covered by the cardelli law)
  good = x > 11
  Ngood = np.count_nonzero(good == True)
  if Ngood > 0:
    ratio[good]=0.

# I extrapolate linearly the law for Mid-IR wavelenghts (not covered by the cardelli law)
# Right now it does not extrapolate outside the validity range --- TO BE DONE
  lasttwo= (x > 0.3)
  # lasttwosort=reverse(sort(lasttwo))
  # xlasttwosort=x[lasttwosort]
  # llasttwosort=wave[lasttwosort]
  # ratiolasttwosort=ratio[lasttwosort]
  xlasttwosort=x[lasttwo][::-1]
  llasttwosort=wave[lasttwo][::-1]
  ratiolasttwosort=ratio[lasttwo][::-1]
  
  mir = x<=0.3
  Nmir = np.count_nonzero(mir == True)
  if Nmir > 0:
    ratio[mir]=np.interp(x[mir],xlasttwosort,ratiolasttwosort)
  bad= ratio > 1
  nbad = np.count_nonzero(bad == True) 
  if nbad > 0:
    ratio[bad]=1


  return ratio














