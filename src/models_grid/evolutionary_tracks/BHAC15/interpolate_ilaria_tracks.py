# 2016-06-10 - CFM@ESTEC
# compare the Baraffe+15 tracks I created with those that Ilaria sent me
# these extend to lower Mstar than the normal ones

import numpy as np
from scipy.io.idl import readsav
import sys
import pylab as pl


path = '/Users/cmanara/work/utilities/evolutionary_tracks/BHAC15/'

# read my interpolated tracks
s = readsav(path+'tracks_bhac15_dense_nolowmass.sav')
lcfm,tcfm,mcfm,acfm = s['l'],s['t'],s['m'],s['a']
s.clear()

# read those by Ilaria
s = readsav(path+'Baraffe_AgeMassGrid.sav')
lti,mi,ai = s['logt_logl'],s['mass'],s['log_age']
s.clear()

# the tracks by Ilaria enxtend only to logage=7.7 (age=50 Myr)
print np.max(np.unique(ai))
# based on the email from Isabelle to Greg (see March 17th 2016) it seems better to stop there
# I then try to interpolate these tracks to a much finer grid

# the lti array contains loglstar and logteff for each mstar and age. logt are lti[:,:,0], while logl are lti[:,:,1]
# the first column is corresponding to ages (200), the second column to masses (198)

# example
# pl.figure()
# pl.plot(lti[:,:,0],lti[:,:,1],'rx')
# pl.xlim(3.9,3.2)
# # plot one isochrone
# pl.plot(lti[10,:,0],lti[10,:,1],'bx')
# # another
# pl.plot(lti[150,:,0],lti[150,:,1],'bx')
# # plot one track
# pl.plot(lti[:,150,0],lti[:,150,1],'g+')
# # another
# pl.plot(lti[:,15,0],lti[:,15,1],'g+')
# # this corresponds to Mstar = 0.02 Msun
# print mi[15]
# # the previous one to Mstar = 0.47 Msun
# print mi[150]
# pl.close('all')


# I'll proceed this way: first interpolate each Mstar on the same logage scale (from 5.71 to 7.7)
# then I will interpolate at each age on more Mstar

minlogage,maxlogage = 5.71,7.7

# create the arrays where to store the variables
loga_new, m_new, logl_new, logt_new = np.array([]),np.array([]),np.array([]),np.array([])


# INTERPOLATE ON LOGAGE
# define a dense logage sequence - there are at most 372 points between 5.7 and maxlogage in the original models
logage_steps = np.linspace(minlogage,maxlogage,num=400)


# for each unique Mstar, interpolate the track on the same logage

pl.figure(figsize=(15,12))
pl.plot(lti[:,:,0],lti[:,:,1],'rx')
pl.xlabel('logTeff [K]')
pl.ylabel('log(Lstar/Lsun)')
ax = pl.gca()
ax.invert_xaxis()

for i in xrange(len((mi))):
    # interpolate logL and Teff
    logl_temp = np.interp(logage_steps, ai, lti[:,i,1])
    logteff_temp = np.interp(logage_steps, ai, lti[:,i,0])           
    # plot to check
    pl.plot(logteff_temp,logl_temp,'b+')
    # append logL, Teff, logage, mstar to the arrays with the interpolated isochrones
    loga_new = np.append(loga_new,logage_steps)
    m_new = np.append(m_new,np.repeat(np.unique(mi)[i],len(logage_steps)))
    logl_new = np.append(logl_new,logl_temp)
    logt_new = np.append(logt_new,logteff_temp)

pl.show()

print len(loga_new),len(m_new), len(logl_new),len(logt_new)



# INTERPOLATE ON Mstar
# define a dense mstar sequence 
mstar_steps_init = 10.**(np.linspace(np.min(np.log10(mi)),np.max(np.log10(mi)),num=400))


# but exclude those masses already included in the interpolated grid
mstar_steps = np.array([0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2,1.3,1.4])
for i in xrange(len(mstar_steps_init)):
#    if mstar_steps_init[i] not in np.unique(mi):
    if np.all(np.abs(mstar_steps_init[i] - (mi)) > 1e-12) :
        mstar_steps = np.append(mstar_steps,mstar_steps_init[i])
    else:
        pass



# now for each logage step, interpolate on the new Mstar step

pl.figure(figsize=(15,12))
pl.plot(lti[:,:,0],lti[:,:,1],'rx',label='ORIGINAL')
pl.xlabel('logTeff [K]')
pl.ylabel('log(Lstar/Lsun)')
ax = pl.gca()
ax.invert_xaxis()

for i in xrange(len(logage_steps)):
    # interpolate logL and Teff
    logl_temp = np.interp(mstar_steps, m_new[loga_new == logage_steps[i]], logl_new[loga_new == logage_steps[i]])
    logteff_temp = np.interp(mstar_steps, m_new[loga_new == logage_steps[i]], logt_new[loga_new == logage_steps[i]])           
    # plot to check
    pl.plot(logteff_temp,logl_temp,'go',ms=3)
    # append logL, Teff, logage, mstar to the arrays with the interpolated isochrones
    loga_new = np.append(loga_new,np.repeat(logage_steps[i],len(mstar_steps)))
    m_new = np.append(m_new,mstar_steps)
    logl_new = np.append(logl_new,logl_temp)
    logt_new = np.append(logt_new,logteff_temp)

pl.show()

print len(loga_new),len(m_new), len(logl_new),len(logt_new)


# seee the difference
pl.plot(logt_new,logl_new,'rx')
pl.xlim(3.9,3.2)
pl.plot(lti[:,:,0],lti[:,:,1],'b+')
pl.show()


# check by plotting some tracks
pl.figure(figsize=(15,12))
pl.plot(lti[:,:,0],lti[:,:,1],'rx')#,label='ORIGINAL')
pl.xlabel('logTeff [K]')
pl.ylabel('log(Lstar/Lsun)')
ax = pl.gca()
ax.invert_xaxis()
pl.plot((logt_new[m_new==np.unique(m_new)[10]]),logl_new[m_new==np.unique(m_new)[10]],'b+',label='INTERP')
pl.plot((logt_new[m_new==np.unique(m_new)[50]]),logl_new[m_new==np.unique(m_new)[50]],'b+')
pl.plot((logt_new[m_new==np.unique(m_new)[80]]),logl_new[m_new==np.unique(m_new)[80]],'b+')
pl.plot((logt_new[m_new==np.unique(m_new)[100]]),logl_new[m_new==np.unique(m_new)[100]],'b+')
pl.plot((logt_new[m_new==np.unique(m_new)[150]]),logl_new[m_new==np.unique(m_new)[150]],'b+')
pl.plot((logt_new[m_new==np.unique(m_new)[200]]),logl_new[m_new==np.unique(m_new)[200]],'b+')
pl.plot((logt_new[m_new==np.unique(m_new)[240]]),logl_new[m_new==np.unique(m_new)[240]],'b+')
pl.plot((logt_new[m_new==np.unique(m_new)[350]]),logl_new[m_new==np.unique(m_new)[350]],'b+')

pl.legend(loc='upper right')
#for i in xrange(len(np.unique(m_new))):
 #   pl.plot((logt_new[m_new==np.unique(m_new)[i]]),logl_new[m_new==np.unique(m_new)[i]],'b+')
#  i+=100
pl.show()


# In[66]:

# save everything in an enormous file (I need to read it in IDL)
np.savetxt(path+'tracks_bhac15_dense_lowmass.txt',np.transpose([logl_new,logt_new,loga_new,m_new]),header='logL logTeff logAge Mstar')



