PRO save_bhac15_isochrones_to_sav

path = '/Users/cmanara/work/utilities/evolutionary_tracks/BHAC15/'

readcol, path+'tracks_bhac15_dense_nolowmass.txt',l,teff,a,m, format='f64,f64,f64,f64',skipline=1

t = alog10(teff)

save,l,t,m,a,filename=path+'tracks_bhac15_dense_nolowmass.sav'

stop


readcol, path+'tracks_bhac15_dense_lowmass.txt',l,t,a,m, format='f64,f64,f64,f64',skipline=1


save,l,t,m,a,filename=path+'tracks_bhac15_dense_lowmass.sav'

stop











END