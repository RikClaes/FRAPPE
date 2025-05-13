PRO save_feiden_isochrones_to_sav

path = '/lustre/opsw/work/cmanara/evolutionary_tracks/Feiden/'

readcol, path+'tracks_feiden_dense.txt',l,t,a,m, format='f64,f64,f64,f64',skipline=1

save,l,t,m,a,filename=path+'tracks_feiden_dense.sav'

stop



END