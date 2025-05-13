pro isochrone_interpolation_example


PATH_ISOCHRONES = 'PATH_WHERE_SAV_FILES_ARE'
PATH_ISOCHRONES = '/Users/cmanara/ESO_laptop_IDLWorkspace/Evolutionary_tracks/'

Teff = [3270,4350,3060] ;K
lum_sol = [0.174,0.245,0.052] ;L_sun

Nsource = n_elements(Teff)

;=========================
;7) HRD STUFF
;=========================
;Restore Baraffe et al. (1998) isochrones
restore,PATH_ISOCHRONES+'tracks_bcah98_dense.sav'

;if plotting_on eq 'on' or plotting_on eq 'HRD' or plotting_on eq 'save' then begin
;________________ 
;7a) PLOT THE HRD
;****************
p = plot(alog10(Teff), alog10(lum_sol), xrange=[3.65,3.34], xstyle=1, yrange=[-3.8,1],ystyle=1,$
        xtitle='logT!Deff!N [K]', ytitle= 'log (L!D*!N/L$_{sun}$) ',  FONT_NAME = 'Courier',$
        font_size=12, DIMENSIONS = [600,800], name='HRD', 'rs', sym_filled = 1)
;plot, alog10(Teff), alog10(lum_sol), psym=3

p_err = errorplot(alog10(Teff), alog10(lum_sol),0.434*100./Teff, replicate(0.434*0.1,n_elements(Teff)),"rs",/overplot,/current)

ages=[2,10,30,100]
strages=['2','10','30','100']
logages=alog10(ages*1e6)
tsun=5780.
gsun=27400.
;mym=m[uniq(m,sort(m))]
mym = [0.02,0.05,0.1,0.2,0.4,0.6,0.8,1.,1.2]
start=fix(where(mym eq 0.1, /null))
masses = indgen(25)*0.1+0.1
for i=0, n_elements(mym)-1 do begin & sel = where(m eq mym[i], /null) & p1=plot(t[sel], l[sel], linestyle=1, /overplot) &$
      t1=text(t[sel[0]],l[sel[0]]+0.1,strmid(strtrim(mym[i],2),0,4)+' M$_{sun}$', target='HRD', /data, font_size=10, FONT_NAME = 'Courier', alignment=1) &$
      t1.rotate,180,yaxis=1 & endfor

for k=0,n_elements(ages)-1 do begin
  myage=logages[k]
  mvalues=m[uniq(m,sort(m))]
  myt=fltarr(n_elements(mvalues))
  myl=fltarr(n_elements(mvalues))
  for i=0,n_elements(mvalues)-1 do begin
     thistrack=where(m eq mvalues[i], /null)
     if myage ge min(a[thistrack]) and myage le max(a[thistrack]) then begin
       myt[i]=interpol(t[thistrack],a[thistrack],myage)
       myl[i]=interpol(l[thistrack],a[thistrack],myage)
     endif else begin
       myt[i]=!values.f_nan
       myl[i]=!values.f_nan
     endelse
  endfor  ;masses
  myr=sqrt(10^myl/(10^myt/tsun)^4)
  myg=alog10(mym*((10^myt/tsun)^4.)/10^myl*gsun)
;  forprint,myt,myl,mym,myr,myg,comment='LogT,Log(L/Lsun),Mass/solar,Radius/solar,Log(g_surface/cgs) AGE='+strages[k]+'Myr',$
;           textout=outputfolder+'palla_isoch_'+strages[k]+'Myr_BT-Settle.dat',format="(F,F,F,F,F)"  
  p4 = plot(myt[0:*], myl[0:*], linestyle=0, /overplot)
endfor ;ages
;  t4=text(3.44,-1.95,strtrim(ages[0],2)+' Myr', target='TWA', /data, font_size=10, FONT_NAME = 'Courier', alignment=1) 
;      t4.rotate,180,yaxis=1  
;  t4=text(3.47,-2.1,strtrim(ages[1],2)+' Myr', target='TWA', /data, font_size=10, FONT_NAME = 'Courier', alignment=1) 
;      t4.rotate,180,yaxis=1  
;  t4=text(3.44,-2.65,strtrim(ages[2],2)+' Myr', target='TWA', /data, font_size=10, FONT_NAME = 'Courier', alignment=1) 
;      t4.rotate,180,yaxis=1  
;  t4=text(3.45,-3.15,strtrim(ages[3],2)+' Myr', target='TWA', /data, font_size=10, FONT_NAME = 'Courier', alignment=1) 
;      t4.rotate,180,yaxis=1  
;leg = legend(TARGET=[p,p2,p3], POSITION=[0.2,0.30],SAMPLE_WIDTH=0.001)
;  p.save, PATH_PLOT+'plot/HRD.eps'
  p.close
;endif ;plot HRD
stop
;_______________________________
;7a) GET PARAMETERS FROM THE HRD
;*******************************
mass_bara = fltarr(Nsource)
logage_bara = fltarr(Nsource)
for i=0, Nsource-1 do begin 
  distance = sqrt((alog10(Teff[i])-t)^2+(alog10(lum_sol[i])-l)^2)
  best = where(distance eq min(distance))
  mass_bara[i] = m[best]
  logage_bara[i] = a[best]
  if min(distance) gt 0.01 then begin & mass_bara[i]=!values.f_Nan & logage_bara[i]=!values.f_Nan & endif
endfor


stop

end