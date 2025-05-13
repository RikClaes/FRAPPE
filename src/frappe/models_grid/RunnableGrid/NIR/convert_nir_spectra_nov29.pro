PRO convert_NIR_spectra_Nov29

PATH = '/Users/cmanara/ClassIII_characterization/Final_analysis_Oct2012/data_final/NIR/'

;SIGMA ORI SPECTRA
;print,  flux_correction(PATH+'flux_SO641_NIR_corr_scaled_phot.fits', 6.d-20, PATH+'flux_SO641_NIR_corr_scaled_phot.fits')
;print,  flux_correction(PATH+'flux_SO797_NIR_corr_scaled_phot.fits', 3.6d-19, PATH+'flux_SO797_NIR_corr_scaled_phot.fits')
;print,  flux_correction(PATH+'flux_SO879_NIR_corr_scaled_phot.fits', 3.8d-19, PATH+'flux_SO879_NIR_corr_scaled_phot.fits')
;print,  flux_correction(PATH+'flux_SO925_NIR_corr_scaled_phot.fits', 5.3d-20, PATH+'flux_SO925_NIR_corr_scaled_phot.fits')
;print,  flux_correction(PATH+'flux_SO999_NIR_corr_scaled_phot.fits', 2.1d-19, PATH+'flux_SO999_NIR_corr_scaled_phot.fits')

;Sz121
;spec_readspec, PATH+'flux_Sz121_NIR_corr_scaled_phot.fits', lold, fold, hdr
;spec_readspec, PATH+'flux_Sz121_JH_corr_scaled_phot_new.fits', ljh, fjh
;spec_readspec, PATH+'flux_Sz121_K_corr_scaled_phot_new.fits', lk, fk
;flux = [fjh*10.,fk[1:*]*10.]
;writefits, PATH+'flux_Sz121_NIR_corr_scaled_phot_new.fits',flux,hdr

;Sz122
;spec_readspec, PATH+'flux_Sz122_NIR_corr_scaled_phot.fits', lold, fold, hdr
;spec_readspec, PATH+'flux_Sz122_K_corr_scaled_phot_new.fits', lk, fk
;kband = where(lold ge 1806.96,/Null)
;flux = fold
;flux[kband] = fk
;writefits, PATH+'flux_Sz122_NIR_corr_scaled_phot_new.fits',flux,hdr

stop





END