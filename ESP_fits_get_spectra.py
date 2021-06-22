#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:59:26 2019

@author: jcampbellwhite001
"""

from matplotlib import *
from matplotlib.pyplot import *
import numpy as np
from numpy.random import rand
from glob import glob
from astropy.wcs import WCS
from astropy.io import fits

#testdir='/Users/jcampbellwhite001/OneDrive - University of Dundee/spectra_data/RY_Lup_esp'
#test_fits_file='/Users/jcampbellwhite001/OneDrive - University of Dundee/spectra_data/RY_Lup_esp/1096814i.fits'
#hdulist = fits.open(test_fits_file)   # Open FITS file
#phu    = hdulist[0].header        # Primary Header Unit: metadata
##scihead = hdulist[1].header       # Header of the first FITS extension: metadata
#scidata = hdulist[0].data 


def read_ESP_fits_spec(filename,verbose=False):
    '''
    modified version of ESO python script to get data from ESPADONS FITS files

    Parameters
    ----------
    filename : string
        path to ESPADONS FITS file.
    verbose : bool, optional
        option to print file info to screen. The default is False.

    Returns
    -------
    info : list
        returns list containing target,start_obs,MJD_start_obs,instrume.
    spec_sorted : array
        wavelength 1d array.
    flux_sorted : array
        flux 1d array.
    err_sorted : array
        error 1d array.

    '''
    
    hdulist = fits.open(filename)   # Open FITS file
    
    phu    = hdulist[0].header        # Primary Header Unit: metadata
    scihead = hdulist[0].header       # Header of the first FITS extension: metadata
    scidata = hdulist[0].data         # Data in the first FITS extension: the spectrum
    
    # CVarious keywords must be defined, among them the ones here below:
    try:
        origfile=phu['FILENAME']    # Original filename as assigned by the data provider
        target=phu['OBJECT']    #object target
        start_obs=phu['DATE-OBS']
        MJD_start_obs=phu['MJD-OBS']
        instrume = phu['INSTRUME']  # Name of the instrument
    except:
        print=('ESP_fits_get_spectra() File not compliant with the 1D spectrum specifications; some of the mandatory keywords were not found in primary header unit')
        print('filename = %s   NOT COMPLIANT' % filename)
       
    
    
    
    if verbose == True:
        # Report main characteristics of the spectrum:
        #print('************************************************************************************************************************')
        print('filename=%s   ORIGFILE=%s'  % (filename,origfile))
        print('Target=%s   '  % (target))
        print('Date OBS=%s   '  % (start_obs))
        print('Instrume=%s  '  % (instrume))
        print('------------------------------------------------------------------------------------------------------------------------')
        
    # Main arrays:

    spec = np.array(scidata[0])
    flux = np.array(scidata[1])
    err  = np.array(scidata[2])
    sort_idx=np.argsort(spec)
    spec_sorted=spec[sort_idx]
    flux_sorted=flux[sort_idx]
    err_sorted=err[sort_idx]
    #info=['target,obs_data,mjd_date,instrument,wavemin,wavemax,respwr,snr']
    info = [target,start_obs,MJD_start_obs,instrume,min(spec),max(spec),'N/A','N/A'] 

    
    return info, spec_sorted, flux_sorted, err_sorted


#info,spec,flux,err=read_ESP_fits_spec(test_fits_file,verbose=True)

