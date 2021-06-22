#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:24:14 2019
Modified version of the ESO python script to obtain the wavelenth, flux, and error from and ESO phase 3 science FITS file
Requires, ESO FITS file, returns data and necesarry heading info
@author: jcw1
"""
from matplotlib import *
from matplotlib.pyplot import *
from numpy import *
from numpy.random import rand
from glob import glob
from astropy.wcs import WCS
from astropy.io import fits

#test_fits_file='/media/jcw1/2TB_int_HD/STFC/spectra_data/T_Cha/FEROS/ADP.2016-09-23T06:51:14.595.fits'
#hdulist = fits.open(test_fits_file)   # Open FITS file
#phu    = hdulist[0].header        # Primary Header Unit: metadata
#scihead = hdulist[1].header       # Header of the first FITS extension: metadata
#scidata = hdulist[1].data 


def read_ESO_fits_spec(filename,verbose=False):
    '''
    Modified version of ESO python script to get data and info from ESO FITS scienec files.

    Parameters
    ----------
    filename : string
        path to ESO FITS file.
    verbose : bool, optional
        option to print file info to screen. The default is False.

    Returns
    -------
    info : list
        returns list containing target,start_obs,MJD_start_obs,instrume,wavelmin,wavelmax,respower,snr.
    spec : array
        wavelength 1d array.
    flux : array
        flux 1d array.
    err : array
        error 1d array.

    '''
    
    hdulist = fits.open(filename)   # Open FITS file
    
    phu    = hdulist[0].header        # Primary Header Unit: metadata
    scihead = hdulist[1].header       # Header of the first FITS extension: metadata
    scidata = hdulist[1].data         # Data in the first FITS extension: the spectrum
    
    # Checking some compliance
    # 1. keyword PRODCATG must be present in primary header unit
    try:
        prodcatg = phu['PRODCATG']
    except:
        errorMessage = 'Keyword PRODCATG not found in primary header.\nFile not compliant with the ESO Science Data Product standard.'
        print('filename = %s   NOT COMPLIANT' % filename)
        print(errorMessage)
        #exit(1)
    
    # 2. value of keyword PRODCATG must match SCIENCE.SPECTRUM*
    if not prodcatg.startswith('SCIENCE.SPECTRUM'):
        errorMessage = "Expected header keyword: PRODCATG = 'SCIENCE.SPECTRUM'\nFound: PRODCATG = '%s'\nFile not compliant with the 1d spectrum specifications\nof the ESO Science Data Product standard." % prodcatg
        print('filename = %s   NOT COMPLIANT' % filename)
        print(errorMessage)
        #exit(1)
    
    # 3. Various keywords must be defined, among them the ones here below:
    try:
        origfile=phu['ORIGFILE']    # Original filename as assigned by the data provider
        target=phu['OBJECT']    #object target
        start_obs=phu['DATE-OBS']
        MJD_start_obs=phu['MJD-OBS']
        instrume = phu['INSTRUME']  # Name of the instrument
        wavelmin = phu['WAVELMIN']  # Minimum wavelength in nm
        wavelmax = phu['WAVELMAX']  # Maximum wavelength in nm
        respower = phu['SPEC_RES']  # Spectral resolving power (lambda / delta_lambda)
        snr      = phu['SNR']       # Signal to Noise Ratio
        specaxisucd  = scihead['TUCD1'] # Gives the type of spectral axis (see SPECTRAL AXIS below)
    except:
        print=('ESO_fits_get_spectra() File not compliant with the 1D spectrum specifications of the ESO Science Data Product standard; some of the mandatory keywords were not found in primary header unit')
        print('filename = %s   NOT COMPLIANT' % filename)
        exit(1)
    
    try:
        bary_corr=phu['HIERARCH ESO DRS BARYCORR']
        respower=bary_corr
    except:
        bary_corr=0

    
    
    # SPECTRAL AXIS: could be either wavelength, frequency, or energy;
    # if wavelength, the distinction between wavelength in air or in vacuum is provided by the presence of the obs.atmos token in the TUCD1.
    # the variable spectype will carry to whole info.
    spectype = None
    if specaxisucd.startswith('em.wl'):
        if specaxisucd == 'em.wl':
            spectype = 'wavelength in vacuum (TUCD1=%s)' % specaxisucd
        elif specaxisucd == 'em.wl;obs.atmos':
            spectype = 'wavelength in air (TUCD1=%s)' % specaxisucd
        else:
            spectype = 'wavelength (TUCD1=%s)' % specaxisucd
    elif specaxisucd.startswith('em.freq'):
        spectype = 'frequency (TUCD1=%s)' % specaxisucd
    elif specaxisucd.startswith('em.ener'):
        spectype = 'energy (TUCD1=%s)' % specaxisucd
    
    if verbose == True:
        # Report main characteristics of the spectrum:
        print('************************************************************************************************************************')
        print('filename=%s   ORIGFILE=%s'  % (filename,origfile))
        print('Target=%s   '  % (target))
        print('Date OBS=%s   '  % (start_obs))
        print('Instrume=%s   Wmin=%snm   Wmax=%snm   R=%s   SNR=%s'  % (instrume,wavelmin,wavelmax,respower,snr))
        print('Spectral axis: %s' % (spectype))
        print('------------------------------------------------------------------------------------------------------------------------')
        
    # Check VO compliance (ESO SDP is based on the VO standard):
    try:
        voclass=scihead['VOCLASS']
    except:
        print('File %s is not a valid VO 1d spectrum (VOCLASS keyword missing)' % (filename))
        #exit(1)
    
    # TFIELDS is a required FITS binary table keyword
    try:
        tfields=int(scihead['TFIELDS'])
    except:
        print('File %s is not a valid ESO SDP 1d spectrum (TFIELDS keyword missing)' % (filename))
        #exit(1)
    
    #################################
    # METADATA PART
    #################################
    
    # Reading name, unit, utype for each column (array) in the FITS binary table (extension 1).
    
    name = []
    unit = []
    utype= [] # lowercase utype string: for case-insensitive matches
    Utype= [] # original utype, with case preserved: for display
    
    if verbose == True:
        print("AVAILABLE ARRAYS:")
        print ("name            index  UNIT                               UTYPE")
    for i in range(1, tfields+1):
        thisname = scihead['TTYPE'+str(i)]
        try:
           thisunit = scihead['TUNIT'+str(i)]
        except:
           thisunit=""
        try:
           thisutype=scihead['TUTYP'+str(i)]
        except:
           thisutype='no_utype_assigned:field_not_part_of_the_standard'
        if verbose == True:
            print ("%-15s %2d     %-34s [%-s]" % (thisname, i, thisunit, thisutype))
        name.append(thisname)
        unit.append(thisunit)
        utype.append(thisutype.lower())
        Utype.append(thisutype)
    
    if verbose == True:
        print('------------------------------------------------------------------------------------------------------------------------')
    
    # Recognising the main scientific arrays (spectral, flux and flux error) and the "other" ones.
    # A 1D spectrum can contain several flux (and fluxerror) arrays, but one is defined to be the best.
    # The best one can be recognised by the (lowercased) utype which is either "spectrum.data.fluxaxis.value" or "spec:data.fluxaxis.value".
    
    other_arrays = []  # It will contain the indeces of the fields not considered main arrays. FITS indeces starts from 1!
    
    # Getting the indexes of the FITS columns
    # for the main spectral array (ispec), flux array (iflux), and flux_error (ierr) array:
    
    for i in range(1, tfields+1):
    
         # Remember that the index of Python arrays starts from 0, while the FITS index from 1.
         tutyp=utype[i-1]
    
         # The ESO Science Data Product standard format
         # prescribes the spectral axis to be stored in column 1;
         # there would be no need to look for it, but we need the other_arrays anyway.
    
         # The TUTYPn keywords follow either the Spectrum Data Model standard v1.1 for spectra with a single flux array,
         # or the Spectral Data Model standard v2.0 for spectra with any number of flux arrays
         # These data model standards are available from the International Virtual Observatory Alliance
         # web site at: http://ivoa.net/documents/
    
         if tutyp == 'spectrum.data.spectralaxis.value':
             ispec = i
         elif tutyp == 'spec:data.spectralaxis.value':
             ispec = i
         elif tutyp == 'spectrum.data.fluxaxis.value':
             iflux = i
         elif tutyp == 'spec:data.fluxaxis.value':
             iflux = i
         elif tutyp == 'spectrum.data.fluxaxis.accuracy.staterror':
             ierr  = i
         elif tutyp == 'spec:data.fluxaxis.accuracy.staterror':
             ierr  = i
         #else:
             # Storing the indeces of other, not considered main, arrays:
             #other_arrays.append( i )
    
    # --------------------------------------------------------------------------------------------------------
    # Checking if other flux and fluxerr arrays exist, and coupling them by the namespace in the utype
    # E.g.: eso:Data.FluxAxis.Value and eso:Data.FluxAxis.Accuracy.StatError form a couple, in the sense
    # that the second array is the error related to the flux stored in the first array.
    
    #coupling_flux_and_fluxerr_arrays()
    
    # --------------------------------------------------------------------------------------------------------
    
    
    # Number of points in the spectrum: NELEM
    NELEM = scihead['NELEM']
     
    if verbose == True:
        print ("MAIN ARRAYS:")
        print ("                   name          index   comment")
        print ("  Spectral column: %-12s %2d       %s" % (name[ispec-1], ispec, spectype))
        print ("      Flux column: %-12s %2d" % (name[iflux-1], iflux))
        print ("Flux Error column: %-12s %2d" % (name[ierr-1], ierr))
        print('------------------------------------------------------------------------------------------------------------------------')
        
        #################################
        # DATA PART and plots
        #################################
        
        print("\nThe spectrum has %d points\n" % (NELEM))
        print('------------------------------------------------------------------------------------------------------------------------')

    
    # Main arrays:

    #info=['target,obs_data,mjd_date,instrument,wavemin,wavemax,respwr,snr']
    info = [target,start_obs,MJD_start_obs,instrume,wavelmin,wavelmax,respower,snr] 
    spec = np.array(scidata[0][ispec - 1])
    flux = np.array(scidata[0][iflux - 1])
    err  = np.array(scidata[0][ierr - 1])
    return info, spec, flux, err


#info,spec,flux,err=read_ESO_fits_spec(test_fits_file,verbose=True)

