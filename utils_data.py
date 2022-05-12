#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:47:23 2019

@author: jcw1
"""
import os
import shutil
import pandas as pd
import numpy as np
from ESO_fits_get_spectra import *
from ESP_fits_get_spectra import *
from astropy.io import fits
from astropy.time import Time,TimeDelta
from astroquery.simbad import Simbad
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display,clear_output


#SIMBAD astroquery setup
customSimbad = Simbad()
customSimbad.add_votable_fields('sp','mk','velocity','rot','rv_value')

#testdir='/Users/jcampbellwhite001/OneDrive - University of Dundee/spectra_data/'
#testdir='/media/jcw1/2TB_int_HD/STFC/spectra_data/T_Cha/'

def listdir_fullpath(d):
    '''
    function to list full path of files in dir D

    Parameters
    ----------
    d : string
        path to directory containing files.

    Returns
    -------
    list
        list of files with full system path within directory.

    '''
    return [os.path.join(d, f) for f in os.listdir(d)]


def get_files(d,*args):
    '''
    function to return full path of child files for given directory d
        optional file extnesion can be given by *arg

    Parameters
    ----------
    d : str
        name of directory to search.
    *args : str within ''
        optional extension to only return, e.g. 'fits' .

    Returns
    -------
    matches : list of str
        returns list of filenames found in directory with matching extension if given.

    '''

    matches = []
    #print (len(args))
    for root, dirnames, filenames in os.walk(d):
        for filename in filenames:
            if filename!='.DS_Store':
                if len(args) > 0:
                    if filename.endswith((args)):
                        matches.append(os.path.join(root, filename))
                else:
                    matches.append(os.path.join(root, filename))
    return matches

def ensure_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print('creating directory %s'%(directory))

#m=get_files(testdir,'fits')#,'.txt')
#hdulist = fits.open(m[1])   # Open FITS file
#scihead = hdulist[0].header       # Header of the first FITS extension: metadata
#scidata = hdulist[0].data 
#wave=scidata[0]
#flux=scidata[1]

#fits_list=get_files(os.path.join(testdir,'DI_Cha'),'.fits')

def closest(lst, K):   
    '''
    Function to find closest entry in list lst for given search term K

    Parameters
    ----------
    lst : list
        input list to search through.
    K : any
        target to search for.

    Returns
    -------
    value
        value from lst cloests to K.

    '''
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

def ecdf(x):
    figure()
    ylabel('Fraction of data')
    for data in x:
        xs = np.sort(data)
        ys = np.arange(1, len(xs)+1)/float(len(xs))
        plot(xs,ys)
    #return xs, ys
#xs,ys=ecdf(f_flat)

    
def show_output(plot='function'):
    '''
    ask for user input whether to show output on screen

    Returns
    -------
    bool
        returns true for user answer yes false otherwise.

    '''
    output=input('show output on screen of "%s" y/n?:  '%(plot))
    if output[0].lower()=='y':
        return True
    else:
        return False

def get_fits_files_simbad(target,standards_dir,simbad_out=False):
    '''
    look up star in SIMBAD and get spectral type 
    then match to closest template star by spectral type and get template star RV

    Parameters
    ----------
    target : str
        name of target star, will search SIMBAD for this name.
    standards_dir : str
        path to standard stars FITS files.

    Returns
    -------
    standard_fits_files : list
        list of standard fits files, to use for RV and Vsini calculations.
    mk : str
        spectral type of target star from SIMBAD.
    stype : str
        spectral type of closest standard star in standards_dir .
    st_rv : float
        radial velocity of template star from SIMBAD.

    '''
    try:
        simbad_table=customSimbad.query_object(target)
        mk=simbad_table['SP_TYPE'][0].decode("utf-8")
        print('spectral type of ',target,' is:', mk)
    except:
        try:
            simbad_table=customSimbad.query_object(target)
            mk=simbad_table['SP_TYPE'][0]
            print('spectral type of ',target,' is:', mk)
        except:
            print('simbad query failed')
            mk=input('enter spectral type of %s: \n'%(target))
    if len(mk)==0:
        mk='K3'
    
    ''' find closest spectral type template star from available data'''
    stand_sp_num=[]
    for stand_star in listdir_fullpath(standards_dir):
        stand_mk=stand_star.split('/')[-1]
        if mk[0].lower() == stand_mk[0].lower():
            stand_sp_num.append(float(stand_mk[1]))
            #print (stand_star)
    try:
        stand_spec_num_closest=closest(stand_sp_num,float(mk[1]))#check for if not a number
    except:
        stand_spec_num_closest=closest(stand_sp_num,0)
    stype=mk[0].upper()+str(stand_spec_num_closest)[0]
    print ('Closest standard spectral type from templates is:' ,stype)
    
    '''get list of file names for given data directory and target'''
    #data_fits_files=get_files(os.path.join(data_dir,target),'.fits')
    standard_fits_files=get_files(os.path.join(standards_dir,stype),'.fits')
    
    
    '''find radial velocity of template star'''
    standard_star=standard_fits_files[0].split('/')[-2]
    try:
        simbad_table2=customSimbad.query_object(standard_star)
        simbad_table2.colnames
        st_rv=simbad_table2['RVZ_RADVEL'][0]
        st_vsini=simbad_table2['ROT_Vsini'][0]
        print('radial velocity of template star is:', st_rv)
        print('vsini of template star is:', st_vsini)
    except:
        print('!! simbad query failed !!')
        st_rv=input('>< >< enter RV of template star: \n')
    
    if simbad_out==False:
        return standard_fits_files,mk,stype,st_rv
    else:
        return standard_fits_files,mk,stype,st_rv,simbad_table

def spec_HST(flux_filename):
    hdu = fits.open(flux_filename)
    hdr = hdu[0].header
    hdr1 = hdu[1].header
    
    wave=hdu[1].data['WAVELENGTH'][0]
    flux=hdu[1].data['FLUX'][0]
    error=hdu[1].data['ERROR'][0]
    
    target=hdr['TARGNAME']
    instrume=hdr['INSTRUME']
    start_obs=hdr1['DATE-OBS']+'T'+hdr1['TIME-OBS']
    MJD_start_obs=hdr1['EXPSTART']
    
    hdu.close()
    wave=np.array(wave)
    flux=np.array(flux)
    error=np.array(error)

    info = [target,start_obs,MJD_start_obs,instrume,min(wave)/10,max(wave)/10,'N/A','N/A'] 

    return info,wave,flux,error      
    
def spec_XMM(flux_filename):
    hdul = fits.open(flux_filename)
    data = hdul[1].data
    columns = hdul[1].columns
    
    wave = []
    flux = []
    error = []
    for record in data:
        a = record[0]
        f = record[1]
        e = record[2]
        wave.append(float(a))
        if np.isnan(f):
            flux.append(0.0)
        else:
            flux.append(float(f))
        if np.isnan(e):
            error.append(0.0)
        else:
            error.append(float(e))
    hdul.close()
    wave=np.array(wave)
    flux=np.array(flux)
    error=np.array(error)
    try:
        target=flux_filename.split('/')[-3]
    except:
        target='XMM'
    start_obs='2000.0' #header['DATE-OBS']
    MJD_start_obs=2000.0 #header['MJD-OBS']
    instrume = 'XMM' #header['INSTRUME']  # Name of the instrument
    info = [target,start_obs,MJD_start_obs,instrume,min(wave)/10,max(wave)/10,'N/A','N/A'] 

    return info,wave,flux,error  
    
def spec_ROTFIT(file):
    hdu=fits.open(file)
    
    flux0=hdu[0].data[0]
    flux_corr=hdu[0].data[2]
    if size(flux0)==1:
        flux0=hdu[0].data
        flux_corr=hdu[0].data
    header=hdu[0].header
    err=[]
    n_wave = header['NAXIS1']
    crval1 = header['CRVAL1']
    cdelt1 = header['CDELT1']
    index = np.arange(n_wave, dtype=np.float64)
    wave = crval1 + index*cdelt1
    
    try:
        target=header['OBJECT']    #object target
        start_obs='2000.0' #header['DATE-OBS']
        MJD_start_obs=2000.0 #header['MJD-OBS']
        instrume = 'ROTFIT' #header['INSTRUME']  # Name of the instrument
    except:
        print=('ROTFIT some of the mandatory keywords were not found in primary header unit')
        #print('filename = %s   NOT COMPLIANT' % file)
        return
    
    info = [target,start_obs,MJD_start_obs,instrume,min(wave)/10,max(wave)/10,'N/A','N/A'] 

    return info,wave,flux_corr,err
    
def spec_readspec(file):
    #from CFM
    wave=[]
    flux=[]
    hdr=[]
    err=[]
    split = file.split('.')
    exten = split[-1]
    if (exten == 'fits') or (exten == 'fit'):
        hdu = fits.open(file)
        hdr = hdu[0].header
        if 'crpix1' in hdu[0].header:
            flux = hdu[0].data
            wave = readlambda(flux,hdu)
            if min(wave) > 1000:
                wave=wave / 10 
            try:
                target=hdr['OBJECT']    #object target
                start_obs=hdr['DATE-OBS']
                MJD_start_obs=hdr['MJD-OBS']
                instrume = hdr['INSTRUME']  # Name of the instrument
            except:
                print=('spec_readspec() some of the mandatory keywords were not found in primary header unit')
                print('filename = %s   NOT COMPLIANT' % file)
                return
            try:
                bary_corr=hdr['HIERARCH ESO QC VRAD BARYCOR']
            except:
                try:
                    bary_corr=hdr['HIERARCH ESO QC BERV']
                except:
                    bary_corr=0
                
        elif hdu[1].columns[0] is not None:
            scidata = hdu[1].data
            wave = scidata[0][0]
            flux = scidata[0][1]
            err = scidata[0][2]
            if min(wave) > 1000:
                wave=wave / 10 
            try:
                target=hdr['OBJECT']    #object target
                MJD_start_obs=hdr['MJD-OBS']
                start_obs=Time(MJD_start_obs, format='mjd').isot
                instrume = hdr['INSTRUME']  # Name of the instrument
            except:
                print=('spec_readspec() some of the mandatory keywords were not found in primary header unit')
                print('filename = %s   NOT COMPLIANT' % file)
                return
            try:
                bary_corr=hdr['HIERARCH ESO QC VRAD BARYCOR']
            except:
                try:
                    bary_corr=hdr['HIERARCH ESO QC BERV']
                except:
                    bary_corr=0

        else:
            print('!!!	Wavelength keyword not found in FITS HEADER 	!!!')
            return
    else:
        print("Not yet supported!")
        return
        #readcol, file, wave, lambda
    hdu.close()	
    #info=['target,obs_data,mjd_date,instrument,wavemin,wavemax,respwr,snr']
    info = [target,start_obs,MJD_start_obs,instrume,min(wave),max(wave),bary_corr,'N/A'] 

    return info,wave,flux,err


def readlambda(spec, hdu_sp):
    #from CFM
	crpix1 = hdu_sp[0].header['crpix1']
    #/ value of ref pixel
	crval1 = hdu_sp[0].header['crval1']
    #/ delta per pixel
	if 'cd1_1' in hdu_sp[0].header:
		cd1_1 = hdu_sp[0].header['cd1_1']
	#cd1_1 is sometimes called cdelt1.
	else:
		cd1_1 = hdu_sp[0].header['cdelt1']
	if cd1_1 == 0:
		print("NOT WORKING")
		return
	n_lambda = len(spec)
	wave = np.zeros(n_lambda)
	for l  in range(n_lambda):
		wave[l] = (l+1.0-crpix1)*cd1_1+crval1
    #Use pixel number starting with 0 if no lambda information is found.
	if (np.min(wave)+np.max(wave) == 0.0):
		print('No lambda information found: used pixel number starting with 0')
		for l  in range(n_lambda):
			wave[l] = l	
	return wave

def readspec_espresso_air(file,err_out='NO',hdr_out='NO'):
    #from CFM
    # USAGE:
    # wl,fl[,err,hdr] = readspec_espresso(filename[,err_out='YES',hdr_out='YES'])
    hdu = fits.open(file)
    hdr = hdu[0].header

    try:
        target=hdr['OBJECT']    #object target
        start_obs=hdr['DATE-OBS']
        MJD_start_obs=hdr['MJD-OBS']
        instrume = hdr['INSTRUME']  # Name of the instrument
    except:
        print=('spec_readspec() some of the mandatory keywords were not found in primary header unit')
        print('filename = %s   NOT COMPLIANT' % file)
        return
    try:
        bary_corr=hdr['HIERARCH ESO QC BERV']
    except:
        bary_corr=0
    
    if 'FLUX' in hdu[1].columns.names:
        # in the *FINAL* product of the pipeline, FLUX is flux calibrated and sky subtracted
        flux = np.array(hdu[1].data['FLUX'],dtype=np.float64)
    elif 'flux_cal' in hdu[1].columns.names:
        flux = np.array(hdu[1].data['flux_cal'],dtype=np.float64)

    wave = np.array(hdu[1].data['WAVE_AIR'],dtype=np.float64)
    if 'ERR' in hdu[1].columns.names:
        err = np.array(hdu[1].data['ERR'],dtype=np.float64)
    elif 'error_cal' in hdu[1].columns.names:
        err = np.array(hdu[1].data['error_cal'],dtype=np.float64)

    if len(flux) == 1:
        # in this case all the wavelengths are in one line, and you have to get the array in the array
        flux = flux[0]
        wave = wave[0]
        err = err[0]
    hdu.close()
    
    if min(wave) > 1000:
        wave=wave / 10 

    info = [target,start_obs,MJD_start_obs,instrume,min(wave),max(wave),bary_corr,'N/A'] 


    if err_out=='NO' and hdr_out=='NO':
        return info,wave,flux,[]
    elif err_out!='NO' and hdr_out=='NO':
        return info,wave,flux,err
    elif err_out=='NO' and hdr_out!='NO':
        return info,wave,flux,hdr
    else:
        return info,wave,flux,err,hdr


def read_fits_files(filename,verbose=False):
    try:
        info,wave,flux,err=read_ESO_fits_spec(filename)
        if verbose==True:
            print('Using ESO read in script')
    except:
        try:
            info,wave,flux,err=read_ESP_fits_spec(filename)
            if verbose==True:
                print('Using ESP read in script')
        except:
            try:
                info,wave,flux,err=readspec_espresso_air(filename,err_out='yes')
                if verbose==True:
                    print('using readspec_espresso_air()')
            except:
                try:
                    info,wave,flux,err=spec_readspec(filename)
                    if verbose==True:
                        print('using spec_readspec()')    
                except:
                    try:
                        info,wave,flux,err=spec_ROTFIT(filename)
                        if verbose==True:
                            print('using spec_ROTFIT()') 
                    except:
                        try:
                            info,wave,flux,err=spec_XMM(filename)
                            if verbose==True:
                                print('using spec_XMM()') 
                        except:
                            try:
                                info,wave,flux,err=spec_HST(filename)
                                if verbose==True:
                                    print('using spec_HST()')                                 
                            except:
                                print('files cannot be read by any of the read in functions')  #files dont work
                                return [],[],[],[]
    info.append(filename)

    #check that wavelength is in Angstrom not nm -this will need to be updated maybe from fits headuer unit?
    #if max(wave) < 1000:
    #    wave=wave * 10 
    
    return info,wave,flux,err

def organise_fits_files(dir_of_files,output_dir):
    files=get_files(dir_of_files,'.fits')
    for f in files:
        try:
            info,wave,flux,err=read_fits_files(f)
            output_dir_file=os.path.join(output_dir,info[0],info[3])
            ensure_dir(output_dir_file)
            try:
                shutil.copy2(f,output_dir_file)
            except shutil.SameFileError:
                pass
        except:
            print('fits file: %s cannot be read'%(f))
            pass



def get_instrument_date_details(data_fits_files,instr='any',all_inst=False,qgrid=False,start_date='1900',end_date='2100'):
    '''
    get date index and info of FITS files, check what data avail. and select instr. 
    optional to specify date range here, get wavelength values for specified range & instrument
    excludes certain bad telluric ranges

    Parameters
    ----------
    data_fits_files : list
        list from get_files() of target star fits files.
    instr : str, optional
        instrument name, if not specified will promt user to input. The default is 'any'.
    start_date : str, optional
        start date of observations to include, format 'yyyy-mm-dd' or subset eg '2010' or '2010-05'. The default is '1900'.
    end_date : str, optional
        end date of observations to include, as above. The default is '2100'.

    Returns
    -------
    data_dates_range : pandas dataframe
        dataframe containing columns 'target','utc','mjd','inst','wmin','wmax','bary','snr','file' for specified instrument and date range.
    instrument : str
        name of instrument.
    w0 : array
        1d array of wavelength values for use in rest of calculations, 
        specified for instrument in function or taken from the FITS files directly.
        also includes any exclusions for telluric and bad insstrument ranges.

    '''

    '''need to update this for ESP files, they do not havee res and snr'''
    info_columns=['target','utc','mjd','inst','wmin','wmax','bary','snr','file']#these are the columns from ESO_fits_spec and filename
    info_list=[]
    for f in data_fits_files:
        info,wave,flux,err=read_fits_files(f)
        
        if info[3]=='SHOOT':#for custom reduced data with different instrument name
            info[3]='XSHOOTER'
        
        if instr=='any':
            info_list.append(info)               
        else:
            if info[3]==instr:
                info_list.append(info)
        
        dates_df=pd.DataFrame(info_list,columns=info_columns).sort_values('mjd').reset_index(drop=True)
        
    ordered_dates=dates_df[dates_df.utc.between(start_date,end_date)]
    
    if qgrid==True or all_inst==True:
        data_dates_range=ordered_dates
        instrument='all'
        if qgrid==True:
            instrument='any'
            w0=[]
    
    if qgrid==False and all_inst==False:
        if len(np.unique(ordered_dates.inst)) > 1:
            insts,counts=np.unique(ordered_dates.inst,return_counts=True)
            print('there are data from these instruments:%s, resp. counts: %s'%(insts,counts))       
            selection = input('select instrument...:')

            print('Selected: %s'%(selection))

            ordered_dates=ordered_dates[ordered_dates.inst==selection]    

        #data_dates_range=get_obs_dates(data_fits_files,instr=instr_select,start_date=start_date,end_date=end_date)
        data_dates_range=ordered_dates
        instrument=data_dates_range.inst.values[0]
        print('-data available from selected instrument:%s -'%(instrument))
      
    
    ''' instrument details, create standard wavelength list'''
    w_step=0.01#step size of 0.01 allows line finder to get narrow lines    
    if instrument=='FEROS':
        w_min=data_dates_range.wmin.values[0]*10
        w_max=data_dates_range.wmax.values[0]*10
        w0=np.arange(w_min,w_max,w_step) #set standard range of wavelength with set steps
        instr_mask=((w0 < 8533.7) | (w0 > 8541.6)) & ((w0 < 8861.5) | (w0 > 8875.8))            
        w0=w0[instr_mask]
    elif instrument=='HARPS':
        w_min=data_dates_range.wmin.values[0]*10
        w_max=data_dates_range.wmax.values[0]*10
        w_step=0.05
        w0=np.arange(w_min,w_max,w_step) #set standard range of wavelength with set steps
        instr_mask=((w0 < 5303) | (w0 > 5338))            
        w0=w0[instr_mask]    
    elif instrument=='ESPRESSO':
        w_min=data_dates_range.wmin.values[0]*10
        w_max=data_dates_range.wmax.values[0]*10
        w_step=0.01
        w0=np.arange(w_min,w_max,w_step)    
    elif instrument=='ESPaDOnS':
        w_min=3700
        w_max=10450  
        w0=np.arange(w_min,w_max,w_step) #set standard range of wavelength with set steps
    elif instrument=='UVES':
        w_step=0.01
        if len(np.unique(round(data_dates_range.wmin))) >1:
            w_min_select = int(input('more than one spectral arm exisits.. select arm from wmin \n possible wmin values for UVES data: %s \n select wmin...:'%(np.unique(round(data_dates_range.wmin)))))
            data_dates_range=data_dates_range[np.isclose(data_dates_range.wmin.values, w_min_select, atol=5)]
        w_min=data_dates_range.wmin.values[0]*10 +100
        w_max=data_dates_range.wmax.values[0]*10 -100
        if w_max > 11000:
            w_min=w_min / 10 
            w_max=w_max / 10
        w0=np.arange(w_min,w_max,w_step) #set standard range of wavelength with set steps
    elif instrument == 'XSHOOTER':
        w_step=0.1
        if len(np.unique(round(data_dates_range.wmin))) >1:
            w_min_select = int(input('more than one spectral arm exisits.. select arm from wmin \n possible wmin values for %s data: %s \n select wmin...:'%(instrument,np.unique(round(data_dates_range.wmin)))))
            data_dates_range=data_dates_range[np.isclose(data_dates_range.wmin.values, w_min_select, atol=5)]
        
        # obs_interval = TimeDelta(600.0, format='sec')
        # t1=Time(data_dates_range.utc.iloc[0])
        # obs_seq=data_dates_range[data_dates_range.utc.between(t1,t1+obs_interval)]
        # data_dates_range=obs_seq
        
        w_min=data_dates_range.wmin.values[0]*10
        w_max=data_dates_range.wmax.values[0]*10
        w0=np.arange(w_min,w_max,w_step) #set standard range of wavelength with set steps
        instr_mask= (w0 > 3050) 
        w0=w0[instr_mask]
        if np.isclose(w_min,5340,atol=10): #remove bad part of mid arm, leaving this range in blue arm
            instr_mask=((w0 < 5336) | (w0 > 5550))   
            w0=w0[instr_mask]
    elif instrument=='all': 
        #this does not really work for creating an average data frame of all the instruments
        #it works if you specify a common smaller wavelength range for all the instruments
        print('selecting >1 instrument only works for reduced range within all coverages')
        print('ensure correct selection is made from qgrid using wmin and wmax sliders')
        print('suggested wmin and wmax: %.0f , %.0f'%(max(data_dates_range.wmin)*10, min(data_dates_range.wmax)*10))       
        w_step=0.01
        w_min, w_max = [float(x) for x in input('Enter min and max wavelength for all instruments in Angstroms e.g. 5000 5500 : \n').split()] 
        w0=np.arange(w_min,w_max,w_step)
    elif instrument=='ROTFIT':
        w_min=data_dates_range.wmin.values[0]*10
        w_max=data_dates_range.wmax.values[0]*10
        w_step=0.1
        w0=np.arange(w_min,w_max,w_step)        
    elif instrument=='XMM':
        w_min=data_dates_range.wmin.values[0]*10
        w_max=data_dates_range.wmax.values[0]*10
        w_step=0.01
        w0=np.arange(w_min,w_max,w_step) 
    elif instrument=='COS' or instrument=='STIS':
        if len(np.unique(round(data_dates_range.wmin))) >1:
            w_min_select = int(input('more than one spectral arm exisits.. select arm from wmin \n possible wmin values for %s data: %s \n select wmin...:'%(instrument,np.unique(round(data_dates_range.wmin)))))
            data_dates_range=data_dates_range[np.isclose(data_dates_range.wmin.values, w_min_select, atol=5)]
        
        w_min=data_dates_range.wmin.values[0]*10
        w_max=data_dates_range.wmax.values[0]*10
        w_step=0.1
        w0=np.arange(w_min,w_max,w_step)   
    elif 'CAFOS' in instrument:
        w_min=data_dates_range.wmin.values[0]*10
        w_max=data_dates_range.wmax.values[0]*10
        w_step=0.1
        w0=np.arange(w_min,w_max,w_step) 
    else:
        if instrument != 'any':
            print('WARNING: instrument ', instrument, 'not tested but read in by read in scripts')
            w_min=data_dates_range.wmin.values[0]*10
            w_max=data_dates_range.wmax.values[0]*10
            w_step=0.1
            w0=np.arange(w_min,w_max,w_step)
        
    if qgrid==False:
        '''remove bad telluric range'''
        telu_mask=((w0 < 7592) | (w0 > 7690)) & ((w0 < 6872) | (w0 > 6920))            
        w0=w0[telu_mask]
    
    return data_dates_range,instrument,w0

def wl_excluder(w0,df_av,df_av_norm=[],w0_cut=[]):
    '''
    user specified wavelength exlusiion to remove given section of the wavelength.
    will replace the w0 wavelength array and average spec dataframe df_av used for all following calculations 

    '''
    
    
    print('-- Current wavelength range, not including instr. and telluric exclusions, is \n %s - %s Angstroms'%(round(min(w0)),round(max(w0))))
    if len(w0_cut)>0:
        print('-- From function argument, will exclude %s'%(w0_cut))
        select_w0_cut = input('Specify further wavelength exclusion? y/n? \n:')
    else:
        select_w0_cut = input('Specify wavelength exclusion? y/n? \n:')
    while select_w0_cut[0].lower()=='y':
        # w0_cut.append( [float(x) for x in input('Enter wavelength exclusion min and max e.g. 5201.8 5205.2 : \n').split()] )
        try:
            w0_cut_min, w0_cut_max = [float(x) for x in input('Enter wavelength exclusion min and max e.g. 5201.8 5205.2 : \n').split()] 
            w0_cut.append([w0_cut_min,w0_cut_max])
            select_w0_cut = input('Specify another wavelength exclusion? y/n? \n:')
        except:
            print('!! ERROR !! must be two integers or floats specified, min and max')
            pass
    
    ''' apply mask to w0 for each of the user specified zones'''
    if len(w0_cut) > 0:
        for n in range(len(w0_cut)):
            user_mask=((w0 < w0_cut[n][0]) | (w0 > w0_cut[n][1]))
            w0=w0[user_mask]
            df_av=df_av[df_av.wave.isin(w0)]
            if len(df_av_norm)>0:
                df_av_norm=df_av_norm[df_av_norm.wave.isin(w0)]
    if len(df_av_norm)>0:
        return w0,df_av,df_av_norm
    else:
        return w0,df_av





