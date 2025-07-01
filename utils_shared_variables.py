#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:21:51 2020
@author: jcampbellwhite001
"""
import time
from matplotlib import *
from matplotlib.pyplot import *
import numpy as np
import pandas as pd
import astropy
from astropy.time import Time
import astropy.units as u
from astropy.stats import sigma_clip
from astroquery.simbad import Simbad
from astropy.timeseries import LombScargle
import numpy.ma as ma
import os
from PyAstronomy import pyasl
from lmfit.models import GaussianModel, LinearModel
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.signal import find_peaks_cwt
from scipy.signal import argrelextrema
from ESO_fits_get_spectra import *
from ESP_fits_get_spectra import *
from utils_data import *
from utils_spec import *

#variables shared between module files
target='USH'
instrument='USH'
radvel=0
vsini=0

inst_res = {
    'ESPRESSO': 140000,
    'HARPS': 115000,
    'ROTFIT':115000,
    'XSHOOTER': 18400,
    'UVES': 60000,
    'PHOENIX': 500000,
}

#figure sizes used in module file
fig_size_s=(6,6)
fig_size_sn=(4.5,5)
fig_size_l=(9,6)
fig_size_n=(9,4)


#Spectral Type = Teff conversion file
try:
    spt_teff=pd.read_csv('Line_resources/HH14_SpT_Teff.csv')
except:
    pass


#Read in line files

lines_file='Line_Resources/JCW_all_lines_2_20kA_100321.csv'
line_table = pd.read_csv(lines_file,delimiter='\t',index_col=False,comment='#',skip_blank_lines=True)
line_table['Ei']=line_table['Ei'].str.replace('[^\d.]', '',regex=True).astype(float)
line_table['Ek']=line_table['Ek'].str.replace('[^\d.]', '',regex=True).astype(float)
line_table.drop(columns=('rm'),inplace=True)
line_table['obs_wl_air']=np.round(line_table['obs_wl_air'],2)
line_table.sort_values(['obs_wl_air','Aki'],ascending=[True,False],inplace=True)#keep the highest Aki for duplicated obs_wl_air entries?
line_table.drop_duplicates('obs_wl_air',inplace=True)
line_table.reset_index(drop=True,inplace=True)

xr_lines_file='Line_Resources/JCW_XR_lines_2_50A.csv'
xr_line_table = pd.read_csv(xr_lines_file,delimiter='\t',index_col=False,comment='#',skip_blank_lines=True)
xr_line_table['Ei']=xr_line_table['Ei'].str.replace('[^\d.]', '',regex=True).astype(float)
xr_line_table['Ek']=xr_line_table['Ek'].str.replace('[^\d.]', '',regex=True).astype(float)
xr_line_table['ritz_wl_vac']=xr_line_table['ritz_wl_vac'].str.replace('[^\d.]', '',regex=True).astype(float)
xr_line_table['ritz_wl_vac']=np.round(xr_line_table['ritz_wl_vac'],3)

xrs_lines_file='Line_Resources/JCW_Stelzer_XR_lines_2_50A.csv'
xrs_line_table = pd.read_csv(xrs_lines_file,delimiter=',',index_col=False,comment='#',skip_blank_lines=True)
#xr_id=pd.Series([18.97,21.61,21.8,22.1,12.13,13.45,13.55,13.699,17.06,24.78,28.79,29.1,29.53,16,18.63,33.74],name='ritz_wl_vac')

sol_lines_file='Line_Resources/JCW_Si4_UV_line.csv'
sol_line_table = pd.read_csv(sol_lines_file,delimiter='\t',index_col=False,comment='#',skip_blank_lines=True)
sol_line_table['ritz_wl_vac']=np.round(sol_line_table['ritz_wl_vac'],3)

prev_obs_file='Line_Resources/previously_observed_lines_120321.csv'
prev_obs=pd.read_csv(prev_obs_file,delimiter=',',index_col=False)
prev_obs_cw21_file='Line_Resources/JCW+21_lines.csv'
prev_obs_cw21=pd.read_csv(prev_obs_cw21_file,delimiter=',',index_col=False)
prev_obs_cw21=prev_obs_cw21[['obs_wl_air','star']]
prev_obs_cw21=prev_obs_cw21.rename(columns={'star':'prev'})
prev_obs_cw21['prev_obs']=prev_obs_cw21['prev'] #check if want to keep previous IDs before cw21...
prev_obs=pd.concat([prev_obs,prev_obs_cw21])
prev_obs_NIST=pd.merge_asof(prev_obs.sort_values('obs_wl_air'),line_table,on='obs_wl_air',direction='nearest')
line_table_prev_obs=pd.concat([prev_obs_NIST,line_table],sort=False)
line_table_prev_obs.sort_values(['obs_wl_air','prev_obs'],inplace=True)
line_table_prev_obs.drop_duplicates(['obs_wl_air'],inplace=True)
line_table_prev_obs.reset_index(drop=True,inplace=True)
line_table_prev_obs = line_table_prev_obs[ [ col for col in line_table_prev_obs.columns if col != 'prev' ] + ['prev'] ]


#photosphere removal data...
def load_phot_templates(template_dir):
    print('reading STAR-MELT photpsheric template stars...')
    data_dir_cl3=os.path.join(template_dir,'PEN_ESP_ClassIII')
    data_fits_files_cl3=get_files(data_dir_cl3,'.fits','.FTZ')
    cl3_rvs=pd.read_csv(os.path.join(data_dir_cl3,'ESP_Cl3_RVs_240509.csv'))
    data_dates_range_cl3,instrument,w0_cl3=get_instrument_date_details(data_fits_files_cl3,qgrid=True)
    data_dates_range_cl3['templ']='Cl3'
    data_dir_ms=os.path.join(template_dir,'templates_HARPS_230629')
    data_fits_files_ms=get_files(data_dir_ms,'.fits','.FTZ')
    data_dates_range_ms,instrument,w0_ms=get_instrument_date_details(data_fits_files_ms,qgrid=True)
    data_dates_range_ms['templ']='MS'
    data_dates_range_templ=pd.concat((data_dates_range_cl3,data_dates_range_ms))
    
    spts=[]
    rvs=[]
    rvs_errs=[]
    for target in data_dates_range_templ.target:
        if target.startswith('V '):
            target=target.replace('V ', '', 1)
        elif target == 'HD816384':
            target='HIP116384'
        try:
            simbad=customSimbad.query_object(target)
            radvel_templ=simbad['RVZ_RADVEL'][0]
            radvel_templ_err=simbad['RVZ_ERROR'][0]
            mk_templ=simbad['SP_TYPE'][0]
            if mk_templ.startswith('d'):
                mk_templ = mk_templ[1:]#remove 'd' from start of sp_t
            if len(mk_templ) >= 3 and mk_templ[2] == '.':
                mk_templ = mk_templ[:4]
            else:
                mk_templ = mk_templ[:2]
        except:
            mk_templ='K9'
            pass  
        if target in cl3_rvs.star.values:
            mk_templ=cl3_rvs.SpT[cl3_rvs['star']==target].values[0]
            radvel_templ=cl3_rvs.rv[cl3_rvs['star']==target].values[0]
            radvel_templ_err=cl3_rvs.rv_err[cl3_rvs['star']==target].values[0]

        spts.append(mk_templ)
        rvs.append(radvel_templ)
        rvs_errs.append(radvel_templ_err)
        #print('%s SpT:  ' %(target) ,mk)
    data_dates_range_templ['sp_t']=spts
    data_dates_range_templ['RV']=rvs
    data_dates_range_templ['RV_err']=rvs_errs
    cols_to_move = ['target','sp_t','templ']
    data_dates_range_templ= data_dates_range_templ[ cols_to_move + [ col for col in data_dates_range_templ.columns if col not in cols_to_move ] ]
    print('read ',len(data_dates_range_templ), '.fits template files')
    return data_dates_range_templ



