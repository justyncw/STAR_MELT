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
from scipy.optimize import curve_fit, ridder
from scipy.signal import savgol_filter
from scipy.signal import find_peaks_cwt
from scipy.signal import argrelextrema
from ESO_fits_get_spectra import *
from ESP_fits_get_spectra import *
from utils_data import *
from utils_spec import *
import utils_shared_variables as USH

timenow=time.strftime("%d_%b_%Y_%H:%M", time.gmtime())


# =============================================================================
# constants
# =============================================================================

clight=astropy.constants.c.to('km/s').to_value()
c=astropy.constants.c.to_value()
h=astropy.constants.h.to_value()
k=astropy.constants.k_B.to_value()
m_e=astropy.constants.m_e.to_value()

# =============================================================================
# functions for equations
# =============================================================================
def B_ik(Aki,v,g_k,g_i):
    '''Einstein B coeff'''
    Bik=Aki * (g_k/g_i) * (c**2 / (2*h*(v**3)))
    return Bik

def tau(Bik,g_i,Ei,T,U,N):
    '''optial depth'''
    tau= ((h*c) / (4*np.pi)) * Bik * g_i * ( (np.exp( -(Ei) / (k*T) )) / (U) ) * (N*1e6)
    return tau

def r_ws(A_w,wl_w,tau_w,A_s,wl_s,tau_s):
    '''ratio of weak to 
    strong line'''
    r_ws= ( (A_w * wl_s) / (A_s * wl_w) ) * ( (1-(np.exp(-tau_w))) / (tau_w) ) * ( (tau_s) / (1 - (np.exp(-tau_s))) )
    return r_ws

def boltz(g_i,g_o,Ei,T):
    '''ratio between transitions of same ion, levels from Boltzmann statistics'''
    #think this should be Ek-Ei, check
    n_ratio = ( g_i / g_o) *  (np.exp( -( (Ei) / 6.242e+18) /(k*T)) )
    return n_ratio


def saha_eq(log_n,T,X_i):
    '''saha equation for density ratio of ionisation states for given T and n_e'''
    n_i_n_j= (1/((10**log_n)*1e6)) * ((( (2*np.pi) * m_e * k * T) / (h**2))**(3./2.) ) * ( (2*U1(T)) / (U(T))) * (np.exp( -((X_i)/ 6.242e+18) / (k*T) ))
    return n_i_n_j

# =============================================================================
# example testing values
# =============================================================================
Aki_w=2200000
wl_w=4890.7548 #e-10
v_w=c/wl_w
g_k_w=5
g_i_w=5
Ei_w=2.87550350 #/6.242e+18
iflux_w=7.1
Aki_s=24400000
wl_s=4871.3179 #e-10
v_s=c/wl_s
g_k_s=5
g_i_s=7
Ei_s=2.86539122 #/6.242e+18
iflux_s=52.1
T=1e4
T_list=np.arange(4e3,5e4,500) #defined in function
log_n=np.arange(8,22,0.2) #this is going to be conventional to give in cm^-3 so need to convert to m^-3 for SI calculations
Te=k*T*6.242e+18

# =============================================================================
# read in file with partition functions for T range 4e3 - 4e4, step=600
# resample values in table to full range of T values via extrapolation
# =============================================================================
U_T_list= pd.read_csv('Line_Resources/U_T_list.csv',index_col=False,comment='#',skip_blank_lines=True)
resample_U_Fe_I=interp1d(U_T_list.dropna()['T'] ,U_T_list.dropna()['U_Fe_I'] ,fill_value='extrapolate')
resample_U_Fe_II=interp1d(U_T_list.dropna()['T'] ,U_T_list.dropna()['U_Fe_II'] ,fill_value='extrapolate')
U_Fe_I=resample_U_Fe_I(T_list)
U_Fe_II=resample_U_Fe_II(T_list)
U_T_list_resampled=pd.concat((U_T_list['T'],U_T_list['Te'],pd.Series(U_Fe_I,name='U_Fe_I'),pd.Series(U_Fe_II,name='U_Fe_II')),axis=1)
#can also the resample_U_Fe_I with T as an arg
U=resample_U_Fe_I
U1=resample_U_Fe_II

#U_list=U_Fe_II

#can create lookup table for these values so that element can be found in col or row name and correct value selected automatically
X_i_Fe=7.9024 #ionisation potential of iron II in eV
X_i_H=13.59844
X_i_He=24.58741
X_I_Ca=6.11316
X_i_Ti=6.7462

#boltz(g_i_w,g_k_w,Ei_w,1e4)
#saha_eq(18,1e4,X_i_Fe)

def sobolev(element,sp_num,
            Aki_s,wl_s,g_k_s,g_i_s,Ei_s,iflux_s,
            Aki_w,wl_w,g_k_w,g_i_w,Ei_w,iflux_w,
            N_range=[6,22],T_range=[4e3,5e4],
            target='temp',output=True,savefig=False):
    '''
    Function to calculate ratio of lines from common upper energy levels and plot results,
    output is ratio versus T for range of N and ratio versus N for rage of T

    '''
    T_list=np.arange(T_range[0],T_range[1],500) 
    log_n=np.arange(N_range[0],N_range[1],0.2)
    U=resample_U_Fe_I(T_list)
    U1=resample_U_Fe_II(T_list)

    
    wl_s=wl_s * 1e-10
    wl_w=wl_w * 1e-10
    Ei_s=Ei_s / 6.242e+18
    Ei_w=Ei_w / 6.242e+18
    
    if sp_num == 2:
        U_list=U
    elif sp_num == 1:
        U_list=U1
    else:
        print('Error, can only compute Fe I or II until more partition functions obtained')
    
    
    iflux_ratio=iflux_w/iflux_s
    #specify which partition fn to use based on input element
    # try:
    #     ele=element.split()[0]
    #     sp_num=element.split()[1]
    #     if ele == 'Fe' and sp_num == 'I': U_list=U_Fe_I 
    #     elif ele == 'Fe' and sp_num == 'II': U_list= U_Fe_II
    # except:
    #     print('element string unkown, setting U list to Fe I')
    #     U_list=U_Fe_I

    r_ws_T=[]
    for T,U in zip(T_list,U_list): #for each T and U pair, plot range of N values
        B_s=B_ik(Aki_s,c/wl_s,g_k_s,g_i_s) #constants for calculation
        B_w=B_ik(Aki_w,c/wl_w,g_k_w,g_i_w) 
        
        tau_s=tau(B_s,g_i_s,Ei_s,T,U,10**log_n) #T, U, n are iterable
        tau_w=tau(B_w,g_i_w,Ei_w,T,U,10**log_n)
        
        r=r_ws(Aki_w,wl_w,tau_w,Aki_s,wl_s,tau_s)
        r_ws_T.append(r)
    #plot(T_list,r_ws_T)

    r_ws_N=[]
    for N in log_n: #for each N, plot range of T (U) values
        B_s=B_ik(Aki_s,c/wl_s,g_k_s,g_i_s) #constants for calculation
        B_w=B_ik(Aki_w,c/wl_w,g_k_w,g_i_w) 
        
        tau_s=tau(B_s,g_i_s,Ei_s,T_list,U_list,10**N) #T, U, n are iterable
        tau_w=tau(B_w,g_i_w,Ei_w,T_list,U_list,10**N)
        
        r=r_ws(Aki_w,wl_w,tau_w,Aki_s,wl_s,tau_s)
        r_ws_N.append(r)
    #plot(log_n,r_ws_N)
    
    #find the values of T and N for the obs ratio
    ni_av=[]
    for T in range(size(T_list)):
        f=interp1d(log_n,r_ws_T[T]-iflux_ratio,bounds_error=False, fill_value="extrapolate")
        try:
            n_av=ridder(f,min(log_n),max(log_n))
        except(ValueError):
            n_av=nan
        ni_av.append(n_av)
    obs_ratio_prop=np.array([T_list,ni_av])
    
   
    if output==True or savefig==True:
        ioff()
        fig, ax = subplots(1, 2,figsize=(11,5))#,gridspec_kw={'wspace':0})
        # fig.suptitle('Sobolev')
    
        ax[0].plot(T_list,r_ws_T)
        ax[0].semilogx()
        ax[0].set_xlabel('T(K)')
        ax[0].set_ylabel(f'{element}{sp_num:.0f} - {wl_w * 1e10 :.1f} / {wl_s * 1e10 :.1f}')
        ax[0].axhline(y=iflux_ratio,c="blue",linewidth=0.5,zorder=0)

        
        ax[1].plot(log_n,r_ws_N)
        ax[1].set_xlabel(r'n dl/dv (cm$^{-3}$ s)')
        ax[1].set_ylabel(f'{element}{sp_num:.0f} - {wl_w * 1e10 :.1f} / {wl_s * 1e10 :.1f}')
        ax[1].axhline(y=iflux_ratio,c="blue",linewidth=0.5,zorder=0)
        figtext(0.5, 0.95,"Sobolev LVG approx", wrap=True,
            horizontalalignment='center', fontsize=12)

    if output==True:
        show()
    if savefig==True:
        #output dir
        dirname=os.path.join('output_plots', 'sobolev_'+target+'_'+timenow)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fig.savefig(os.path.join(dirname,element.split()[0]+element.split()[1]+'_'+str(int(wl_w * 1e10))+'_'+str(int(wl_s * 1e10))+'.pdf'))
        print('saving file',os.path.join(dirname,element.split()[0]+element.split()[1]+'_'+str(int(wl_w * 1e10))+'_'+str(int(wl_s * 1e10))+'.pdf'))
        if output==False:
            close()
    ion()
    return obs_ratio_prop
    
def sobolev_by_element(em_line_date_results_common_Ek,target,N_range=[6,22],T_range=[4e3,5e4],output=True,savefig=False):
    #em_lines_Ek_element=em_line_date_results_common_Ek[em_line_date_results_common_Ek.element==element]
    obs_ratio_fits=[]
    line_labels=[]
    for Ek in unique(em_line_date_results_common_Ek.Ek):
        Ek_lines=em_line_date_results_common_Ek[em_line_date_results_common_Ek.Ek == Ek]
        element=Ek_lines.element.any()
        sp_num=Ek_lines.sp_num.values[0]
        if sp_num==1:
            sp='I'
        else:
            sp='II'
        weak=Ek_lines.loc[Ek_lines.Aki.idxmin()]
        strong=Ek_lines.loc[Ek_lines.Aki.idxmax()]
        #print(weak.int_flux / strong.int_flux) #if < 1?
        if weak.int_flux/strong.int_flux < 3 and weak.Aki!=strong.Aki:
            obs_ratio_prop=sobolev(element,sp_num,strong.Aki,strong.obs_wl_air,strong.g_k,strong.g_i,float(strong.Ei),strong.int_flux,
                        weak.Aki,weak.obs_wl_air,weak.g_k,weak.g_i,float(weak.Ei),weak.int_flux,
                        N_range=N_range,T_range=T_range,target=target,output=output,savefig=savefig)
            obs_ratio_fits.append(obs_ratio_prop)
            line_labels.append('%s%s %s/%s'%(element,sp,weak.obs_wl_air,strong.obs_wl_air))
    
    if output==True:
        figure(figsize=USH.fig_size_s)
        for i in range(len(obs_ratio_fits)):
            if np.isnan(obs_ratio_fits[i][1]).all()==False:
                plot(obs_ratio_fits[i][0],obs_ratio_fits[i][1],label=line_labels[i])
        xscale('log')
        xlabel('T(K)')
        ylabel(r'n dl/dv (cm$^{-3}$ s)')
        legend(fontsize=12)
        tight_layout()
    
    #return obs_ratio_fits,line_labels
    
            
        

# sobolev('Fe II',
#         Aki_s,wl_s,g_k_s,g_i_s,Ei_s,iflux_s,
#         Aki_w,wl_w,g_k_w,g_i_w,Ei_w,iflux_w)