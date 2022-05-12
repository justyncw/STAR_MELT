#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 19:05:15 2019
@author: jcampbellwhite001
"""
import time
import pandas as pd
import numpy as np
import astropy.constants
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from ESO_fits_get_spectra import *
from ESP_fits_get_spectra import *
from utils_data import *
from PyAstronomy import pyasl
from matplotlib import *
from matplotlib.pyplot import *
from astropy.stats import sigma_clip
from astropy.timeseries import LombScargle
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from lmfit.models import GaussianModel, LinearModel
import utils_shared_variables as USH

clight=astropy.constants.c.to('km/s').to_value()
timenow=time.strftime("%d_%b_%Y_%H:%M", time.gmtime())

line_table=USH.line_table
line_table=USH.line_table_prev_obs


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_av_spec_simple(df_spec,w0,columns=None,norm=False,output=False,plot_av=True):
    '''
    Create dataframe of observations for list of df_spec dataframes with 'wave' and 'f0' columns

    Parameters
    ----------
    df_spec : list of dataframes
        list containing dataframes with wave and f0 coluumns.
    w0 : array
        wavelength array.
    columns : list, optional
        list of column names for the flux columns. The default is None.
    norm : bool, optional
        normalise the flux values by dividing by the median flux. The default is False.
    output : bool, optional
        plot flux data frame. The default is False.
    plot_av : bool, optional
        plot average spectra. The default is True.

    Returns
    -------
    df_av : dataframe
        dataframe of wave, flux1,...,fluxi,av_flux,med_flux,std_flux .

    '''
        
    '''create empty data frames for the values and col names'''
    df_av=pd.DataFrame({'wave': w0})
    df_av_col_names=['wave']
    if output == True:
        ioff()
        fig=figure(figsize=USH.fig_size_s)
    '''create average data framne of flux values'''
    i=0
    for f in df_spec:
        data_flux=f.f0
        data_wave=f.wave
        if min(data_wave) < 1000: #convert from nm to angstroms
            data_wave=data_wave * 10 

        resample=interp1d(data_wave,data_flux,fill_value='extrapolate')
        f0_data=resample(w0)

        if norm == True:
            f0_data=(f0_data)/median(f0_data)
            
        #if min(data_wave) < min(w0) < max(data_wave):
        df_av=pd.concat([df_av,pd.Series(f0_data)],axis=1)#this makes the full data frame
        df_av_col_names.append(i)#this names the flux columns
        i+=1
        #df_av_col_names.append((data_info[2]))#this names the flux columns by mjd        
        if output == True or savefig == True:
            plot(w0,f0_data)#,label=data_info[0]+' '+data_info[1]+' '+data_info[3])
    if columns != None:
        df_av.columns=columns
    else:
        df_av.columns=df_av_col_names
    av_flux=df_av.iloc[:,1:len(df_av.columns)].mean(axis=1).rename('av_flux')
    med_flux=df_av.iloc[:,1:len(df_av.columns)].median(axis=1).rename('med_flux')
    std_flux=df_av.iloc[:,1:len(df_av.columns)].std(axis=1).rename('std_flux')
    df_av=pd.concat([df_av,av_flux,med_flux,std_flux],axis=1)
    if output == True:
        if plot_av==True:
            plot(w0,df_av.av_flux,'k',linewidth=2,label='average')
            plot(w0,df_av.med_flux,'b--',linewidth=2,label='median')   
        #legend(loc='upper left', fontsize=7, numpoints=1)
        ylabel('Flux')
        xlabel('Wavelength [Angstroms]')
        #if instr == 'FEROS':
        #    ylim(-0.01,0.1)
        if output==True:
            show()
    ion()
    return df_av


def get_av_spec(data_dates_range,w0,label='mjd',norm=False,output=False,plot_av=True,savefig=False):
    
    '''
    function to return data frame of all fluxes, av flux and std for given fits file list
    and stanard wavelength range w0 from get_instrument_date_details()

    Parameters
    ----------
    data_dates_range : data frame
        output of get_instrument_date_details() providing info and filenames to create
        average dataframe from.
    w0 : array
        output of get_instrument_date_details() giving specified wavelength range to
        interpolate flux values between.
    label : str, optional
        how to label the flux columns, either 'mjd' or 'utc_inst'. The default is 'mjd'.
    norm : bool, optional
        option to normalise the spectra using median values. The default is False.
    output : bool, optional
        option to plot results. The default is False.
    plot_av : bool, optional
        plot average spectra. The default is True.
    savefig : bool, optional
        option to save the plot. The default is False.

    Returns
    -------
    df_av : data frame
        data frame containing all flux values, mean flux, median flux and std flux
        columns are wave, followed by mjd of each individual obs, mean, median, std.

    '''
    
    target=data_dates_range['target'].any() #added this any, check if filename saving still works
    instr=data_dates_range['inst'].any() #added this any, check if filename saving still works

           
    '''create empty data frames for the values and col names'''
    df_av=pd.DataFrame({'wave': w0})
    df_av_col_names=['wave']
    if output == True or savefig == True:
        #ioff()
        fig=figure(figsize=USH.fig_size_l)
    '''create average data framne of flux values'''
    for f in data_dates_range.file:
        data_info,data_wave,data_flux,data_err=read_fits_files(f,verbose=False)   
        #convert to Angstrom for fits files that use nm
        if instr!='XMM':
            if min(data_wave) < 1000:
                data_wave=data_wave * 10 
                
        if data_info[3]=='UVES' or data_info[3]=='SHOOT' or data_info[3]=='XSHOOTER':# or data_info[3]=='ESPRESSO':
            bary_shift=(data_wave * data_info[6]) / clight #shift in the rest wl due to bary
            data_wave=data_wave + bary_shift

        resample=interp1d(data_wave,data_flux,fill_value='extrapolate')
        f0_data=resample(w0)

        if norm == True:
            #f0_data=(f0_data-median(f0_data))/median(f0_data)
            f0_data=(f0_data)/ median(f0_data)
            #f0_data=NormalizeData(f0_data)
            #f0_data=f0_data-median(f0_data)

        #if min(data_wave) < min(w0) < max(data_wave):
        df_av=pd.concat([df_av,pd.Series(f0_data)],axis=1)#this makes the full data frame
        #df_av_col_names.append(data_info[1].replace('-',''))#this names the flux columns by date
        #df_av_col_names.append((data_info[2]))#this names the flux columns by mjd
        if label=='utc_inst':
            df_av_col_names.append(data_info[1]+'_'+data_info[3])#this names the flux columns by utc data and instrument
        elif label=='mjd':
            df_av_col_names.append((data_info[2]))#this names the flux columns by mjd
        
        if output == True or savefig == True:
            plot(w0,f0_data,label=data_info[0]+' '+data_info[1]+' '+data_info[3])
    df_av.columns=df_av_col_names
    av_flux=df_av.iloc[:,1:len(df_av.columns)].mean(axis=1).rename('av_flux')
    med_flux=df_av.iloc[:,1:len(df_av.columns)].median(axis=1).rename('med_flux')
    std_flux=df_av.iloc[:,1:len(df_av.columns)].std(axis=1).rename('std_flux')
    df_av=pd.concat([df_av,av_flux,med_flux,std_flux],axis=1)
    if output == True or savefig == True:
        if plot_av==True:
            plot(w0,df_av.av_flux,'k--',linewidth=1,label='Mean Flux')
            plot(w0,df_av.med_flux,'b--',linewidth=1,label='Median Flux')   
        #legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        legend(loc='upper right',fontsize=8)
        if norm==True:
            ylabel('Noramlised Flux')
        else:
            ylabel('Flux')
        xlabel('Wavelength [Angstroms]')
        #if instr == 'FEROS':
        #    ylim(-0.01,0.1)
        if output==True:
            show()
            tight_layout()
        if savefig==True:
            #output dir
            dirname=os.path.join('av_spec_plots'+'_'+timenow)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            fig.savefig(os.path.join(dirname,target)+'_'+instr+'.pdf')
            print('saving file',os.path.join(dirname,target)+'_'+instr+'.pdf')
            if output == False:
                close()
    #ion()
    return df_av


def wl_plot(df_av,plot_av=True,fs=USH.fig_size_l,output=True,savefig=False,legend=True):
    '''
    plotter for df_av dataframe, for use after wl_exluder has been run

    Parameters
    ----------
    w0 : array
        wavelength values to be plotted.
    df_av : data frame
        result from get_av_spec() or wl_excluder().

    Returns
    -------
    None.

    '''
    ioff()
    #fs=(5,5)
    fig=figure(figsize=fs)
    plot(df_av.wave,df_av.iloc[:,1:len(df_av.columns)-3],linewidth=1)
    if plot_av==True:
        plot(df_av.wave,df_av.av_flux,'k',linewidth=3,label='average')
        plot(df_av.wave,df_av.med_flux,'b--',linewidth=3,label='median')
        #plot(df_av.wave,df_av.std_flux/np.mean(df_av.av_flux),'k--',linewidth=2) 
        fill_between(df_av.wave,0,df_av.std_flux/np.mean(df_av.av_flux),color='grey',label='sd')
    #legend(df_av.columns[1:,], loc='upper left',  fontsize=7, numpoints=1)
    ylabel('Flux')
    xlabel('Wavelength [Angstroms]')
    if legend==True:
        fig.legend(df_av.columns[1:,], fontsize=10, numpoints=1)
    #legend(df_av.columns[1:,], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7, numpoints=1)
    tight_layout()
    if output==True:
        show()
    if savefig==True:
        dirname=os.path.join('wl_plots')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename=os.path.join(dirname,USH.target)+'_'+USH.instrument+'_'+str(int(median(df_av.wave)))+'.pdf'
        fig.savefig(filename)
        print('saving figure: ',filename)
        if output == False:
            close()
    ion()

def get_line_spec(df_av,line,w_range=0.6,vel_offset=0,vel=False):
    '''
    get subset of df_av for given  wavelength range around a line, 
    convert to vel if vel==True, with zero at specified line

    Parameters
    ----------
    df_av : data frame
        average flux dataframe from get_ave_spec().
    line : float
        wavelength position of emission line to consider.
    w_range : float, optional
        wavelength range +/- around line to include. The default is 0.6.
    vel : bool, optional
        option to return velcity values in km/s around line. The default is False.

    Returns
    -------
    df_line_av : data frame
        subset of df_av around line, either as is or with wave replaced with vel if vel==True.

    '''
    if vel==False:
        df_line_av=df_av[df_av['wave'].between(line-w_range,line+w_range)]
    else:
        df_line_av=df_av[df_av['wave'].between(line-200,line+200)]
        if w_range < 10:
            w_range = 10
        
        
    #convert wavelength to velocity around given line
    if vel==True:
        df_vel=pd.Series(((df_line_av.wave - line)*clight/line)-vel_offset,name='vel')
        df_line_av=pd.concat([df_vel,df_line_av],axis=1).drop('wave',axis=1)
        #df_line_av=df_line_av[df_line_av['vel'].between(0-w_range,0+w_range)]

        df_line_av1=df_line_av[df_line_av['vel'].between(0-w_range,0+w_range)]
        df_line_av=df_line_av1
        
        #this is to centre around the max of the line, works for well defined em lines but failing for em lines within absorptions
        #max_flux_idx=numpy.where(df_line_av1.med_flux==max(df_line_av1.med_flux))[0][0]
        #max_flux_vel=df_line_av1.vel.iloc[max_flux_idx]
        #df_line_av=df_line_av[df_line_av['vel'].between(max_flux_vel-w_range,max_flux_vel+w_range)]

    return df_line_av


def vel_plot(df_av_line,start_date='1900',end_date='2100',line=0,fs=USH.fig_size_s,output=True,plot_av=True,plot_sd=False,
             savefig=False):
    global target
    
    '''plotting function for spectra,
        option to convert to velocity given line and radvel'''
    ioff()
    fig=figure(figsize=fs)
    plot(df_av_line.vel,df_av_line.iloc[:,1:len(df_av_line.columns)-3],linewidth=2)
    if plot_av==True:
        plot(df_av_line.vel,df_av_line.av_flux,'k',linewidth=1,label='mean')
        plot(df_av_line.vel,df_av_line.med_flux,'b',linewidth=1,label='median')
        fig.legend(df_av_line.columns[1:-1], fontsize=10, numpoints=1)
    if plot_sd==True:    
        plot(df_av_line.vel,df_av_line.std_flux/df_av_line.med_flux,color='green',alpha=0.5, linestyle='dashed',linewidth=2,label='sd')       
        fill_between(df_av_line.vel,0,(df_av_line.std_flux),color='grey',alpha=0.5)
        if plot_av==True:
            fig.legend(np.append(df_av_line.columns[1:-1],['std_flux/med_flux','std_flux']), fontsize=10, numpoints=1)
        else:
            fig.legend(np.append(df_av_line.columns[1:-3],['std_flux/med_flux','std_flux']), fontsize=10, numpoints=1)
    if plot_av==False and plot_sd==False:
        i=1
        fig.legend(df_av_line.columns[1:-3].values, fontsize=10, numpoints=1)#,bbox_to_anchor=(1.04,1))
    axvline(x=0,color='k',linewidth=0.5,linestyle='--')
    title('Plot of line at %s Angstroms'%(line))
    ylabel('Flux')
    xlabel('v [km/s]')
    tight_layout()
    print(USH.target)
    print(USH.instrument)
    
    if output==True:
        show()
    else:
        close()
    if savefig==True:
        dirname=os.path.join('vel_plots')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename=os.path.join(dirname,USH.target)+'_'+USH.instrument+'_'+str(line)+'.pdf'
        fig.savefig(filename)
        print('saving figure: ',filename)
        if output == False:
            close()
    ion()
    
def vel_xcorr(df1,df2=[],undersample=1,line1=0,line2=0,fs=(8,6)):
    flux1=df1.iloc[:,1:len(df1.columns)-3].values[::undersample]
    vel1=df1['vel'].values[::undersample]
    
    if len(df2)==0:
        flux2=flux1
        vel2=vel1
    else:
        flux2=df2.iloc[:,1:len(df2.columns)-3].values[::undersample]
        vel2=df2['vel'].values[::undersample]
        
    xcor = np.zeros([len(flux1),len(flux2)])
    pcor = np.zeros([len(flux1),len(flux2)])
    for i in range(0,len(flux1)):
        for j in range(0,len(flux2)):
            v=flux1[i]
            w=flux2[j]
            r,p=spearmanr(v,w)
            xcor[i,j]=r
            pcor[i,j]=p
    #print(xcor)
    vel_plot(df1[::undersample],line=line1,plot_av=False,plot_sd=False,fs=USH.fig_size_sn)
    if len(df2)>0:
        vel_plot(df2[::undersample],line=line2,plot_av=False,plot_sd=False,fs=USH.fig_size_sn)

    figure(figsize=fs)
    ax2=[min(vel2), max(vel2), min(vel1), max(vel1)]
    axis(ax2)

    x=[min(vel2),max(vel2)]
    y=[min(vel1),max(vel1)]
    plot(x,y,color='w',linewidth=2, linestyle='--')

    corre=pcolor(vel2,vel1,xcor,cmap=cm.jet, vmin=0, vmax=1,shading='auto')#,norm=colors.PowerNorm(gamma=2))
    #blank low values colour, try white lower end 
    
    #contour(vel,vel,pcor,levels=[1e-15,1e-10,1e-5,1e-4],colors='w',linewidths=1)
    contour(vel2,vel1,xcor,levels=[-0.8,0.8],colors='k',linewidths=1)

    colorbar(corre, shrink=1, aspect=30)

    xlabel(f'{line2} v (km/s)')
    ylabel(f'{line1} v (km/s)')
    text(max(vel2)+max(vel2)*0.4, 0, r'r')


def get_RV(w0,df_av,st_wave,st_flux,st_rv,date='med_flux',w_min=5610,w_max=5710,multi=False,output=True):
    '''
    old version of RV, to be removed. calculate radial velocity of target star using template/standard star and cross correlation

    Parameters
    ----------
    w0 : array
        wavelength array.
    df_av : data frame
        average dataframe containing all flux values.
    st_wave : array
        wavelength array of standard star.
    st_flux : array
        flux array of standard star.
    st_rv : float
        radial velocity of standard star.
    date : float or str, colname of df_av, optional
        the date or which average flux to use. The default is 'med_flux'.
    w_min : float, optional
        min wavelength for rv calculation. The default is 5610.
    w_max : float, optional
        max wavelength for rv calculation. The default is 5710.
    multi : bool, optional
        option to rerun rv calculation across shifting wavelength range to get average and std. The default is False.

    Returns
    -------
    radvel : float
        radial velocity of the target star.

    '''
    print('now calculating radial velocity of target star using template spectra')
    rvmin=-50-st_rv
    rvmax=50-st_rv
    drv=0.1 #steps of rad velo
    vt=st_rv #template radial velocity
    
    df_av_rv=df_av[df_av['wave'].between(w_min,w_max)]
    w0_rv=df_av_rv['wave'].values
    f0_data_rv=df_av_rv[date].values
    
    resample_st=interp1d(st_wave,st_flux,fill_value='extrapolate')
    #f0_st_rv=resample_st(w0_rv)
    w0_st=np.arange(w_min-200,w_max+200,0.1)
    f0_st_rv=resample_st(w0_st)
    
    xcor,ycor=pyasl.crosscorrRV(w0_rv, f0_data_rv, w0_st, f0_st_rv, rvmin, rvmax, 
                                drv, mode='doppler', skipedge=70, edgeTapering=1.)
    fol=(ycor==max(ycor))
    radvel=float(xcor[fol])+vt
    rvs=[radvel]
    
    if output == True:
        wl_plot(df_av_rv,plot_av=False,fs=USH.fig_size_n)
        #plot(w0_st,f0_st_rv)
        figure()
        plot(xcor+vt,ycor/max(ycor), ':', label='Temp')

    
    if multi == True:
        for i in range(1,6):
            w_min += 50
            w_max += 50
            df_av_rv=df_av[df_av['wave'].between(w_min,w_max)]
            w0_rv=df_av_rv['wave'].values
            f0_data_rv=df_av_rv[date].values
            f0_st_rv=resample_st(w0_rv)
            xcor,ycor=pyasl.crosscorrRV(w0_rv, f0_data_rv, w0_rv, f0_st_rv, rvmin, rvmax, drv, mode='doppler', skipedge=200, edgeTapering=1.)
            #plot(xcor+vt,ycor/max(ycor), ':', label='Temp')
            fol=(ycor==max(ycor))
            radvel_i=float(xcor[fol])+vt
            rvs.append(radvel_i)
            if output == True:
                plot(xcor+vt,ycor/max(ycor))
        radvel=np.round(np.mean(rvs),2)
        print ('av rad vel = %.2f km/s, sd = %.2f' %(np.mean(rvs),np.std(rvs)))
    else:
        print ('rad vel = %.2f km/s' %(radvel))
    
    if output == True:   
        axvline(x=radvel,color='k',linewidth=0.5,linestyle='--')
        xlabel('Radial Velocity [km/s]')
        ylabel('Normalised Xcor')
    return radvel

def get_vsini(w0,df_av,st_wave,st_flux,st_rv,date='med_flux',w_min=5610,w_max=5710,output=True):
    '''
    old version of vsini, to be removed.
    calculate the projected rotational velocity, vsini, or target star using template/sandard
    spectra, cross correlation and broadening

    Parameters
    ----------
    w0 : array
        wavelength array of target star.
    df_av : data frame
        data frame containing all flux observations for each wavelength and averages.
    st_wave : array
        standard star wavelength array.
    st_flux : array
        standard star flux array.
    st_rv : float
        standard star radial velocity.
    date : float or str, colname of df_av, optional
        the date or which average flux to use. The default is 'med_flux'.
    w_min : float, optional
        min wavelength for rv calculation. The default is 5610.
    w_max : float, optional
        max wavelength for rv calculation. The default is 5710.
    output : bool, optional
        option to plot results of vsini calculation. The default is True.

    Returns
    -------
    vsini : float
        vsini of target star.

    '''
    
    print('now calculating vsini of target star using template spectra')
    start_time = time.time()
    
    df_av_rv=df_av[df_av['wave'].between(w_min,w_max)]
    w0_rv=df_av_rv['wave'].values
    f0_data_rv=df_av_rv[date].values
    
    resample_st=interp1d(st_wave,st_flux,fill_value='extrapolate')
    f0_st_rv=resample_st(w0_rv)  
    
    radvel=get_RV(w0_rv,df_av,st_wave,st_flux,st_rv,date,w_min,w_max,multi=True,output=False)
    
    if output == True:
        
        fig, ax = subplots(1, 2,figsize=(10,5))#,gridspec_kw={'wspace':0})
        fig.suptitle('V sin i calculation')    
        
        ax[0].set(xlabel='Rad. Vel. (km/s)',ylabel='Normalized Xcor')
        ax[1].set(ylabel='Vsini (km/s)',xlabel='Xcorr width (km/s)')
    
        #Parameters for the xcor and broadening:
    rvmin=-50.
    rvmax=50.
    drv=0.1 #steps of rad velo
    epsilon=0 #0.6 #limb darkening for models, good for young stars, Dahm+12, Hartmann+86
    vm=-50.
    vmm=50. #max and min vel for the fit.
    cutparam=0.1 #0.2
    
    #Now do some broadened spectra:    
    kms_list=[3,5,7,9,12,15,20,25,30,35]#,50]
    
    '''create empty data frames for the values and col names'''
    broad_flux=pd.DataFrame({'wave': w0_rv})
    col_names=['wave']
    
    for kms in kms_list:
        #f=pyasl.rotBroad(w0_rv,f0_st_rv, epsilon, kms, edgeHandling='firstlast')
        f=pyasl.fastRotBroad(w0_rv, f0_st_rv, epsilon, kms)
        broad_flux=pd.concat([broad_flux,pd.Series(f)],axis=1)
        col_names.append('%s km/s'%(kms))
        #plot(w0,f)
    broad_flux.columns=col_names        
      
    #Get xcors for all   
    x_xcor=pd.DataFrame()
    y_xcor=pd.DataFrame()
    widths=[]
    widthplace=0.997
    for kms in broad_flux.columns[1:len(broad_flux.columns)]:
        x1,y1=pyasl.crosscorrRV(w0_rv, numpy.array(broad_flux.loc[:,kms]), w0_rv, f0_st_rv, rvmin, rvmax, drv, mode='doppler', skipedge=200, edgeTapering=1.)
        #plot(x1,y1)
        filter1=(y1>min(y1)+cutparam*(max(y1)-min(y1))) & (x1>vm) & (x1<vmm)
        #filter1=(x1>vm) & (x1<vmm)
        x1=x1[filter1]
        y1=y1[filter1]/max(y1)
        
        #guassian fit of the xcorr
        gfit=fit_gauss(x1,y1)
        #y1=gfit.best_fit
        #width=gfit.best_values['g1_sigma']
        #width=gfit.values['g1_fwhm']
        
        if output == True:
            ax[0].plot(x1,y1,label=kms)
                  
        x_xcor=pd.concat([x_xcor,pd.Series(x1,name=kms)],axis=1)
        y_xcor=pd.concat([y_xcor,pd.Series(y1,name=kms)],axis=1)
              
        foli=(x1<x1[np.argmax(y1)]) & (y1<widthplace)
        folo=(x1>x1[np.argmax(y1)]) & (y1<widthplace)
        width=abs(min(x1[folo])-max(x1[foli]))
        widths.append(width)
    

    p_vsini=polyfit(widths,kms_list,deg=2) 
    xx=arange(min(widths), max(widths), 0.1)
    yy=polyval(p_vsini,xx)
    if output == True:
        ax[1].plot(widths,kms_list,'bo')
        ax[1].plot(xx,yy,'k:')

    
    w0_st=np.arange(w_min-200,w_max+200,0.1)
    f0_st_rv=resample_st(w0_st)
    xcor,ycor=pyasl.crosscorrRV(w0_rv, f0_data_rv, w0_st, f0_st_rv, rvmin-st_rv, rvmax-st_rv, drv, mode='doppler', skipedge=200, edgeTapering=1.)
    #Get the vsini:
    filter2=(ycor>min(ycor)+cutparam*(max(ycor)-min(ycor)))
    #filter2=((xcor+st_rv)-radvel>vm) & ((xcor+st_rv)-radvel<vmm)
    ycorf=ycor[filter2]/max(ycor)
    xcorf=xcor[filter2]

    
    #guassian fit of the xcorr
    gfit=fit_gauss(xcorf,ycorf)
    #ycorf=gfit.best_fit
    #width=gfit.best_values['g1_sigma']
    #width=gfit.values['g1_fwhm']
    
    
    if output == True:
        ax[0].plot((xcorf+st_rv)-radvel,ycorf,'k--', linewidth=3, label='target star')
        comps = gfit.eval_components(x=xcorf)
        #ax[0].plot((xcorf+st_rv)-radvel, comps['g1_'], 'g--', label='Gauss 1')
        #ax[0].plot((xcorf+st_rv)-radvel, comps['line_'], 'k--', label='linear')

        ax[0].legend(fontsize=8)
            
    #Just measuring the width
    foli=(xcorf<xcorf[argmax(ycorf)]) & (ycorf<widthplace)
    folo=(xcorf>xcorf[argmax(ycorf)]) & (ycorf<widthplace)
    width=abs(min(xcorf[folo])-max(xcorf[foli]))
    vsini=polyval(p_vsini, width)
    if output == True:
        ax[1].hlines(vsini, 0, width)
        ax[1].vlines(width, 0, vsini)
   

    print('width = %f' %(width))   
    print('vsini = %f km/s' %(vsini))
    
                
    elapsed_time = time.time() - start_time
    #print('duration of vsini calculation:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    print('rv=%.1f, vsini=%.2f'%(radvel,vsini))
    
    
    return np.round(vsini,2),gfit
    
    
def get_rv_vsini(df_av,st_wave,st_flux,st_rv,date='med_flux',vsini_max=50,w_min=5000,w_max=5500,output=True,rtn_err=False):
    '''
    

    Parameters
    ----------
    df_av : dataframe
        input observation dataframe.
    st_wave : array
        standard star wavelength values.
    st_flux : array
        standard star flux values.
    st_rv : float
        radial velocity of standard star.
    date : str, optional
        observation date to use from observation dataframe. The default is 'med_flux'.
    w_min : float, optional
        min wavelength to consider for observation data. The default is 5000.
    w_max : float, optional
        max wavelength to consider for observation data. The default is 5500.
    output : bool, optional
        plot outputs. The default is True.

    Returns
    -------
    radvel : float
        radial velocity of target star.
    vsini : float
        vsini of target star.

    '''
    print('now calculating radial velocity of target star using template spectra')
    vt=st_rv #template radial velocity
    rvmin=-100-vt #adjust range so that zero is centred
    rvmax=100-vt
    drv=0.1 #steps of rad velo
    
    df_av_rv=df_av[df_av['wave'].between(w_min,w_max)] #range to take for RV
    w0_rv=df_av_rv['wave'].values
    f0_data_rv=df_av_rv[date].values
    f0_data_rv=NormalizeData(f0_data_rv)
    
#     w_smooth=np.arange(min(w0_rv),max(w0_rv),5) #set a larger step interval to undersample input data
#     smooth=interp1d(w0_rv,f0_data_rv,fill_value='extrapolate')
#     f0_data_rv=smooth(w_smooth)
#     w0_rv=w_smooth
    
    if max(st_wave) < 1000:
        st_wave=st_wave*10
    resample_st=interp1d(st_wave,st_flux,fill_value='extrapolate')
    w0_st=np.arange(w_min-100,w_max+100,0.05) #larger range for template
    f0_st_rv=resample_st(w0_st)
    f0_st_rv=NormalizeData(f0_st_rv)
    
    xcor,ycor=pyasl.crosscorrRV(w0_rv, f0_data_rv, w0_st, f0_st_rv, rvmin, rvmax, 
                                drv, mode='doppler', skipedge=70, edgeTapering=1.)
    #guassian fit of the xcorr
    gfit=fit_gauss(xcor,ycor)
    g1_stderr=gfit.params['g1_center'].stderr
    ycor=gfit.best_fit
    ycor=ycor/median(ycor)
    t_width=gfit.best_values['g1_sigma']
    t_chi=gfit.redchi
    fol=(ycor==max(ycor))
    radvel=float(xcor[fol])+vt
    rvs=[radvel]
    t_widths=[t_width]
    t_chis=[t_chi]
    t_ycors=[ycor]
    g1_stderrs=[g1_stderr]
    
    if output == True:
        #wl_plot(df_av_rv,plot_av=False,fs=USH.fig_size_n)
        figure(figsize=USH.fig_size_n)
        plot(w0_rv,(f0_data_rv),label='target')
        plot(w0_st,(f0_st_rv),alpha=0.5,label='template')   
        legend(loc='upper left', fontsize=8, numpoints=1)
        figure()
        plot(xcor+vt,ycor)
        
    for i in range(1,6):
        w_min += 10
        w_max += 10
        df_av_rv=df_av[df_av['wave'].between(w_min,w_max)]
        w0_rv=df_av_rv['wave'].values
        f0_data_rv=df_av_rv[date].values
        xcor,ycor=pyasl.crosscorrRV(w0_rv, f0_data_rv, w0_st, f0_st_rv, rvmin, rvmax, drv, mode='doppler', skipedge=20, edgeTapering=1.)
        gfit=fit_gauss(xcor,ycor)
        ycor=gfit.best_fit
        ycor=ycor/median(ycor)
        t_width=gfit.best_values['g1_sigma']
        t_chi=gfit.redchi
        fol=(ycor==max(ycor))
        radvel_i=float(xcor[fol])+vt
        rvs.append(radvel_i)
        g1_stderrs.append(gfit.params['g1_center'].stderr)
        t_widths.append(t_width)
        t_chis.append(t_chi)
        t_ycors.append(ycor)
        if output == True:
            plot(xcor+vt,ycor)
            xlabel('Radial Velocity [km/s]')
            ylabel('Normalised Xcor')
            axvline(x=radvel,color='k',linewidth=0.5,linestyle='--')
    radvel=np.round(np.mean(rvs),2)
    print ('av rad vel = %.2f km/s, sd = %.2f' %(np.mean(rvs),np.std(rvs)))
    #print ('rv cen mean std err= %.4f' %(np.mean(g1_stderr)))
    best_t_width=t_widths[argmin(t_chis)]
    best_t_ycors=t_ycors[argmin(t_chis)]
    
    #now have the width of the xcor and values for the best fit, with lowest chisq
    
    print('now calculating vsini of target star using template spectra')
    
    if output == True:
        
        fig, ax = subplots(1, 2,figsize=(10,5))#,gridspec_kw={'wspace':0})
        fig.suptitle('V sin i calculation')    
        
        ax[0].set(xlabel='Rad. Vel. (km/s)',ylabel='Normalized Xcor')
        ax[1].set(ylabel='Vsini (km/s)',xlabel='Xcorr width (km/s)')
    
    #Parameters for the xcor and broadening:
    rvmin=-100
    rvmax=100
    drv=0.1 #steps of rad velo
    epsilon=0 #0.6 #limb darkening for models, good for young stars, Dahm+12, Hartmann+86
    vm=-100.
    vmm=100. #max and min vel for the fit.
   
    #Now do some broadened spectra:    
    kms_list=[2,3,5,7,9,12,15,20,25,30,35,50,100]
    kms_list=np.arange(2,vsini_max,np.round((vsini_max-2)/10,0))
    
    '''create empty data frames for the values and col names'''
    broad_flux=pd.DataFrame({'wave': w0_st})
    col_names=['wave']
    
    for kms in kms_list:
        #f=pyasl.rotBroad(w0_rv,f0_st_rv, epsilon, kms, edgeHandling='firstlast')
        f=pyasl.fastRotBroad(w0_st, f0_st_rv, epsilon, kms)
        broad_flux=pd.concat([broad_flux,pd.Series(f)],axis=1)
        col_names.append('%s km/s'%(kms))
        #plot(w0,f)
    broad_flux.columns=col_names        
      
    #Get xcors for all   
    x_xcor=pd.DataFrame()
    y_xcor=pd.DataFrame()
    s_widths=[]
    for kms in broad_flux.columns[1:len(broad_flux.columns)]:
        x1,y1=pyasl.crosscorrRV(w0_st, numpy.array(broad_flux.loc[:,kms]), w0_st, f0_st_rv, rvmin, rvmax, drv, mode='doppler', skipedge=200, edgeTapering=1.)
        filter1=(x1>vm) & (x1<vmm)
        x1=x1[filter1]
        y1=y1[filter1]/median(y1) 
        #y1=
        #guassian fit of the xcorr
        gfit=fit_gauss(x1,y1)
        y1=gfit.best_fit
        y1=NormalizeData(y1)
        s_width=gfit.best_values['g1_sigma']
        
        if output == True:
            ax[0].plot(x1,y1,label=kms)
                  
        x_xcor=pd.concat([x_xcor,pd.Series(x1,name=kms)],axis=1)
        y_xcor=pd.concat([y_xcor,pd.Series(y1,name=kms)],axis=1)
    
        s_widths.append(s_width)
        
    
    #xcor,ycor=pyasl.crosscorrRV(w0_rv, f0_data_rv, w0_st, f0_st_rv, rvmin-st_rv, rvmax-st_rv, drv, mode='doppler', skipedge=20, edgeTapering=1.)
    #filter2=((xcor+st_rv)-radvel>vm) & ((xcor+st_rv)-radvel<vmm)
    #ycorf=ycor[filter2]/median(ycor)
    #ycorf=NormalizeData(ycorf)
    #xcorf=xcor[filter2]
    #guassian fit of the xcorr
    #gfit=fit_gauss(xcorf,ycorf)
    #ycorf=gfit.best_fit
    #ycorf=NormalizeData(ycorf)
    #width=np.round(gfit.best_values['g1_sigma'],2)
    #width_stderr=gfit.params['g1_sigma'].stderr
    
    width=np.mean(t_widths)
    width_err=np.std(t_widths)

    p_vsini=polyfit(s_widths,kms_list,deg=2) 
    xx=arange(min(s_widths), max(s_widths), 0.1)
    yy=polyval(p_vsini,xx)
    av_vsini=[]
    for i in t_widths:
        vsini=np.round(polyval(p_vsini, i),2)
        av_vsini.append(vsini)
    vsini=np.round(np.mean(av_vsini),2)
    
    if output == True:
        ax[1].plot(s_widths,kms_list,'bo')
        ax[1].plot(xx,yy,'k:')
        #ax[1].hlines(vsini, 0, width)
        #ax[1].vlines(width, 0, vsini)
        for i in range(len(t_widths)):
            ax[1].hlines(av_vsini[i], 0, t_widths[i])
            ax[1].vlines(t_widths[i], 0, av_vsini[i])
        ax[1].set_xlim(min(xx)-2,max(xx)+2)
        
        #ax[0].plot(xcor+vt-radvel,NormalizeData(best_t_ycors),'b--', linewidth=3, label='b target star')
        #ax[0].plot((xcorf+st_rv)-radvel,ycorf,'k--', linewidth=3, label='target star')
        for i in t_ycors:
            ax[0].plot(xcor+vt-radvel,NormalizeData(i),'b--', linewidth=1)
        ax[0].legend(fontsize=8)
    
    #print('best width = %.2f' %(best_t_width))  
    print('av width = %.2f , sd = %.2f' %(width,width_err))  
    print('av vsini = %.2f km/s , sd = %.2f' %(vsini,np.std(av_vsini)))
    
    if vsini<2:
        print('ERROR with vsini calculation, value %.2f < template vel.'%(vsini))
        print('Try a different wavelength range, different instrument, or different template')
        print('Setting vsini to 5 km/s')
        vsini=5.0
        
    USH.radvel=radvel
    USH.vsini=vsini
    
    if rtn_err==True:
        return radvel, np.std(rvs), vsini,np.std(av_vsini)
    else:
        return radvel,vsini#,gfit
    

def subtract_cont(df_av,av='med',poly=3,wl_win=0.5,coeff=31,output=False,plot_x=[],plot_y=[]):
    '''
    For a given flux and wavelength, apply sigma clipping, undersampling and
    savgol-golay filter to approximately remove the continuum.
    This is only used for finding the emission lines and is not an accurate model of the continuum.

    Parameters
    ----------
    df_av : dataframe
        wave and flux dataframe.
    av : str, optional
        obs to use. The default is 'med'.
    poly : int, optional
        polynomial order to use in SG filter. The default is 3.
    wl_win : float, optional
        wavelength window to use in the SG filter. The default is 0.5.
    coeff : int, optional
        coeff of SG filter. The default is 31.
    output : bool, optional
        plot outputs. The default is False.
    plot_x : float, optional
        min wave range for plotting. The default is [].
    plot_y : float, optional
        max wave range for plotting. The default is [].

    Returns
    -------
    f_flat : array
        flux array of roughly continuum subtracted spectra.

    '''

    w0=df_av.wave
    if plot_x==[]:
        plot_x=[min(w0),max(w0)]
         
    if av=='med':
        f0_data=df_av.med_flux
    else:
        f0_data=df_av.av_flux
    
    if plot_y==[]:
        plot_y=[min(f0_data),max(f0_data)]
    
   
    '''clip the data then resample back to original w0'''
    f0_mask=sigma_clip(f0_data,sigma_lower=2.5,sigma_upper=5)
    w0_clip=w0[~f0_mask.mask]
    f0_clip=f0_data[~f0_mask.mask]
    clip_resample=interp1d(w0_clip,f0_clip,fill_value='extrapolate')
    f0_sc=clip_resample(w0)
    
    '''undersample the data to get rough continuum'''
    w_smooth=np.arange(min(w0),max(w0),wl_win) #set a larger step interval to undersample input data
    smooth=interp1d(w0,f0_sc,fill_value='extrapolate')
    f_smooth=smooth(w_smooth)
    #print('smooth std:',np.std(f_smooth))
    
    '''apply Savitzky-Golay filter to get continuum
        expand values back out to original wavelength intervals
        subtract continuum from raw flux values'''
    f_sf=savgol_filter(f_smooth,coeff,poly)
    expand=interp1d(w_smooth,f_sf,fill_value='extrapolate')
    f_sf_full=expand(w0)
    f_flat=f0_data-f_sf_full
    print('savgol std:',np.std(f_sf_full))
    
    if output == True:
        #ioff()
        figure(figsize=USH.fig_size_n)
        cla()
        plot(w0,f0_data,linewidth=0.75,label='Input Average Flux')
        #plot(w0,f0_sc,label='sigma clipped')
        plot(w_smooth,f_smooth,linewidth=0.75,label='Undersampled')
        plot(w_smooth,f_sf,linewidth=1,label='S-G Flter, order=%s'%(poly))
        plot(w0,f_flat,linewidth=0.75,label='Continuum Subtracted')
        ylabel('Normalised Flux')
        xlabel('Wavelength [Angstroms]')
        xlim(plot_x)
        ylim(plot_y)
        
        legend(loc='upper left', fontsize=8, numpoints=1)
    
        tight_layout()
        show()
        #ion()
    return f_flat


def subtract_temp(df_av,w0_st,f0_st,av='med',poly=3,wl_win=1,coeff=31,output=False,plot_x=[],plot_y=[]):
    '''
    template subtraction
    
    Parameters
    ----------
    df_av : dataframe
        wave and flux dataframe.
    av : str, optional
        obs to use. The default is 'med'.
    poly : int, optional
        polynomial order to use in SG filter. The default is 3.
    wl_win : float, optional
        wavelength window to use in the SG filter. The default is 0.5.
    coeff : int, optional
        coeff of SG filter. The default is 31.
    output : bool, optional
        plot outputs. The default is False.
    plot_x : float, optional
        min wave range for plotting. The default is [].
    plot_y : float, optional
        max wave range for plotting. The default is [].

    Returns
    -------
    f_flat : array
        flux array of roughly continuum subtracted spectra.

    '''

    w0=df_av.wave
    if plot_x==[]:
        plot_x=[min(w0),max(w0)]
         
    if av=='med':
        f0_data=df_av.med_flux
    else:
        f0_data=df_av.av_flux
    
    if plot_y==[]:
        plot_y=[min(f0_data),max(f0_data)]
      
    f0_st=NormalizeData(f0_st)
    #f0_data=NormalizeData(f0_data)
    #f0_st=(f0_st)/median(f0_st)
    f0_data=(f0_data)/median(f0_data)
    
    '''clip the input data then resample back to original w0'''
    f0_mask=sigma_clip(f0_data,sigma_lower=0,sigma_upper=1)
    w0_clip=w0[~f0_mask.mask]
    f0_clip=f0_data[~f0_mask.mask]
    clip_resample=interp1d(w0_clip,f0_clip,fill_value='extrapolate')
    f0_sc=clip_resample(w0)
    
    '''undersample the data to get rough continuum'''
    w_smooth=np.arange(min(w0_st),max(w0_st),wl_win) #set a larger step interval to undersample input data
    smooth=interp1d(w0_st,f0_st,fill_value='extrapolate')
    f_smooth=smooth(w_smooth)
    #print('smooth std:',np.std(f_smooth))
    
    '''apply Savitzky-Golay filter to get continuum
        expand values back out to original wavelength intervals
        subtract continuum from raw flux values'''
    f_sf=savgol_filter(f_smooth,coeff,poly)
    expand=interp1d(w_smooth,f_sf,fill_value='extrapolate')
    f_sf_full=expand(w0_st)
    print('savgol std:',np.std(f_sf_full))
    
    '''subtract template from target spectra'''
    expand=interp1d(w_smooth,f_sf,fill_value='extrapolate')
    f_sf_data_full=expand(w0)
    f_sub=f0_data-f_sf_data_full
    
    if output == True:
        figure(figsize=USH.fig_size_n)
        plot(w0,(f0_data),label='target')
        plot(w0_st,(f0_st),alpha=0.5,label='template')
        legend(loc='upper left', fontsize=8, numpoints=1)
    
    if output == True:
        #ioff()
        figure(figsize=USH.fig_size_l)
        cla()
        plot(w0,f0_data,linewidth=0.75,label='Input  norm. target')
        plot(w0_st,f0_st,linewidth=0.75,label='Input Norm. template')
        #plot(w0,f0_sc,label='sigma clipped')
        plot(w_smooth,f_smooth,linewidth=0.75,label='Undersampled')
        plot(w_smooth,f_sf,linewidth=1,label='S-G Flter, order=%s'%(poly))
        plot(w0,f_sub,linewidth=0.75,label='Template Subtracted')
        ylabel('Normalised Flux')
        xlabel('Wavelength [Angstroms]')
        xlim(plot_x)
        ylim(plot_y)
        
        legend(loc='upper left', fontsize=8, numpoints=1)
    
        tight_layout()
        show()
        #ion()
    return f_sub #check




def find_em_lines(df_av,f_flat,radvel,vsini,sigma=2.5,av='med',atol=0.5,wl_win=1,
                  output=False,line_id=False,prev_lines_only=False,xrange=[],xlim_min='min',xlim_max='max'):
    '''
    function to find EM lines for flat spectra
    req. database of lines to compare with 'obs_wl_air' as the wavelength col

    Parameters
    ----------
    w0 : array
        wavelength array of target star.
    f_flat : array
        output of subtract_cont() flux values with continuum subtracted.
    f0_data : array
        original flux values used in subtract_cont() in order to position em points on plots.
    radvel : float
        radial velocity of target star.
    vsini : float
        vsini of target star.
    sigma : float, optional
        sigma level for thresholding of emission lines. The default is 2.5.
    output : bool, optional
        option to plot results. The default is False.
    line_id : bool, optional
        option to plot labels on matched emission lines. The default is False.
    xlim_min : float, optional
        x limits for output and final fits. The default is min(w0).
    xlim_max : float, optional
        x limits for output and final fits. The default is max(w0).
        
    Returns
    -------
    em_matches : data frame
        dataframe containing list of matched emission lines and properties such as centre, fwhm, snr.
    em_match_common_Ek : data_frame
        subset of em_matches of lines originating from common upper energy levels.

    '''
    
    tar_inst=USH.target + '_' + str(USH.instrument)
    
    w0_ini=df_av.wave
       
    if av=='med':
        f0_data=df_av.med_flux
    else:
        f0_data=df_av.av_flux
        
    '''undersample the data to get rough continuum'''
    w0=np.arange(min(w0_ini),max(w0_ini),wl_win) #set a larger step interval to undersample input data
    smooth=interp1d(w0_ini,f_flat,fill_value='extrapolate')
    f_flat=smooth(w0)
    f0_data=smooth(w0)
    #print('smooth std:',np.std(f_smooth))
    
    if xrange==[]:
        xlim_min=min(w0)
        xlim_max=max(w0)
    else:
        xlim_min=xrange[0]
        xlim_max=xrange[1]
    
    f_flat_clip=sigma_clip(f_flat,sigma_lower=20,sigma_upper=sigma)
    
    w0_flat_clip=w0[f_flat_clip.mask]#take the corresponding wavelength values from the mask creating the clip
    a=np.array(f_flat[f_flat_clip.mask])#take just the clipped flux values from flat spectra
    a_f0=np.array(f0_data[f_flat_clip.mask])#clipped flux values from input data

        
    em1=np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True] #find max points from nn 1d
    em2=argrelextrema(a, np.greater,order=1)[0] #find max points using argrelextrema signal filter
    #the signal filter seems to drop some lines based on the amount of 'line' left either side of the max point, hence using the em1
    
    w0_em=np.array(w0_flat_clip[em1]) #take index values of max points to get w0 of em lines
    f0_em=a[em1]#flux value from flat spectra
    f0_data_em=a_f0[em1]#flux values from input spectra
    
    
    
    ''' list of lines''' 
    
    if prev_lines_only==True:
        line_table=USH.JCW_lines_NIST
        wave='obs_wl_air'
        print('Using previously observed lines only and',wave)
    elif USH.instrument[0]=='XMM':
        line_table=USH.xr_line_table
        line_table=USH.xrs_line_table
        wave='ritz_wl_vac'
        print('Using x-ray line file and ',wave)
    elif USH.instrument[0]=='Sol':
        line_table=USH.sol_line_table
        wave='ritz_wl_vac'
        print('Using Sol line file and ',wave)
    elif USH.instrument[0]=='COS' or USH.instrument[0]=='STIS':
        line_table=USH.sol_line_table
        wave='ritz_wl_vac'
        print('Using Sol line file and ',wave)
    else:
        line_table=USH.line_table
        line_table=USH.line_table_prev_obs
        wave='obs_wl_air'
        print('Using NIST list with previous observations indicated and',wave)
    
    rv_shift=(line_table[wave].values * radvel) / clight #shift in the rest wl due to rv
    vsini_shift=(line_table[wave].values * vsini) / clight #shift in the rest vl due to vsini (broadening)
    rv_wl=line_table[wave].values + rv_shift #comparison wl to compare to observed, accounting for rv shift
    line_table=pd.concat([pd.Series(np.around(rv_wl,decimals=2),name='rv_wl'),
                          line_table,
                          pd.Series(np.around(rv_shift,decimals=2),name='rv_shift'),
                          pd.Series(np.around(vsini_shift,decimals=2),name='vsini_shift')],axis=1)
    
    #find lines that are close to obs wl and ref wl, filter out all lines that are not close
    em_lines_mask=np.isclose(line_table['rv_wl'].values[:,None],w0_em, atol=atol).any(axis=1)
    em_matches=line_table[em_lines_mask].reset_index(drop=True)
    
    
    w0_matches=[]
    f0_flat_matches=[]
    f0_data_matches=[]
    for wl in range(len(em_matches)):
        w0_line_pos=np.isclose(em_matches.rv_wl.values[wl],w0_em,atol=atol)
        w0_matches.append(w0_em[w0_line_pos][0])
        f0_flat_matches.append(f0_em[w0_line_pos][0])
        f0_data_matches.append(f0_data_em[w0_line_pos][0])
        
    w0_matches=[round(x,2) for x in w0_matches]
    em_matches=pd.concat([pd.Series(w0_matches,name='w0'),
                          pd.Series(((w0_matches - em_matches.rv_wl)*clight/em_matches.rv_wl),name='vel_diff'),
                          em_matches,
                          pd.Series(f0_flat_matches,name='f0_flat'),
                          pd.Series(f0_data_matches,name='f0_data'),
                          pd.Series((f0_flat_matches/std(f_flat_clip)),name='SNR')],axis=1)
    
    if USH.instrument[0]!='XMM':
    #remove lines that are further away from the reference line than the shift induced by the vsini
        em_matches=em_matches.drop(em_matches[abs(em_matches.w0 - em_matches.rv_wl) > 2* abs(em_matches.vsini_shift)].index)
    
    #only keep values between x limits
    em_matches=em_matches[em_matches['w0'].between(xlim_min,xlim_max)].sort_values(by=['w0','Acc'],ascending=[True, True])
    
    
    #pd.set_option('mode.chained_assignment',None)
    #check for lines with >1 match to databse
    #for wl in em_matches.w0:
    #    if len(em_matches[em_matches.w0 == wl]) > 1:
    #        em_matches.element.loc[em_matches.w0 == wl] = em_matches[em_matches.w0 == wl].element.values[0]+'*'
    em_matches['multi']=np.where([len(em_matches[em_matches.w0 == wl])>1 for wl in em_matches.w0],'yes','no')        
        
    em_matches['tar_inst']=tar_inst
    
    em_matches['abs_vel_diff']=abs(em_matches['vel_diff'])
    em_matches.sort_values(['w0','prev_obs','sp_num','Aki'],ascending=[True,True, True, False],inplace=True)
    
    #find em lines that are from the same upper energy level
    check_Ek=np.column_stack(np.unique(em_matches.Ek,return_counts=True))
    common_Ek=check_Ek[check_Ek[:,1]>1][:,0]
    em_match_common_Ek=em_matches[em_matches.Ek.isin(common_Ek)]
    
        
    if output == True:
        figure(figsize=USH.fig_size_n)
        cla()
        #plot(w0,f0_data+1,'b',label='Input Spectra')
        plot(w0,f_flat,'r',label='Continuum Sub.')
        xlim(xlim_min,xlim_max)
        #ylim(-2,10.0)
        plot(w0,f_flat_clip,label='Threshold')
        plot(w0_em,f0_em,'b.',label='Potential Em. Line')
        plot(w0_matches,f0_flat_matches,'go',label='NIST Matched Em. Line')
        if line_id==True:
             [axvline(x=_x,color='k',linewidth=0.5,linestyle='--') for _x in w0_matches]
             line=0.0 #this stops multiple labels being plotted for lines matched to more than one emission
             for index,row in em_matches.iterrows():
                 if row.w0 != line:
                     line = row.w0
                     flux = row.f0_flat
                     name = '%s%s %.2f'%(row.element,row.sp_num,row.w0)
                     annotate(name,(line,flux),rotation=90,size=14,
                              xytext=(10, 10),  # 3 points vertical offset
                              textcoords="offset pixels",
                              horizontalalignment='left', verticalalignment='bottom')
                
        legend(loc='upper left', fontsize=8, numpoints=1)
        ylabel('Normalised Flux')
        xlabel('Wavelength [Angstroms]')
        locator_params(axis='x', nbins=4)        
        #tight_layout()

    if output == True:
        figure(figsize=USH.fig_size_n)
        cla()
        #plot(w0,f0_data+1,'b',label='Input Spectra')
        plot(w0,f0_data,'r',label='Input spectra')
        xlim(xlim_min,xlim_max)
        #ylim(-2,10.0)
        #plot(w0,f_flat_clip,label='Threshold')
        plot(w0_em,f0_data_em,'b.',label='Potential Em. Line')
        plot(w0_matches,f0_data_matches,'gx',label='NIST Matched Em. Line')
        if line_id==True:
             [axvline(x=_x,color='k',linewidth=0.5,linestyle='--') for _x in w0_matches]
             line=0.0 #this stops multiple labels being plotted for lines matched to more than one emission
             for index,row in em_matches.iterrows():
                 if row.w0 != line:
                     line = row.w0
                     flux = row.f0_data
                     name = '%s%s %.2f'%(row.element,row.sp_num,row.w0)
                     annotate(name,(line,flux),rotation=90,size=14,
                              xytext=(10, 10),  # 3 points vertical offset
                              textcoords="offset pixels",
                              horizontalalignment='left', verticalalignment='bottom')
                
        legend(loc='upper left', fontsize=8, numpoints=1)
        ylabel('Flux')
        xlabel('Wavelength [Angstroms]')
        locator_params(axis='x', nbins=4)
        #tight_layout()
    
        
    return em_matches,em_match_common_Ek

def plot_em_lines(df_av,em_matches,plot_av=False,fs=USH.fig_size_l):
    '''
    function to plot the average data frame with a list of identified lines

    Parameters
    ----------
    df_av : dataframe
        dataframe of wave and flux values.
    em_matches : dataframe
        dataframe of emission line matches.
    plot_av : bool, optional
        plot average spectra. The default is False.
    fs : tuple, optional
        figure size. The default is USH.fig_size_l.

    Returns
    -------
    plot.

    '''
    
    em_matches_cut=em_matches[em_matches.w0.between(min(df_av.wave),max(df_av.wave))]
    f0_data_matches=em_matches_cut.f0_data
    w0_matches=em_matches_cut.w0
    figure(figsize=fs)
    cla()
    #plot(w0,f0_data+1,'b',label='Input Spectra')
    plot(df_av.wave,df_av.med_flux,linewidth=1,label='median')
    plot(w0_matches,f0_data_matches,'gx',label='NIST Em. Line from ref table')
    
    [axvline(x=_x,color='k',linewidth=0.5,linestyle='--') for _x in w0_matches]
    line=0.0 #this stops multiple labels being plotted for lines matched to more than one emission
    for index,row in em_matches.iterrows():
        if row.w0 != line:
            line = row.w0
            flux = row.f0_data
            name = '%s%s %.2f'%(row.element,row.sp_num,row.w0)
            annotate(name,(line,flux),rotation=90,size=14,
                  xytext=(10, 10),  # 3 points vertical offset
                  textcoords="offset pixels",
                  horizontalalignment='left', verticalalignment='top')

    legend(loc='upper left', fontsize=8, numpoints=1)
    ylabel('Flux')
    xlabel('Wavelength [Angstroms]')
    locator_params(axis='x', nbins=4)    
    

    
    
def fit_gauss(x,y,ngauss=1,neg=False,g1_cen=None,g2_cen=None,g3_cen=None,neg_cen=None,
             g1_sig=None,g2_sig=None,g3_sig=None,neg_sig=None):
    '''
    

    Parameters
    ----------
    x : array or list
        wave.
    y : array or list
        flux.
    ngauss : int, optional
        number of positie Gaussians to fit. The default is 1.
    neg : bool, optional
        Whether to include a negative Gaussian. The default is False.
    g1_cen : list, optional
        list of min and max. The default is None.
    g2_cen : list, optional
        list of min and max. The default is None.
    g3_cen : list, optional
        list of min and max. The default is None.
    neg_cen : list, optional
        list of min and max. The default is None.
    g1_sig : list, optional
        list of min and max. The default is None.
    g2_sig : list, optional
        list of min and max. The default is None.
    g3_sig : list, optional
        list of min and max. The default is None.
    neg_sig : list, optional
        list of min and max. The default is None.

    Returns
    -------
    out : lmfit model
        lmfit model results.

    '''
    
    gauss1 = GaussianModel(prefix='g1_')
    gauss2 = GaussianModel(prefix='g2_')
    gauss3 = GaussianModel(prefix='g3_')
    gauss4 = GaussianModel(prefix='g4_')
    line1=LinearModel(prefix='line_')
    
    pars_g1 = gauss1.guess(y, x=x)
    pars_line = line1.guess(y, x=x)
    pars_g2 = gauss2.guess(y, x=x)
    pars_g3 = gauss3.guess(y, x=x)
    pars_g4 = gauss4.guess(y, x=x ,negative=True)
    
    if ngauss==1:
        mod = gauss1 + line1
        pars=pars_g1 + pars_line
        #pars['g1_amplitude'].set(min=0)
        #pars['g1_sigma'].set(max=100)

    elif ngauss==2:
        mod = gauss1 + gauss2 + line1
        pars=pars_g1 + pars_g2 + pars_line
        #pars['g1_amplitude'].set(min=0)
        pars['g2_amplitude'].set(min=0)
    
    elif ngauss==3:
        mod = gauss1 + gauss2 + gauss3 + line1
        pars=pars_g1 + pars_g2 + pars_g3 +pars_line
        #pars['g1_amplitude'].set(min=0)
        pars['g2_amplitude'].set(min=0)
        pars['g3_amplitude'].set(min=0)
    
    if neg==True:
        mod += gauss4
        pars += pars_g4
        pars['g4_amplitude'].set(max=0)
    
    if g1_cen != None:
        pars['g1_center'].set(value=(g1_cen[0]+g1_cen[1])/2, min=g1_cen[0], max=g1_cen[1])
    if g2_cen != None and ngauss==2:
        pars['g2_center'].set(value=(g2_cen[0]+g2_cen[1])/2, min=g2_cen[0], max=g2_cen[1])
    if g3_cen != None and ngauss==3:
        pars['g3_center'].set(value=(g3_cen[0]+g3_cen[1])/2, min=g3_cen[0], max=g3_cen[1])
    if neg_cen != None and neg==True:
        pars['g4_center'].set(value=(neg_cen[0]+neg_cen[1])/2, min=neg_cen[0], max=neg_cen[1])


    if g1_sig != None:
        pars['g1_sigma'].set(value=(g1_sig[0]+g1_sig[1])/2, min=g1_sig[0], max=g1_sig[1])
    if g2_sig != None and ngauss==2:
        pars['g2_sigma'].set(value=(g2_sig[0]+g2_sig[1])/2, min=g2_sig[0], max=g2_sig[1])
    if g3_sig != None and ngauss==3:
        pars['g3_sigma'].set(value=(g3_sig[0]+g3_sig[1])/2, min=g3_sig[0], max=g3_sig[1])
    if neg_sig != None and neg==True:
        pars['g4_sigma'].set(value=(neg_sig[0]+neg_sig[1])/2, min=neg_sig[0], max=neg_sig[1])
    
    out = mod.fit(y, pars, x=x, weights = 1/np.std(y))    #use weights to obtain red. chi sq

        
    return out

def gauss_stats(df_av_line,obs,ngauss=1,neg=False,em_row=999,target='temp',
                gof_min=0.2,printout=False,output=False,savefig=False,subplot=False,plot_comps=True,legend=True,
                reject_low_gof=False,reject_line_close=True,g1_cen=None,g2_cen=None,g3_cen=None,
                vred=False,neg_cen=None,title='full',g1_sig=None,g2_sig=None,g3_sig=None,neg_sig=None):
    '''
    

    Parameters
    ----------
    df_av_line : data frame
        subset of average data frame around given emission line, result of get_line_spec().
    obs : str
        obesrvation to use from df_av_line, one of the column names or av_spec, med_sepc.
    ngauss : int, optional
        number of gauss to fit, 1 to 2. The default is 1.
    neg : bool, optional
        whether to force one of the gauss to be negative, for ngauss > 1. The default is False.
    em_row : pd series, optional
        row from em_matches containing matched info for outputs. The default is 999.
    target : str, optional
        target star name, for file saving. The default is 'temp'.
    gof_min : flat, optional
        minumum goodness of fit value to keep lines. The default is 0.2.
    printout : bool, optional
        option to print details of fit to screen. The default is False.
    output : bool, optional
        optiion to plot results of fitting. The default is False.
    savefig : bool, optional
        option to save output plot. The default is False.
    reject_low_gof : bool, optional
        whether to reject fits that do not meet gof_min. The default is False.

    Returns
    -------
    g_fit : lmfit output
        details of the fit.
    x : array
        x values of the fit.
    g_fit.best_fit : array
        y values of best fit.
    line_info : list
        list of parmeters of the fit.

    '''
    
    clight=astropy.constants.c.to('km/s').to_value()

    #if a row from the emmission line matching results table is parsed, take needed values from that, if not, assign values and unknowns from the w0 position
    try:
        line=em_row.obs_wl_air #rest wavelenth used for plot titles, not for 0 vel point
        ele=em_row.element
        sp_num=em_row.sp_num
        J_i=em_row.J_i
        J_k=em_row.J_k
        #w0_vel=rv
        # w0_vel=((em_row.w0 - line)*clight/line)-rv
        SNR=em_row.SNR
    except:
        try:
            line=em_row.ritz_wl_vac #rest wavelenth used for plot titles, not for 0 vel point
            ele=em_row.element
            sp_num=em_row.sp_num
            J_i=em_row.J_i
            J_k=em_row.J_k
            #w0_vel=rv
            # w0_vel=((em_row.w0 - line)*clight/line)-rv
            SNR=em_row.SNR
        except:
            line=em_row
            ele='unk'
            sp_num=0
            J_i='0'
            J_k='0'
            #w0_vel=0
            SNR=0
        

    x=df_av_line['vel'].values
    #y=df_av_line.iloc[:,2].values #change iloc to user input or average 
    y=df_av_line[obs].values #take observation date from function specified input

    flux_scaled=False
    if mean(y) < 1e-5: #for absolute flux units, remove the small order of mag for error calculations in the fitting
        scale_factor=floor(log10(mean(y)))
        y=y/10**scale_factor
        flux_scaled=True
    
    try:
        #y -= min(y) #shift all the lines to be min 0 flux
        g_fit=fit_gauss(x,y,ngauss,neg,g1_cen=g1_cen,g2_cen=g2_cen,g3_cen=g3_cen,neg_cen=neg_cen,
                       g1_sig=g1_sig,g2_sig=g2_sig,g3_sig=g3_sig,neg_sig=neg_sig) #fit the linear model using above function
    except:
        #this is the exception for the fit failing, will just pass
        print(line, obs,'has no data within specified range')
        return None,None,None,None
        
    gof=g_fit.redchi # / np.std(y)**2 #do not need to divide here as it is included in the weights in the fit_gauss() fn
    #note on gof, as window of line increased, gof is larger because more either side of gauss is included, 0.1 is good for 0.6 range in wl, 0.3 good for 1.0 range in wl
    
    y_base=g_fit.best_fit - min(g_fit.best_fit) # determine y values starting from 0 min
    line_values= (g_fit.best_values['line_slope'] * x) + g_fit.best_values['line_intercept']
    y_sub_line=g_fit.best_fit - line_values # remove line component from final fit, not tested with two or more gauss compenents yet. 

    #calculate intergrated flux just from flux above continuum, i.e. subtract line compnent before integrating
    #int_flux=np.round(np.trapz(y_sub_line,x),4)
    int_flux=np.trapz(y_sub_line,x)

    #calculate asym from the intergrated flux above the zero baseline, comparing each side of peak
    #centre_x=closest(x,g_fit.best_values['g1_center'])
    centre_x=closest(x,0) #calculate wrt to 0 velocity rather than g1 centre
    centre_x_idx=np.where(x==centre_x)[0][0]
    centre_x1=closest(x,g_fit.best_values['g1_center'])
    peak_y=float(g_fit.best_fit[centre_x_idx])
    peak_y_base=y_base[centre_x_idx]
    lhs_int_flux=np.trapz(y_sub_line[0:centre_x_idx],x[0:centre_x_idx])
    rhs_int_flux=np.trapz(y_sub_line[centre_x_idx:-1],x[centre_x_idx:-1])
    asym=lhs_int_flux/(lhs_int_flux + rhs_int_flux)
    #asym=lhs_int_flux/(int_flux)
    
    g1_stderr=g_fit.params['g1_center'].stderr
    if (g1_stderr) is None:
        g1_stderr=999#np.nan
    g1_amp_stderr=g_fit.params['g1_amplitude'].stderr
    if g1_amp_stderr is None:
        g1_amp_stderr=999
    
    try:
        dely=g_fit.eval_uncertainty(sigma=3)
    except:
        dely=0
    
    if ngauss==2:
        centre_x2=closest(x,g_fit.best_values['g2_center'])
        g2_stderr=g_fit.params['g2_center'].stderr
        if (g2_stderr) is None:
            g2_stderr=999
    if ngauss==3:
        centre_x2=closest(x,g_fit.best_values['g2_center'])
        g2_stderr=g_fit.params['g2_center'].stderr
        if (g2_stderr) is None:
            g2_stderr=999
        centre_x3=closest(x,g_fit.best_values['g3_center'])
        g3_stderr=g_fit.params['g3_center'].stderr
        if (g3_stderr) is None:
            g3_stderr=999
            
    if neg==True:
        centre_x4=closest(x,g_fit.best_values['g4_center'])
        g4_stderr=g_fit.params['g4_center'].stderr
        if (g4_stderr) is None:
            g4_stderr=999
    depth10_x=0
    if vred==True:
        try:
            y_min=min(g_fit.best_fit[x>0]) #find min of redshifted absorption
            y_min_idx=np.where(g_fit.best_fit==y_min)[0][0] #index of this value
            x_min=float(x[y_min_idx]) #vel of min flux
            line_y_min=line_values[y_min_idx]
            depth=line_y_min-y_min #depth from continuum fit
            depth10=depth*0.1
            vred_max=closest(g_fit.best_fit[x>x_min],line_y_min+depth) #find where the absorption meets the continuum
            vred_idx=np.where(g_fit.best_fit==vred_max)
            vred_max_x=x[vred_idx]
            depth10_y=closest(g_fit.best_fit[(x>x_min) & (x<vred_max_x)],line_y_min-depth10) #find 10% depth that is greater than min flux and less than where the absorption meets continuum
            depth10_y_idx=np.where(g_fit.best_fit==depth10_y)[0][0]
            depth10_x=x[depth10_y_idx]
        except:
            depth10_y=0
            depth10_x=0
        
    #for reject_low_gof==True, also reject lines whose gauss centre are far from ref centre
    #also reject lines where peak value is negative (may have to change this in future for abs lines)
    #if g_fit.values['g1_center'] > w0_vel-10 and g_fit.values['g1_center'] < w0_vel+10 and g_fit.values['g1_fwhm'] < 30 and peak_y > 0:
    if g_fit.values['g1_center'] > min(x) and g_fit.values['g1_center'] < max(x):# and g1_stderr < 900:# and int_flux > 0:# and abs(g_fit.best_values['line_slope']/peak_y)<0.02: #and g_fit.values['g1_fwhm'] < 50
        line_close=True
    elif reject_line_close==False:
        line_close=True
    else:
        line_close=False
        
    if reject_low_gof==True and gof < gof_min and line_close==True or reject_low_gof==False:
        line_info=pd.Series(({'gof':gof,'g1_cen':g_fit.values['g1_center'],'g1_stderr':g1_stderr, 'g1_sigma':g_fit.values['g1_sigma'],
                            'g1_fwhm':g_fit.values['g1_fwhm'],'g1_fwhm_stderr':g_fit.params['g1_fwhm'].stderr,'g1_amp':g_fit.values['g1_amplitude'],'g1_amp_stderr':g1_amp_stderr,
                              'peak':peak_y, 'asym':asym, 'int_flux':int_flux, 'mjd':obs,'Vred':depth10_x}))
        try:
            line_info=pd.concat([line_info,em_row],axis=0)
        except:
            pass

        if ngauss==2 or ngauss==3:
            line_info2=pd.Series(({'g2_cen':g_fit.values['g2_center'],'g2_stderr':g2_stderr,
                                    'g2_fwhm':g_fit.values['g2_fwhm'],'g2_fwhm_stderr':g_fit.params['g2_fwhm'].stderr,
                                   'g2_amp':g_fit.values['g2_amplitude'],'g2_amp_stderr':g_fit.params['g2_amplitude'].stderr}))
            line_info=pd.concat([line_info,line_info2],axis=0)
        if ngauss==3:
            line_info3=pd.Series(({'g3_cen':g_fit.values['g3_center'],'g3_stderr':g3_stderr,
                                    'g3_fwhm':g_fit.values['g3_fwhm'],'g3_fwhm_stderr':g_fit.params['g3_fwhm'].stderr,
                                   'g3_amp':g_fit.values['g3_amplitude'],'g3_amp_stderr':g_fit.params['g3_amplitude'].stderr}))
            line_info=pd.concat([line_info,line_info3],axis=0)
        if neg==True:
            line_info4=pd.Series(({'g4_cen':g_fit.values['g4_center'],'g4_stderr':g4_stderr,
                                    'g4_fwhm':g_fit.values['g4_fwhm'],'g4_fwhm_stderr':g_fit.params['g4_fwhm'].stderr,
                                   'g4_amp':g_fit.values['g4_amplitude'],'g4_amp_stderr':g_fit.params['g4_amplitude'].stderr}))
            line_info=pd.concat([line_info,line_info4],axis=0)
    else:
        line_info=None
    
    pass_gof='N'
    if gof < gof_min and line_close==True:
        pass_gof='Y'
        
    
    if printout==True:
        print(g_fit.fit_report(min_correl=0.25))
        #print('corrected chi^2: %.5f' %(g_fit.redchi / np.std(y)**2))
        #print(np.sum(((y - g_fit.best_fit)**2) / g_fit.best_fit) / (g_fit.nfree))
        #print(np.sum(((y - g_fit.best_fit)**2)/ np.std(y)**2) / (g_fit.nfree))
        #print(np.sum(((y - g_fit.best_fit)**2)/ np.sqrt(np.mean(y**2))**2) / (g_fit.nfree))
        if reject_low_gof==True and gof > gof_min:
            print('GoF too low to produce output / save file')
    
    if reject_low_gof==True and gof < gof_min and line_close==True or reject_low_gof==False:
        if output==True or savefig==True:
            ioff()
            if subplot==True:
                fig, ax = subplots(1, 2)#,figsize=(9,6))#,gridspec_kw={'wspace':0})
                fig.suptitle('%s Fit of line at %.2f Angstroms   Pass GoF:%s \n'  %(obs,line,pass_gof),fontsize=8)
                xlabel("common X")
                ylabel("common Y")
                for x1 in ax:
                    x1.set(xlabel='Velocity (km/s)', ylabel='Flux')
                for x2 in ax:
                    x2.label_outer()
            
                ax[0].set_title('GoF: %.5f, FWHM: %.2f \n Int.Flux: %.4f, Asym: %.4f' %(gof ,g_fit.values['g1_fwhm'], int_flux, asym),fontsize=6)
                ax[0].plot(x,y, 'b--',lw=2,label='Input')
                #ax[0].plot(x, g_fit.init_fit, 'k--', label='initial fit')
                ax[0].plot(x, g_fit.best_fit, 'm-',lw=3, label='Best fit')
                ax[0].axvline(x=centre_x,color='k',linewidth=0.5,linestyle='--')#centre
                #try:
                #    ax[0].axvline(x=x[centre_x_idx-50],color='k',linewidth=0.5,linestyle='--')#asym window
                #    ax[0].axvline(x=x[centre_x_idx+50],color='k',linewidth=0.5,linestyle='--')#asym window
                #except:
                #    pass
                ax[0].legend(loc='upper right')

                #figure(6,figsize=(5,5))
                comps = g_fit.eval_components(x=x)
                ax[1].plot(x, y, 'b')
                ax[1].plot(x, comps['g1_'], 'g--', label='Gauss 1')
                ax[1].plot(x, comps['line_'], 'k--', label='Cont.')
                ax[1].axvline(x=centre_x,color='k',linewidth=0.5,linestyle='--')
                if ngauss==1:
                    ax[1].set_title('Line: %s%s %s-%s, g1_cen= %.1f \n SNR: %.2f, line slope: %.2f ' %(
                    ele,sp_num,J_i,J_k,g_fit.best_values['g1_center'],SNR,abs(g_fit.best_values['line_slope']/peak_y)),fontsize=6)
                if ngauss==2:
                    ax[1].set_title('Line: %s%s %s-%s, g1_cen= %.1f g2_cen=%.1f \n SNR: %.2f, line slope: %.2f ' %(
                    ele,J_i,J_k,g_fit.best_values['g1_center'],g_fit.best_values['g2_center'],SNR,abs(g_fit.best_values['line_slope']/peak_y)),fontsize=6)
                    ax[1].plot(x, comps['g2_'], 'm--', label='Gauss comp. 2')
                    ax[1].axvline(x=centre_x2,color='k',linewidth=0.5,linestyle='--')
                if neg==True:
                    ax[1].plot(x, comps['g4_'], 'c--', label='Neg Gauss comp')
                ax[1].legend(loc='upper right')
            
            elif subplot==False:
                fig, ax = subplots(1,1,figsize=USH.fig_size_s)#,gridspec_kw={'wspace':0})
                if title=='full':
                    fig.suptitle('%s fit of line at %.2f Angstroms,   Pass GoF:%s \n'  %(obs,line,pass_gof),fontsize=10)
                if flux_scaled==False:
                    ax.set(xlabel='Velocity (km/s)', ylabel='Flux')
                else:
                    ax.set(xlabel='Velocity (km/s)', ylabel='Flux x10^(%.0f)'%(scale_factor))
                   
                ax.plot(x,y, 'b-',lw=2,label='Input')
                #ax[0].plot(x, g_fit.init_fit, 'k--', label='initial fit')
                ax.plot(x, g_fit.best_fit, 'r--',lw=2, label='Best fit')
                if plot_comps==True:
                    ax.axvline(x=centre_x1,color='k',linewidth=0.5,linestyle='--')#centre
                ax.fill_between(x,g_fit.best_fit-dely,g_fit.best_fit+dely,color='#ABABAB',label='3-$\sigma$ uncertainty')
                                   
                
                comps = g_fit.eval_components(x=x)
                #ax[1].plot(x, y, 'b')
                if plot_comps==True:
                    ax.plot(x, comps['g1_'], 'g--', label='Gauss comp. 1')
                    ax.axvline(x=centre_x1,color='k',linewidth=0.5,linestyle='--')
                if ngauss==1:
                    if title=='full':
                        ax.set_title('GoF: %.2e, FWHM: %.2f Int.Flux: %.2E, Asym: %.4f \n Line: %s%s %s-%s, g1_cen= %.1f$\pm$%.2f SNR: %.2f' %(
                        gof ,g_fit.values['g1_fwhm'], int_flux, asym,ele,sp_num,J_i,J_k,g_fit.best_values['g1_center'],g1_stderr,SNR),fontsize=8)
                    elif title=='simple':
                        ax.set_title('%s %s %.2f' %(obs,ele,line),fontsize=12)                        
                if ngauss==2:
                    if title=='full':
                        ax.set_title('GoF: %.2e, g1_FWHM: %.2f, Int.Flux: %.2E, Asym: %.2f \n Line: %s%s %s-%s, G1_cen= %.1f$\pm$%.2f, G2_cen= %.1f$\pm$%.2f, SNR: %.2f' %(
                        gof ,g_fit.values['g1_fwhm'], int_flux, asym,ele,sp_num,J_i,J_k,g_fit.best_values['g1_center'],g1_stderr,g_fit.best_values['g2_center'],g2_stderr,SNR),fontsize=8)                   
                    elif title=='simple':
                        ax.set_title('%s %s %.2f' %(obs,ele,line),fontsize=12) 
                    if plot_comps==True:
                        ax.plot(x, comps['g2_'], 'm--', label='Gauss comp. 2')
                        ax.axvline(x=centre_x2,color='k',linewidth=0.5,linestyle='--')
                if ngauss==3:
                    if title=='full':
                        ax.set_title('GoF: %.2e, g1_FWHM: %.2f, Int.Flux: %.2E, Asym: %.2f \n Line: %s%s %s-%s, G1_cen= %.1f$\pm$%.2f, G2_cen= %.1f$\pm$%.2f, G3_cen= %.1f$\pm$%.2f, SNR: %.2f' %(
                        gof ,g_fit.values['g1_fwhm'], int_flux, asym,ele,sp_num,J_i,J_k,g_fit.best_values['g1_center'],g1_stderr,g_fit.best_values['g2_center'],g3_stderr,g_fit.best_values['g3_center'],g3_stderr,SNR),fontsize=8)                   
                    elif title=='simple':
                        ax.set_title('%s %s %.2f' %(obs,ele,line),fontsize=12) 
                    if plot_comps==True:    
                        ax.plot(x, comps['g2_'], 'm--', label='Gauss comp. 2')
                        ax.axvline(x=centre_x2,color='k',linewidth=0.5,linestyle='--')
                        ax.plot(x, comps['g3_'], 'y--', label='Gauss comp. 3')
                        ax.axvline(x=centre_x3,color='k',linewidth=0.5,linestyle='--')
                if neg==True:
                    if plot_comps==True:
                        ax.plot(x, comps['g4_'], 'c--', label='Neg Gauss comp.')
                        ax.axvline(x=centre_x4,color='k',linewidth=0.5,linestyle='--')
                    if vred==True:
                        ax.plot(x_min,y_min,'yo',markersize=12)
                        ax.plot(depth10_x,depth10_y,'ro',markersize=12)
                        if ngauss==1:
                            if title=='full':
                                ax.set_title('GoF: %.2e, FWHM: %.2f Int.Flux: %.2E, Asym: %.4f \n Line: %s%s %s-%s, g1_cen= %.1f, neg_cen= %.1f, SNR: %.2f' %(
                                gof ,g_fit.values['g1_fwhm'], int_flux, asym,ele,sp_num,J_i,J_k,g_fit.best_values['g1_center'],g_fit.best_values['g4_center'],SNR),fontsize=8)
                            elif title=='simple':
                                ax.set_title('%s %s %.2f' %(obs,ele,line),fontsize=12)                        
                        if ngauss==2:
                            if title=='full':
                                ax.set_title('GoF: %.2e, g1_FWHM: %.2f, Int.Flux: %.2E, Asym: %.2f \n Line: %s%s %s-%s, G1_cen= %.1f, G2_cen= %.1f, neg_cen= %.1f, SNR: %.2f' %(
                                gof ,g_fit.values['g1_fwhm'], int_flux, asym,ele,sp_num,J_i,J_k,g_fit.best_values['g1_center'],g_fit.best_values['g2_center'],g_fit.best_values['g4_center'],SNR),fontsize=8)                   
                            elif title=='simple':
                                ax.set_title('%s %s %.2f' %(obs,ele,line),fontsize=12)                 
                if plot_comps==True:
                    ax.plot(x, comps['line_'], 'k--', label='Cont. comp.')
                if legend==True: 
                    ax.legend(fontsize=12)

            if output==True:
                #tight_layout()
                show()
            else:
                close()
            if savefig==True:
                #output dir
                #dirname=os.path.join('output_plots', target+'_'+timenow)
                dirname=os.path.join('output_plots',timenow)                
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                #fig.savefig(os.path.join(dirname,ele+'_'+str(np.round(line,2))+'_'+str(obs)+'.pdf'))#,bbox_inches="tight")
                fig.savefig(os.path.join(dirname,target+'_'+ele+'_'+str(np.round(line,2))+'_'+str(obs)+'.pdf'))#,bbox_inches="tight")
                #print('saving file',os.path.join(dirname,ele+str(np.round(line,2))+'.pdf'))
                if output==False:
                    close()
            ion()
        
    return g_fit,x,g_fit.best_fit, line_info

def get_line_results(em_matches,df_av_norm,line_date_list,target,w_range=0.6,title='full',radvel=USH.radvel,
                     ngauss=1,neg=False,vred=False,gof_min=0.2,reject_low_gof=True,reject_line_close=True,
                     g1_cen=None,g2_cen=None,g3_cen=None,neg_cen=None,g1_sig=None,g2_sig=None,g3_sig=None,neg_sig=None,
                     printout=False,output=False,savefig=False,plot_comps=True):
    '''
    

    Parameters
    ----------
    em_matches : data frame
        results from get_em_lines().
    df_av_norm : data frame
        normalised dataframe of flux values.
    line_date_list : list
        list of observation dates for available data, can use one obs but still within a list
        eg ['med_flux'].
    target : str
        target star name for saving plots.
    w_range : float, optional
        wavelength range passed to get_line_spec(). The default is 0.6.
    ngauss : int, optional
        number of gauss to fit, 1 to 3. The default is 1.
    neg : bool, optional
        whether to force one of the gauss to be negative, for ngauss > 1. The default is False.
    gof_min : float, optional
        min value for keeping fits. The default is 0.2.
    reject_low_gof : bool, optional
        option to only include lines below gof_min. The default is True.
    printout : bool, optional
        print out details of fits to screen. The default is False.
    output : bool, optional
        plot results of fiting on screen. The default is False.
    savefig : bool, optional
        save plot results. The default is False.

    Returns
    -------
    em_line_date_results : data frame
        data frame of gauss line results for all lines and all dates.
    em_line_date_results_common_Ek : data frame
        subset of above for lines originating from common upper every levels.

    '''
    
    #check that list of lines within range of data, good for loaded in lists of lines covering larger ranges
    em_matches=em_matches[em_matches.w0.between(min(df_av_norm.wave),max(df_av_norm.wave))]
    
    if USH.instrument[0]=='XMM' or USH.instrument[0]=='Sol' or USH.instrument[0]=='COS' or USH.instrument[0]=='STIS':
        wave='ritz_wl_vac'
    else:
        wave='obs_wl_air'

    
    print('Fitting lines using',wave,' and radvel=',radvel)
    
    line_results=pd.DataFrame()
    for index,row in em_matches.iterrows():
        line = row[wave]
        df_av_line=get_line_spec(df_av_norm,line,vel_offset=radvel,w_range=w_range,vel=True)
        for date in line_date_list:# ['med_flux']:#[df_av_line.columns[2]]:# df_av_line.columns[1:-3]:
            out,x,y,line_info=gauss_stats(df_av_line,date,ngauss=ngauss,neg=neg,em_row=row,target=target,vred=vred,
                                      gof_min=gof_min,printout=printout,output=output,savefig=savefig,title=title,
                                      reject_low_gof=reject_low_gof,reject_line_close=reject_line_close,plot_comps=plot_comps,
                                          g1_cen=g1_cen,g2_cen=g2_cen,g3_cen=g3_cen,neg_cen=neg_cen,
                                         g1_sig=g1_sig,g2_sig=g2_sig,g3_sig=g3_sig,neg_sig=neg_sig)
            #line_results=pd.concat([line,info,line_results])
            line_results=line_results.append(line_info,ignore_index=True)
    
    try:
        if neg==False:
            if ngauss==3:
                cols_to_move = ['mjd','w0','rv_wl',wave,'element','sp_num','int_flux','asym', 'g1_cen', 'g1_stderr','g2_cen','g2_stderr','g1_fwhm','g2_fwhm','g3_cen', 'g3_stderr','g3_fwhm', 'gof']
                em_line_date_results= line_results[ cols_to_move + [ col for col in line_results.columns if col not in cols_to_move ] ]
            elif ngauss==2:
                cols_to_move = ['mjd','w0','rv_wl',wave,'element','sp_num','int_flux','asym', 'g1_cen', 'g1_stderr','g2_cen','g2_stderr','g1_fwhm','g2_fwhm', 'gof']
                em_line_date_results= line_results[ cols_to_move + [ col for col in line_results.columns if col not in cols_to_move ] ]
            else:
                cols_to_move = ['mjd','w0','rv_wl',wave,'element','sp_num','int_flux','asym', 'g1_cen', 'g1_stderr','g1_fwhm', 'gof']
                em_line_date_results= line_results[ cols_to_move + [ col for col in line_results.columns if col not in cols_to_move ] ]
        else:
            if ngauss==3:
                cols_to_move = ['mjd','w0','rv_wl',wave,'element','sp_num','int_flux','asym', 'g1_cen', 'g1_stderr','g2_cen','g2_stderr','g1_fwhm','g2_fwhm','g3_cen', 'g3_stderr','g3_fwhm','g4_cen', 'g4_stderr','g4_fwhm', 'Vred', 'gof']
                em_line_date_results= line_results[ cols_to_move + [ col for col in line_results.columns if col not in cols_to_move ] ]
            elif ngauss==2:
                cols_to_move = ['mjd','w0','rv_wl',wave,'element','sp_num','int_flux','asym', 'g1_cen', 'g1_stderr','g2_cen','g2_stderr','g1_fwhm','g2_fwhm','g4_cen', 'g4_stderr','g4_fwhm', 'Vred', 'gof']
                em_line_date_results= line_results[ cols_to_move + [ col for col in line_results.columns if col not in cols_to_move ] ]
            else:
                cols_to_move = ['mjd','w0','rv_wl',wave,'element','sp_num','int_flux','asym', 'g1_cen', 'g1_stderr','g1_fwhm','g4_cen', 'g4_stderr','g4_fwhm', 'Vred','gof']
                em_line_date_results= line_results[ cols_to_move + [ col for col in line_results.columns if col not in cols_to_move ] ]   
        print('total number of em lines fit:',len(em_line_date_results))
        print('number of observed em lines fit:',len(unique(em_line_date_results.w0)))
    except:
        print('---no lines fit---')
        em_line_date_results=0
        
    #em_line_date_results=line_results
        
    # # find em lines that are from the same upper energy level that were fitted
    try:
        em_line_date_results_no_dup=em_line_date_results.drop_duplicates(subset='w0')
        check_Ek=np.column_stack(np.unique(em_line_date_results_no_dup.Ek,return_counts=True))
        common_Ek=check_Ek[check_Ek[:,1]>1][:,0]
        em_line_date_results_common_Ek=em_line_date_results_no_dup[em_line_date_results_no_dup.Ek.isin(common_Ek)]
        print('number of em lines fit from same upper energy level:',len(em_line_date_results_common_Ek))
    except:
        em_line_date_results_common_Ek=0
        print('no lines fit from same upper energy level')
        
    if savefig==True:
        print('saving files to output dir')

    
    return em_line_date_results,em_line_date_results_common_Ek

    
def periodogram_indiv(line_results,method='auto',cen_cor=False,plot_title=False):
    #mjd=Time(em_line_dates.mjd,format='mjd')
    date_range=max(line_results.mjd)-min(line_results.mjd)
    mjd=line_results.mjd
    fig, ax = subplots(2, 1)#,figsize=(9,6))#,gridspec_kw={'wspace':0})
    if plot_title==True:
        fig.suptitle('%s %.0f line at %.2f Angstroms'%(line_results.element.any(),line_results.sp_num.values[0],line_results.obs_wl_air.values[0]),fontsize=14)

    centre=line_results.g1_cen
    ax[0].plot(mjd,centre,'b.',label='Observed')

    ax[0].set_xlabel('MJD')
    ax[0].set_ylabel('Line centre [km/s]')
    ls=LombScargle(mjd,centre)
    frequency,power=ls.autopower(method=method)
    ax[1].plot(1/frequency, power)
    ax[1].set_xlabel('Period [day]')
    ax[1].set_xlim([0,date_range])
    ax[1].set_ylabel('Power')    
    tight_layout()
    print('max power:',power.max())
    print('F.A.P at max power:',ls.false_alarm_probability(power.max()))
    print('power req. for F.A.P of 50%,10%:',ls.false_alarm_level([0.5,0.1]))


def phase_period(em_line_date_results,linewav,mjd0,period=17,gofmin=0.2,filmin=-20,filmax=20,
                maxper=100,minper=0,errmin=100,mjmin=0,mjmax=600000):
    
    #At present, the program does the wrapping with the period you give by hand
    #It also calculates the period by itself and prints some info on it.
    #It does not take the period from the periodogram, but the one you provide, for the plots
    #The line fit is also done for that period. This allows for instance to wrap a line to check
    #around one period and get the fit even if that line does not show a significant period in terms of
    #FAP. It also allows to, once you know the period, fit the lines between two MJD to check
    #if the fit (phase, amplitude, offset) changes in time.
    period=period  #period to check
    gofmin=gofmin #filter the velo
    filmin=filmin #only velos between those limits
    filmax=filmax 
    maxper=maxper #if you want to find a period below a given limit
    minper=minper #the other limit to the period to avoid the 1d thingy if that gives problems
    mjd0=mjd0 #initial date, use the same for all lines from one source or phases will be odd!
    errmin=errmin #maximum error to use value 
    mjmin=mjmin #minimum mjd for selection
    mjmax=mjmax	#maximum mjd for selection

    mjd=em_line_date_results.mjd[em_line_date_results.obs_wl_air==linewav]
    velo=em_line_date_results.g1_cen[em_line_date_results.obs_wl_air==linewav]
    veloerr=em_line_date_results.g1_stderr[em_line_date_results.obs_wl_air==linewav]
    gof=em_line_date_results.gof[em_line_date_results.obs_wl_air==linewav]
    ele=em_line_date_results.element[em_line_date_results.obs_wl_air==linewav].any()
    linename=ele+' '+str(int(linewav))  # line name to use it for the figure labels


    # -----------------------------------------------------------------------------------------------

    #Function to fit
    #RV=V0+A*sin(phi-phi0) 
    #where V0 is the velo offset, A is the amplitude (Vsini*sin(theta_spot)) 
    #phi0 is the phase origin, phi the phase

    def rv(x,v0,a0,phi0):
        return v0+abs(a0)*sin(2.*np.pi*(x-phi0))

    #initial params
    pini=[0.,2.,0.]

    # -----------------------------------------------------------------------------------------------

    phaseu=numpy.mod(mjd/period,1.0)

    phasetime=floor((mjd-mjd0)/period) #color with respect to first period

    approxer=gof

    #filter data according to gof, velo limits (use to exclude crazy values), max error, and mjd
    fil=(gof<gofmin) & (velo>filmin) & (velo<filmax) & (veloerr<errmin)  & (mjd>mjmin) & (mjd<mjmax)
    phaseu=phaseu[fil]
    velo=velo[fil]
    veloerr=veloerr[fil]
    phasetime=phasetime[fil]
    mjdu=mjd[fil]
    
    fig = figure()
    #fig.set_size_inches(12, 8, forward=True)
    subplots_adjust(left=0.15, bottom=0.12, right=0.98, top=0.97, wspace=0.25, hspace=0.25)
    #ax=[-0.1, 2.1,-2, 2]
    x=[-0.5,2.5]
    y=[0,0]
    # -----------------------------------------------------------------------------------------------

    errorbar(phaseu,velo,yerr=veloerr, fmt='',ecolor='k', alpha=0.5, elinewidth=1,linewidth=0)
    errorbar(phaseu+1,velo,yerr=veloerr, fmt='',ecolor='k', alpha=0.5, elinewidth=1,linewidth=0)

    #scatter(phaseu, velo,s=250,c=phasetime, marker = 'o', edgecolor='none', alpha=0.8, cmap= cm.terrain,vmin=min(phasetime),vmax=max(phasetime))
    #scatter(phaseu+1, velo,s=250,c=phasetime, marker = 'o', edgecolor='none', alpha=0.8, cmap= cm.terrain,vmin=min(phasetime),vmax=max(phasetime))

    #phasetime=np.log(abs(phasetime))

    scatter(phaseu, velo,s=150,c=phasetime, marker = 'o', edgecolor='none', alpha=0.7, cmap= cm.gist_rainbow,vmin=min(phasetime),vmax=max(phasetime))
    scatter(phaseu+1, velo,s=150,c=phasetime, marker = 'o', edgecolor='none', alpha=0.7, cmap= cm.gist_rainbow,vmin=min(phasetime),vmax=max(phasetime))

    # -----------------------------------------------------------------------------------------------
    #plot and fit the curve
    pout,cova=curve_fit(rv,phaseu,velo,pini)
    #plot fit
    xx=arange(-0.5,2.5,0.01)
    yy=rv(xx,pout[0],pout[1],pout[2])
    plot(xx,yy,'k:', linewidth=3, alpha=0.5)

    #Because this gives a complete turn when x goes from 0-2pi, units of phi0 are "phase" and not degrees.
    #Therefore, the total angle offset is 2*pi*phi0 in radians.
    #Thus to convert the phase into degrees I will need to do 2*pi*phi0*180/pi = 360*phi0 	

    #Get the uncertainty from the covariance matrix, I assume correlated part negligible
    print('Fit; for the period given in argument',period,'d')
    print('Offset',pout[0],'+-',sqrt(cova[0,0]), 'km/s')
    print('Amplitude',pout[1],'+-',sqrt(cova[1,1]), 'km/s')
    print('Phase',pout[2]*360.,'+-',sqrt(cova[2,2])*360, 'degrees')
    # -----------------------------------------------------------------------------------------------
    #Do Lomb Scargle
    perio = np.linspace(1.3,700, 100000)
    freq= 1 / perio

    #use median errors to avoid issues with LSP
    veloerr0=np.ones([np.size(velo)])*np.median(veloerr)

    ls=LombScargle(mjdu,velo,veloerr0) #.power(freq)
    f,ls=LombScargle(mjdu,velo,veloerr0).autopower(minimum_frequency=min(freq),maximum_frequency=max(freq),samples_per_peak=50)
    autoperio=1/f

    #plot(1/freq, ls)
    ls0=LombScargle(mjdu,velo,veloerr0)
    f0,p0=ls0.autopower(minimum_frequency=min(freq),maximum_frequency=max(freq),samples_per_peak=50)
    #plot(1/f0,p0, alpha=0.2)
    fap= ls0.false_alarm_probability(max(ls),method='baluev')
    print('Line velocity periodicity:\n Estimated fap=', fap)
    print(' For period:', autoperio[np.argmax(ls)],'d \n')
    print(' Nr of datapoints:', np.size(velo),'\n')
    level99=ls0.false_alarm_level(0.001)
    #a=[min(perio),max(perio)]
    #b=[level99,level99]
    #plot(a,b,'k-',alpha=0.2)

    #For period limit too, that tells me the significance of any other point I see by eye
    #or to help getting rid of annoying features like the 1d period or the long period bump.

    #fli=(perio<maxper)
    fli=(autoperio<maxper) & (autoperio>minper)
    lslim=ls[fli]
    autoperiolim=autoperio[fli]
    faplim= ls0.false_alarm_probability(max(lslim),method='baluev')
    print('Best period within limits', minper, '-', maxper, 'd')
    print('Line velocity: Estimated fap=', faplim)
    print('For period:', autoperiolim[np.argmax(lslim)],'d \n')
    # -----------------------------------------------------------------------------------------------

    #legend(loc=4, fontsize=20)
    #legend(loc='upper left', fontsize=15)
    ax=[-0.1,2.1,min(velo)-0.5,max(velo)+0.5]

    ytext='V (km/s) for '+ linename
    xtext='Phase (for a ' + str(round(period,3)) + 'd period)'
    #xlabel ('Phase (for a 7.41d period)')
    xlabel(xtext)
    ylabel (ytext)
    axis(ax)
    show()

    peri=re.sub('\.','p',str(round(period,3)))
    linename2=re.sub(' ','',sys.argv[2])
    perithing='_p'+ peri + '_' + linename2 +  '_gof_'+ str(round(gofmin,1))+'_mjd_' + str(round(average(mjdu))) + '_wrapped.png'
    #namefig=re.sub('.csv',perithing,filename)

    #savefig(namefig)
    ####

    fig = figure()
    #fig.set_size_inches(12, 8, forward=True)
    #matplotlib.rc('font', family='serif',size=20)
    subplots_adjust(left=0.13, bottom=0.12, right=0.98, top=0.97, wspace=0.25, hspace=0.25)

    perithing2='_p'+ peri + '_' + linename + '_gof_'+ str(round(gofmin,1)) + '_mjd_' + str(round(average(mjdu))) + '_GLS.png'
    #linename2 to avoid gap in filename
    #namefig2=re.sub('.csv',perithing2,filename)

    #mark the rotational period being checked
    xx=autoperio[np.argmax(ls)]  
    x=[xx,xx]
    y=[0,1]
    plot(x,y,'k-',linewidth=5,alpha=0.2)

    #plot(1/freq, ls)
    plot(1/f,ls,linewidth=2)
    #plot(1/f0,p0, 'r-',alpha=0.2)

    #Limit to plot this is a bit arbitrary but zoomed for the features we see so far in stars
    ax=[min(perio),45,0,max(ls)+0.1]
    #ax=[min(perio),45,0,1]#max(ls)+0.1]
    axis(ax)
    powertext='Power for ' + linename 
    ylabel(powertext)
    xlabel('Period (d)')
    #savefig(namefig2)

    show()
    

def bary_corr(mjd_insts,simbad_table,observatory='lasilla'):
    
    coords=SkyCoord(simbad_table['RA'][0] +' '+ simbad_table['DEC'][0],unit=(u.hourangle, u.deg))
    location=EarthLocation.of_site(observatory)
    
    mjd_table=mjd_insts[mjd_insts.inst=='FEROS']
    mjd_table.reset_index(drop=True,inplace=True)
    
    bary_corr_list=[]
    bary_corr_list1=[]
    for mjd in mjd_table['mjd']:
        obs_time=Time(mjd,format='mjd')
        bary_corr,HJD=pyasl.helcorr(location.lon.deg,location.lat.deg,location.height.value,
                  coords.ra.deg,coords.dec.deg,obs_time.jd)
        heli_corr,bary_corr1=pyasl.baryCorr(obs_time.jd,coords.ra.deg,coords.dec.deg)
        bary_corr_list.append(bary_corr)
        bary_corr_list1.append(bary_corr1)
    
    mjd_bary=pd.DataFrame({'mjd':mjd_table['mjd'],
                           'DRS':mjd_table['bary'],
                           'bary_cor':bary_corr_list,
                            'diff':mjd_table['bary']-bary_corr_list})

    return mjd_bary

def apply_bary_cor_FEROS(mjd_insts,simbad_table,em_line_date_results):
    
    if 'av_flux' in str(em_line_date_results.mjd.values) or 'med_flux' in str(em_line_date_results.mjd.values):
        print('ERROR: cannot apply barycentric correction to average flux')
        return None
    
    bary=bary_corr(mjd_insts,simbad_table)
    em_line_date_results_bary=em_line_date_results.copy()
    
    for i in range(len(bary)):
        em_line_date_results_bary.loc[isclose(bary['mjd'][i],em_line_date_results_bary['mjd']),'g1_cen']=em_line_date_results_bary[isclose(bary['mjd'][i],em_line_date_results_bary['mjd'])].g1_cen+bary['diff'][i]
        try:
            em_line_date_results_bary.loc[isclose(bary['mjd'][i],em_line_date_results_bary['mjd']),'g2_cen']=em_line_date_results_bary[isclose(bary['mjd'][i],em_line_date_results_bary['mjd'])].g2_cen+bary['diff'][i]
        except:
            pass
        try:
            em_line_date_results_bary.loc[isclose(bary['mjd'][i],em_line_date_results_bary['mjd']),'Vred']=em_line_date_results_bary[isclose(bary['mjd'][i],em_line_date_results_bary['mjd'])].Vred+bary['diff'][i]
        except:
            pass
        
    return em_line_date_results_bary


