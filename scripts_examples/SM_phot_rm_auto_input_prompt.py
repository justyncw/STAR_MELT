#####
#STAR-MELT PHOT RM auto script
#This script performs automatic removal of photospheric features from stellar spectra.
#It loads the data, filters it based on instrument and wavelength range, and performs the removal
#using templates. The results are saved to a CSV file in a specified output directory.
#This script is designed to be run with user interaction for confirmation of parameters.
#It is part of the STAR-MELT package, which is used for analyzing stellar spectra
#Justyn Campbell-White


#Packages required by STAR-MELT, some are not used directly in this script, but called by the modules, check that they are all installed here
import time
from matplotlib import *
from matplotlib.pyplot import *
import numpy as np
import pandas as pd
import astropy
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.stats import sigma_clip
from astroquery.simbad import Simbad
from astropy.timeseries import LombScargle
from astropy.table import Table
import numpy.ma as ma
import os
from PyAstronomy import pyasl
from lmfit.models import GaussianModel, LinearModel
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit 
from scipy.signal import savgol_filter

from star_melt import * #import all modules from star_melt package





rcParams.update({'figure.max_open_warning': 50})
rcParams['figure.dpi'] = 100
matplotlib.rc('font', family='sans',size=14)
USH.fig_size_s=(6,5)
USH.fig_size_l=(9,5)
USH.fig_size_n=(9,3)

line_table=USH.line_table
line_table=USH.line_table_prev_obs
spt_teff=USH.spt_teff


template_dir='/Users/jcampbel/Library/CloudStorage/OneDrive-ESO/standard_stars/'
data_dates_range_templ = USH.load_phot_templates(template_dir)


data_dir='/Users/jcampbel/Library/CloudStorage/OneDrive-ESO/PEN_data_final_copy_180923'
#data_dir='change_me'

star_list=os.listdir(data_dir)
try:
    star_list.remove('.DS_Store')#remove temp file on mac because python will think it's a star!
except:
    pass

print('Data available for following regions/directories')
print(star_list)

region=input('select region:')


data_fits_files=get_files(os.path.join(data_dir,region),'.fits','.FTZ')

data_dates_range2,instrument,w0=get_instrument_date_details(data_fits_files,qgrid=True)
data_dates_range2.sort_values('wmin',inplace=True)


spts=[]
for target in data_dates_range2.target:
    #some common simbad query issues for getting sp_t estimate of target stars
    target=target.replace('O-','O')
    target=target.replace('EM','')
    target=target.replace('YL','Y L')
    #target=target.replace('SO','HHM2007 ')
    if target.startswith('V '):
        target=target.replace('V ', '', 1)
    if target.startswith('SO'):
        target=target.replace('SO','HHM2007 ')
    try:
        simbad=customSimbad.query_object(target)
        mk_tar=simbad['sp_type'][0]
        if mk_tar.startswith('d'):
            mk_tar = mk_tar[1:]#remove 'd' from start of sp_t
        if len(mk_tar) >= 3 and mk_tar[2] == '.':
            mk_tar = mk_tar[:4]
        else:
            mk_tar = mk_tar[:2]
        if target=='DI Cha':
            mk_tar='K0'
    except:
        mk_tar='K9'
        pass    
    spts.append(mk_tar)
data_dates_range2['sp_t']=spts

print('loading data from:',os.path.join(data_dir,region),'\n with ',len(data_dates_range2),'target .fits files')



print('loading all templates for removal...')
w_min=3770#data_dates_range_templ_sel.wmin.values[0]*10
w_max=7900#data_dates_range_templ_sel.wmax.values[0]*10
w_step=0.01
w0_templ=np.arange(w_min,w_max,w_step)
templ_av=get_av_spec(data_dates_range_templ,w0_templ,norm=False,output=False,plot_av=False,savefig=False,label='spt')
templ_av_norm=get_av_spec(data_dates_range_templ,w0_templ,norm=True,output=False,plot_av=False,label='spt')
obs_templ_list=templ_av.columns[1:-3]
print('done')





inst_select_tar = ['UVES','ESPRESSO']

line_sel=float(input('select line centre for phot.rm.'))
range_sel=float(input('select range +/- for phot.rm.'))
mask_sel=float(input('select wave range to mask around line centre'))

rv_cen_sel=float(input('select line centre for RV calc wrt to template'))

spt_sel=1

filename_sel=input('filename for output summary file (incl. .csv)')



print(f"<<< >>>")
print(f"<<< >>>")
print(f"About to run full auto removal for line: '{line_sel}' with range +/-: '{range_sel}' and centre mask +/-'{mask_sel}' ")
print(f"RV calc centre at '{rv_cen_sel}' +/- 100 ")
print(f"Current output filename is '{filename_sel}' ")
confirmation = input(f"Is this correct? (y/yes): ").lower()
if confirmation not in ['y', 'yes']:
    raise ValueError("Script execution stopped. Please confirm the filename.")

line=line_sel
width_pm=range_sel



if 'any' in inst_select_tar:
    data_dates_range1 = data_dates_range2
else:
    data_dates_range1 = data_dates_range2[data_dates_range2['inst'].isin(inst_select_tar)]

#remove the non TELL CORR UVES data...
if data_dates_range1['file'].str.contains('tell.fits').any():
    data_dates_range1 = data_dates_range1[~((data_dates_range1['inst'] == 'UVES') & (data_dates_range1['file'].str.contains('REDU.fits')))]
    data_dates_range1 = data_dates_range1[~((data_dates_range1['inst'] == 'UVES') & (data_dates_range1['file'].str.contains('redu.fits')))]

sub_params=pd.DataFrame()

#for index, row in data_dates_range1.head(3).iterrows(): #for checking the loop before running it on everything, can change to e.g. head(2) for first two starrs in list
for index, row in data_dates_range1.iterrows():
    try:
        data_dates_range5=pd.DataFrame([row])
        USH.target=data_dates_range5.target.values[0]
        USH.instrument=data_dates_range5.inst.values[0]
        target_inst=USH.instrument
        w_min=data_dates_range5.wmin.values[0]*10
        w_max=data_dates_range5.wmax.values[0]*10
        w_step=0.01
        if (w_min < line) & (w_max > line):
            w0=np.arange(w_min,w_max,w_step)
            df_av=get_av_spec(data_dates_range5,w0,norm=False,output=False,plot_av=False,savefig=False)#,label='utc_inst')
        
            df_line=get_line_spec(df_av,line,width_pm,norm=True,full_norm=False,cont_sub=False)
            obs_target=df_line.columns[1]
    
            spt_code_target=spt_coding(data_dates_range5.sp_t.values[0])
            
        
            #select template (to be looped)
            for obs_templ in obs_templ_list:#[0:2]:
    
                templ_spt=obs_templ.split('_-_')[0]
                templ_name=obs_templ.split('_-_')[1]
                templ_inst=obs_templ.split('_-_')[-1]
                spt_code_template=spt_coding(templ_spt)
    
                if abs(spt_code_target - spt_code_template) <= spt_sel:
                    #read in template fits file for RV/vsini calculation
                    data_dates_range_templ_sel2=data_dates_range_templ[data_dates_range_templ.target==templ_name]
                    st_info,st_wave,st_flux,st_err=read_fits_files(data_dates_range_templ_sel2.file.iloc[0],verbose=True)
                    #calculate rv and vsini wrt to template
                    rv_wl_min=rv_cen_sel-100
                    rv_wl_max=rv_cen_sel+100

                    if USH.inst_res[templ_inst] > USH.inst_res[target_inst]:
                        e_res=effective_res(USH.inst_res[templ_inst],USH.inst_res[target_inst])
                    else:
                        e_res=None
                    
                    templ_rv=data_dates_range_templ_sel2.RV.values[0]

                    radvel_t,vsini_t=get_rv_vsini(df_av,st_wave,st_flux,st_rv=templ_rv,date=obs_target,adj_templ_res=e_res,
                                                                        w_min=rv_wl_min,w_max=rv_wl_max,vsini_max=80,output=False)
                    df_line_templ=get_line_spec(templ_av,line,width_pm,norm=True,full_norm=False,cont_sub=False) 
                    #run phot.sub. with best fit values
                    df_av_sub,params=subtract_templ(df_line,obs_target,target_inst,df_line_templ,obs_templ,templ_inst,rv_templ=templ_rv,
                                                    rv_shift=radvel_t,vsini=vsini_t,r=0,fs=USH.fig_size_l,plot_x=[],mask_pm=[None,mask_sel],
                                                 shift=0,plot_subtracted=True,plot_divided=False,
                                                    return_params=True,auto_r=True,auto_vsini=True,chi_output=False,
                                                   output=False,savefig=True,savefits=True,localdirsave=False)
                    sub_params = pd.concat([sub_params, pd.DataFrame([params])], ignore_index=True)
    except:
        continue

dirname='sub_params'
filename=filename_sel #'loop_tests.csv'###########################
timenow_bu=time.strftime("%d_%b_%Y_%H_%M", time.gmtime())
if os.path.exists(os.path.join(dirname,filename)):
    os.system('cp %s backups/%s'%(os.path.join(dirname,filename),timenow_bu+filename))
sub_params.to_csv(os.path.join(dirname,filename), mode='a', header=not os.path.exists(os.path.join(dirname,filename)),index=False)
print('saving results to output file: ',filename)
print(len(sub_params),' removals performed')















