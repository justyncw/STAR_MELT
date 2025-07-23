#####
#STAR-MELT continuum flux measurement script
#This script measures the continuum flux in a specified region of the spectrum for multiple targets.
#It loads the data, filters it based on instrument and wavelength range, and calculates the mean flux in the specified continuum region.
#Plots are generated for each target showing the continuum region and the mean flux.
#The results are saved to a CSV file in a specified output directory.
#This script is designed to be run autonomously, without user interaction during execution.
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



rcParams['figure.dpi'] = 100
matplotlib.rc('font', family='sans',size=14)
USH.fig_size_s=(6,5)
USH.fig_size_l=(9,5)
USH.fig_size_n=(9,3)



data_dir='/Users/jcampbel/Library/CloudStorage/OneDrive-ESO/PEN_data_final_copy_180923/ChaI'

star_list=os.listdir(data_dir)
try:
    star_list.remove('.DS_Store')#remove temp file on mac because python will think it's a star!
except:
    pass
print(f"<<< >>>")
print('Data available for following targets/regions')
print(star_list)
print(f"<<< >>>")

#region=input('select region:')

#data_fits_files=get_files(os.path.join(data_dir,region),'.fits','.FTZ')
data_fits_files=get_files(data_dir,'.fits','.FTZ')

data_dates_range2,instrument,w0=get_instrument_date_details(data_fits_files,qgrid=True)
print(f"<<< >>>")
print('loading data from:',os.path.join(data_dir),'\n with ',len(data_dates_range2),'target .fits files')
print(f"<<< >>>")

inst_select_tar = ['XSHOOTER']
print(f"<<< >>>")
print('using instrument selection:', inst_select_tar)
print(f"<<< >>>")

#cont_region_sel=float(input('select continuum region'))
cont_region_sel=4020
print(f"<<< >>>")
print('using continuum region:', cont_region_sel)

cont_range = 20  # Define the range around the continuum region (±10 units)
print(f"<<< >>>")
print('using continuum range: ±', cont_range, 'A')

save_plots=True
print(f"<<< >>>")
print('save plots:', save_plots)

#filename_sel=input('filename for output summary file (incl. .csv)')
filename_sel='xs_cont_measures.csv'
print(f"<<< >>>")
print('filename for output summary file:', filename_sel)

print(f"<<< >>>")


if 'any' in inst_select_tar:
    data_dates_range1 = data_dates_range2
else:
    data_dates_range1 = data_dates_range2[data_dates_range2['inst'].isin(inst_select_tar)]
    # Filter rows where wmin and wmax cover the cont_region_sel region (in same units as wmin/wmax)
    # Assuming wmin/wmax are in units of 1000s (e.g., Angstroms), so cont_region_sel/10
wmin_check = (data_dates_range1['wmin'] <= cont_region_sel / 10)
wmax_check = (data_dates_range1['wmax'] >= cont_region_sel / 10)
data_dates_range1 = data_dates_range1[wmin_check & wmax_check]

print(f"<<< >>>")
print('measuring continuum flux for data with wavelength range covering the continuum region:', cont_region_sel, 'A')
print('number of targets with this wavelength range:', len(data_dates_range1))

#remove the non TELL CORR UVES data...
if data_dates_range1['file'].str.contains('tell.fits').any():
    data_dates_range1 = data_dates_range1[~((data_dates_range1['inst'] == 'UVES') & (data_dates_range1['file'].str.contains('REDU.fits')))]
    data_dates_range1 = data_dates_range1[~((data_dates_range1['inst'] == 'UVES') & (data_dates_range1['file'].str.contains('redu.fits')))]


dirname = 'cont_measures'
ensure_dir(dirname) #make sure the output directory exists
print(f"<<< >>>")
print('output directory:', dirname)



cont_list = []
w0=np.arange(cont_region_sel - 100, cont_region_sel + 100, 0.1)  # Define w0 around the cont_region_sel
for index, row in data_dates_range1.iterrows():
    # Load spectrum for this row only
    df_av = get_av_spec(pd.DataFrame([row]), w0, norm=False, output=False, plot_av=False, savefig=False)
    # Extract region around cont_region_sel 
    df_line = get_line_spec(df_av, cont_region_sel, w_range=cont_range) 
    mean_flux = df_line['med_flux'].mean()
    if save_plots:
        # Save the plot for this target
        plot_filename = f"{row['target']}_cont_region_{cont_region_sel}.png"
        save_path = os.path.join(dirname, plot_filename)
        ax = df_line.plot(x='wave', y='med_flux', title=f"Continuum Region {cont_region_sel} A for {row['target']}", xlabel='Wavelength (A)', ylabel='Flux')
        ax.axhline(mean_flux, color='k', linestyle='--', label=f"Mean Flux = {mean_flux:.3e}")
        ax.legend()
        savefig(save_path)
        close()
    target = row['target']
    mjd = row['mjd']
    cont_list.append([target, mjd, mean_flux])

cont_df = pd.DataFrame(cont_list, columns=["star", "mjd", "mean_flux"])


filename = filename_sel
cont_df.to_csv(os.path.join(dirname, filename), mode='a', header=not os.path.exists(os.path.join(dirname, filename)), index=False)
print('saving results to output file: ', filename)
print(len(cont_df), ' continuum flux measurements saved to file: ', os.path.join(dirname, filename))















