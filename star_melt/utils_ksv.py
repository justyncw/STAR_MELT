"""
Kyara Soto Villarreal, ESO 2024
"""
#Packages required by STAR-MELT, some are not used directly in this notebook, but called by the modules, check that they are all installed here
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
import numpy.ma as ma
import os
from PyAstronomy import pyasl
from lmfit.models import GaussianModel, LinearModel
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.signal import find_peaks_cwt
from scipy.signal import argrelextrema
from star_melt.ESO_fits_get_spectra import *
from star_melt.ESP_fits_get_spectra import *
from star_melt.utils_data import *
from star_melt.utils_spec import *
from star_melt.utils_physics import *
import star_melt.utils_shared_variables as USH
#from star_melt.utils_saha_av import *
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widget
from IPython.display import display,clear_output
import ipympl
import qgridnext as qgrid
import seaborn as sns

# ---------- functions -----------

def read_id_lines(saved_lists_dir,list_select_v):
    """
    In
    saved_list_dir = (str/path) - directory
    list_select_v = (str/path) - list_select.value if it is run following the notebook
    (you can also write the name of the file if the notebook and the file are in the same folder)
    ---
    Out
    (df) -List of lines identified in the object
    """
    dfp = pd.read_csv(os.path.join(saved_lists_dir,list_select_v))
    dfp = dfp.loc[:, ~dfp.columns.str.contains('^Unnamed')]
    dfp = dfp.drop_duplicates()    
    #----------- get el_wl
    sp = list(dfp["sp_num"])
    element = list(dfp["element"])
    obs_wl_air = list(dfp["obs_wl_air"])
    joinname =  [f"{a}{b} {c}" for a, b, c in zip(element, sp, obs_wl_air)]
    dfp = dfp.assign(el_wl = joinname)
    #---------- get g
    g_k = list(dfp["g_k"])
    g_i = list(dfp["g_i"])
    g_weight = [(a/b) for a, b in zip(g_k, g_i)]
    dfp = dfp.assign(g = g_weight)
    return dfp

def join_fl(join_select_v,saved_lists_dir_join):
    """
    In
    join_select_v - list of files you want to join
    ---
    Out 
    List of forbidden lines across the objects selected 
    """
    join_list = list(join_select_v)
    t = len(join_select_v.value) - 1 
    print("Joining ...")
    
    line_table=USH.line_table
    line_table=USH.line_table_prev_obs
    df_all =line_table["obs_wl_air"]
    
    for i in join_list: 
        print(i)
        df = read_id_lines(saved_lists_dir_join, i)
        df = df.dropna(subset=["Type"])
        df = df[["obs_wl_air","element"]]
        df = df.rename(columns={"element" : i})
        df_all = pd.merge(df_all, df, how="outer", on=["obs_wl_air"])
        df_comp = df_all
    df_comp = df_comp.dropna(thresh=t)
    return df_comp


def absolute_difference(a, b):
    """
    Method for the correlation plots
    """
    return np.abs(a - b).sum().round(decimals=1)

def fl_corr(dfp,wlwin=None,r=0):
    """
    In
    dfp - (df) List of identified lines for an object
    --- filter - take only the lines that are close to a specific velocity 
    wlwin - (int/float) differential velocity 
    r - (int/float) range 
    ---
    Out 
    (plot) - Correlation plot comparing the differential velocities of forbidden lines
    """
    if wlwin != None:
        df_wlwin = dfp.loc[(dfp['vel_diff'] >= wlwin-r) & (dfp['vel_diff'] <=wlwin+r)]
        dfp = df_wlwin
    else:
        dfp = dfp
    dfp = dfp.dropna(subset=["Type"])
    dfp = dfp[["el_wl","vel_diff"]]
    dfp= dfp.set_index("el_wl")
    dfp= dfp.T
    dfp= pd.concat([dfp]*(dfp.size), ignore_index=True)
    sns.heatmap(dfp.corr(method=absolute_difference), cmap=sns.cubehelix_palette(as_cmap=True))
    plt.show()

def el_vel_corr(dfp,el_list_select_v):
    """
    In
    dfp - (df) List of identified lines for an object
    el_list_select_v = (str) el_list_select.value if you are following the notebook
    you can also provide the element with a string
    Out
    (plot) - Correlation plot comparing the differential velocities of an element
    """
    dfp = dfp[["element","vel_diff","el_wl"]]
    el = el_list_select_v
    dfp = dfp[dfp.element==el]
    dfp= dfp.drop(columns=['element'])
    dfp= dfp.set_index("el_wl")
    dfp= dfp.T
    dfp= pd.concat([dfp]*(dfp.size), ignore_index=True)
    sns.heatmap(dfp.corr(method=absolute_difference), cmap="crest")
    plt.show()

def fl_likeliness(aki_table,fl,cm,vel):
    """
    In
    aki_table - (df) List of the lines identified at the same wavelenght 
                as the selected forbidden line
    cm - (str) Colormap parameters
    vel - (int/float) velocity of interest 
    Out
    (plot) - Annotated plot with differential velocity on the x-axis
             g, which is g_k/g_i, is the y-axis 
             the colormap is a customazible parameter
             the vertical line is the velocity provided
    """
    p = "g"
    axvline(vel, color="k", linestyle="--")

    temp = aki_table[cm]
    sc = scatter(aki_table["vel_diff"],aki_table[p], marker="o", c=temp)
    colorbar(sc, label=cm)

    for i, row in aki_table.iterrows():
        f = aki_table["Type"][i]
        name = aki_table["el_wl"][i]
        if f=="M1" or f=="M2" or f=="E2" or f=="M1+E2":
            c  = "palevioletred"
            if name == fl:
                c = "darkred"
        else:
            c = "k"
        #print(aki_table['el_wl'][i])
        annotate(aki_table['el_wl'][i], (aki_table["vel_diff"][i],aki_table[p][i]),
                textcoords="offset points", xytext=(0,7), ha='center', color=c)

    #xlim(-80,80)
    #ylim(0,15)
    #yscale(log)    
    xlabel("Differential Velocity")
    ylabel(p)
    show()
    tight_layout()
    #title(fl)
    
def cut_hyperfine(df,decimals=2):
    """
    Cuts the hyperfine level transitions from the identified lines
    ----
    Input
    df - Dataframe of lines identified in the spectra
    decimals - Decimal placement to round up Ei (upper energy level)
    ---
    Output
    df - Dataframe without hyperfine level transitions "duplicates". Only the most likely transition was kept
    """
    pd.options.mode.copy_on_write = True 
    df["Ek_trunc"] = np.trunc(df["Ek"] *(10**decimals))/(10**decimals)
    df["Ek_rounded"] = df["Ek"].round(decimals=2)
    df.sort_values("Aki", inplace=True)
    #df = df.drop_duplicates(subset=["Ek_rounded","w0"])#, keep="first")
    df = df.drop_duplicates(subset=["Ek_trunc","w0"])#, keep="first")
    return df