<img src="STAR_MELT_logo.png" width="600">

# STAR MELT
Repository for the STAR-MELT emission line analysis Jupyter notebook and Python package.\
[See the one-minute STAR-MELT overview video here.](https://youtu.be/grDMizYmU6U)


------------
## Binder Notebook

To launch and try the STAR MELT tutorial notebook on binder, click the badge below.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/justyncw/STAR_MELT/HEAD)

This will open the repository in an online Jupyter instance. Click on STAR_MELT_example_notebook.ipynb to open the notebook in a new window. Once in the notebook, click on a code cell and hit shift+enter to run it and advance to the next cell. Selections can be made with the ipywidgets and qgrids. 



------------
## Release Notes
#### This is the development version of the STAR-MELT package (Campbell-White+,MNRAS,under review).

#### Example data and standard star FITS files are from the [ESO Science Archive](http://archive.eso.org/).

* EX Lupi: ESO Programme IDs 099.A-9010, 082.C-0390, 085.C-0764
* GQ Lupi: ESO Programme IDs 075.C-0710, 085.A-9027
* CVSO109: ESO Programme IDs 106.20Z8.009, 106.20Z8.002, [ODYSSEUS & PENELLOPE Zenodo](https://zenodo.org/communities/odysseus/)     
   Manara, C. F., et al. (2021), A&A, arXiv e-prints, [arXiv:2103.12446](https://arxiv.org/abs/2103.12446).

#### Emission line parameters are from the [NIST database](https://physics.nist.gov/PhysRefData/ASD/lines_form.html).  
* Kramida, A., Ralchenko, Yu., Reader, J. and NIST ASD Team (2020). NIST Atomic Spectra Database (version 5.8), [Online]. Available: <https://physics.nist.gov/asd> [Tue Jun 22 2021]. 



------------
## Download
To use the STAR-MELT Jupyter notebook, download or clone the repository into a local directory and start Jupyter notebook from that directory:
```
cd STAR_MELT-main
jupyter notebook 
```
Then open the STAR_MELT_example_notebook.ipynb notebook.

Package requirements are given within the example notebook and in requirements.txt.\
The example notebook contains a tutorial for the package functions using the example data.


------------
## Instrument Compatibility
STAR-MELT will read the spectral data directly from the FITS files for the following instruments:
* ESO FEROS
* ESO HARPS
* ESO XSHOOTER
* ESO UVES
* ESO ESPRESSO
* CFHT ESPaDOnS
* HST COS
* HST STIS
* XMM-Newton RGS

Reference emission lines and radial velocity standard stars are provided for the ground based data.

If your FITS files have a similar structure to these, they may also work. 
Further full instrument compatibility is ongoing. 

Alternatively, spectral data from any source can be provided as a txt/csv file of wave vs flux.

Full package compatibility with HST and XMM spectra is still under development. 


------------
#### QGRID install and enable
The STAR-MELT notebook uses the [QGRID package](https://github.com/quantopian/qgrid) for filtering dataframes

Installing with pip::
```
pip install qgrid
jupyter nbextension enable --py --sys-prefix qgrid

# only required if you have not enabled the ipywidgets nbextension yet
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```

Installing with conda::
```
# only required if you have not added conda-forge to your channels yet
conda config --add channels conda-forge

conda install qgrid
```

Usage:

```python
#control/cmd/shift click to make selections
new_dataframe=qgrid_widget1.get_selected_df()
```
```python
#use qgrid column filters
new_dataframe=qgrid_widget1.get_changed_df()
```
