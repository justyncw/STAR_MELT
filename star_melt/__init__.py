import os
import glob

# Package metadata
__version__ = "0.9.0"  # Ensure version information is defined
__author__ = "Justyn Campbell-White"

from .ESP_fits_get_spectra import read_ESP_fits_spec
from .ESO_fits_get_spectra import read_ESO_fits_spec
from .utils_saha_av import saha_av
from .utils_data import *
from .utils_physics import *
from .utils_spec import * 
import star_melt.utils_shared_variables as USH            

from .utils_shared_variables import * #fig_size_s, fig_size_l, fig_size_n, inst_res, radvel, vsini, target, instrument, line_resources_dir, line_table_prev_obs

# Add imported modules to globals and __all__
globals()["saha_av"] = saha_av
globals()["read_ESP_fits_spec"] = read_ESP_fits_spec
globals()["read_ESO_fits_spec"] = read_ESO_fits_spec
globals()["utils_data"] = utils_data
globals()["utils_spec"] = utils_spec
globals()["utils_physics"] = utils_physics
globals()["USH"] = USH  # Ensure USH is also available globally

__all__ = [
    "saha_av",
    "read_ESP_fits_spec",
    "read_ESO_fits_spec",
    *dir(utils_data),
    *dir(utils_spec),
    *dir(utils_physics),
    *dir(USH),
]

__bibtex__ = "BibTeX reference string or metadata"
__version_info__ = (0, 9, 0)  # Version tuple for compatibility checks
__warningregistry__ = {}