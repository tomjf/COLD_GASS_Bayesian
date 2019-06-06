from scipy import stats
# import pyximport
from astropy.io import fits
from astropy.table import Table
import math
import numpy as np
from scipy import integrate
from scipy.stats import lognorm, gennorm, genlogistic
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import random
from scipy.optimize import minimize
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import emcee
import corner
from scipy.integrate import quad, dblquad, nquad
from scipy import special
import random
from integrand import integrand_MHI, integrand_MHI_var_sigma, integrand_MHI_double, integrand_MHI_logistic
import os
import time
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
pd.options.mode.chained_assignment = None  # default='warn'
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)


masses = fits.open('data/totlgm_dr7_v5_2.fit')
masses = Table(masses[1].data).to_pandas()

sfrs = fits.open('data/gal_totsfr_dr7_v5_2.fits')
sfrs = Table(sfrs[1].data).to_pandas()
masses['sfr'] = sfrs['AVG']
masses = masses[masses['AVG']>9]
masses = masses[masses['AVG']<12]
masses = masses[masses['sfr']>-2]
masses = masses[masses['sfr']<.5]
# masses = masses[~masses.isin([np.nan, np.inf, -np.inf]).any(1)]
print (masses[['AVG', 'sfr']])
plt.hist2d(masses['AVG'], np.log10(masses['sfr']),bins=10)
plt.xlim(9,12)
plt.ylim(-2,0.5)
plt.show()
