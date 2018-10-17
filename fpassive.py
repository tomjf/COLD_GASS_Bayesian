# import pyximport
from astropy.io import fits
from astropy.table import Table
import math
import numpy as np
from scipy import integrate
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
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
from integrand import integrand_MHI, integrand_MHI_var_sigma
import os
import time
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
pd.options.mode.chained_assignment = None  # default='warn'
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)


def f_passive():
    a = 10.804
    b = -2.436
    zeta = -1.621
    mstar = np.logspace(7,12,100)
    mstar2 = np.linspace(7,12,100)
    c = 1 + np.tanh(zeta)
    fpassive = c + ((1-c)/(1+np.power(mstar/np.power(10, a), b)))
    fpassive2 = c + ((1-c)/(1+np.power(np.power(10,mstar2-a), b)))
    return mstar2, fpassive, fpassive2


mstar2, fpassive, fpassive2 = f_passive()
plt.plot(mstar2,fpassive)
plt.plot(mstar2,fpassive2)
plt.show()
