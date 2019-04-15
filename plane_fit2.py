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

def read_GASS():
    xxGASS = fits.open('data/xxGASS_MASTER_CO_170620_final.fits')
    xxGASS = Table(xxGASS[1].data).to_pandas()
    xxGASS = xxGASS[xxGASS['SFR_best'] > -80]
    # print (xxGASS.columns)
    # print (xxGASS[['HIar_flag', 'Gdcode', 'GASSDR', 'zHI', 'W50cor', 'lgMHI_old', 'lgMHI', 'lgGF', 'HIconf_flag']])
    data = xxGASS[['SFR_best', 'lgMHI', 'lgMstar', 'SFRerr_best', 'HIsrc', 'HIconf_flag']]
    det = data[data['HIsrc']!=4]
    nondet = data[data['HIsrc']==4]
    det = data[data['HIconf_flag']==0]
    nondet = data[data['HIconf_flag']==-99]
    det['SFRerr_best'] = det['SFRerr_best']/(det['SFR_best']*np.log(10))
    det['SFR_best'] = np.log10(det['SFR_best'])
    nondet['SFRerr_best'] = nondet['SFRerr_best']/(nondet['SFR_best']*np.log(10))
    nondet['SFR_best'] = np.log10(nondet['SFR_best'])
    return xxGASS, det, nondet

N_POINTS = 1000
TARGET_X_SLOPE = 2
TARGET_y_SLOPE = 3
TARGET_OFFSET  = 5
EXTENTS = 5
NOISE = 5

xxGASS, det, nondet = read_GASS()
xxGASS_final = fits.open('data/xGASS_RS_final_Serr_180903.fits')
xxGASS_final = Table(xxGASS_final[1].data).to_pandas()
xxGASS['SNR'] = xxGASS_final['SNR']
xxGASS['MHI_err'] = np.power(10, xxGASS['lgMHI'])/xxGASS['SNR']
xxGASS['lgMHI_err'] = xxGASS['MHI_err']/(np.power(10,xxGASS['lgMHI'])*np.log(10))
xxGASS['lgMstar_err'] = 0.0
det = xxGASS[xxGASS['HIconf_flag']==0]
nondet = xxGASS[xxGASS['HIconf_flag']==-99]

xs = det['lgMstar'].values
ys = np.log10(det['SFR_best'].values)
zs = det['lgMHI'].values

# # create random data
# xs = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
# ys = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
# zs = []
# for i in range(N_POINTS):
#     zs.append(xs[i]*TARGET_X_SLOPE + \
#               ys[i]*TARGET_y_SLOPE + \
#               TARGET_OFFSET + np.random.normal(scale=NOISE))

# plot raw data
plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(xs, ys, zs, color='b')

# do fit
tmp_A = []
tmp_b = []
for i in range(len(xs)):
    tmp_A.append([xs[i], ys[i], 1])
    tmp_b.append(zs[i])
b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)
fit = (A.T * A).I * A.T * b
errors = b - A * fit
residual = np.linalg.norm(errors)

print ("solution:")
print ("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
print ("errors:")
print (errors)
print ("residual:")
print (residual)

# plot plane
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
ax.plot_wireframe(X,Y,Z, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
