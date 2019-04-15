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

# coordinates (XYZ) of C1, C2, C4 and C5
XYZ = np.array([
        [0.274791784, -1.001679346, -1.851320839, 0.365840754],
        [-1.155674199, -1.215133985, 0.053119249, 1.162878076],
        [1.216239624, 0.764265677, 0.956099579, 1.198231236]])

# Inital guess of the plane
p0 = [0.506645455682, -0.185724560275, -1.43998120646, 1.37626378129]

def f_min(X,p):
    plane_xyz = p[0:3]
    distance = (plane_xyz*X.T).sum(axis=1) + p[3]
    return distance / np.linalg.norm(plane_xyz)

def residuals(params, signal, X):
    return f_min(X, params)

from scipy.optimize import leastsq
sol = leastsq(residuals, p0, args=(None, XYZ))[0]

print("Solution: ", sol)
print("Old Error: ", (f_min(XYZ, p0)**2).sum())
print("New Error: ", (f_min(XYZ, sol)**2).sum())

""" showing data and fit result"""
x= np.linspace(-10,10,10)
y= np.linspace(-10,10,10)
X,Y=np.meshgrid(x,y)
Z = -(1/p0[2])*(p0[0]*X + p0[1]*Y + p0[3])
fig1 = plt.figure(1)
ax=fig1.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(X,Y,Z,color='#4060a6',alpha=.6)
ax.scatter(XYZ[0,:], XYZ[1,:], XYZ[2,:])
# ax.set_title("({:1.2f},{:1.2f})".format(-params[0]/params[1],1/params[1]))
plt.show()
