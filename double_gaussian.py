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

def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z

def log_schechter_true(logL, log_phi, log_L0, alpha):
    # print (log_phi, log_L0, alpha)
    log = np.log(10)
    frac = np.power(10,(alpha+1)*(logL-log_L0))
    exp = np.exp(-np.power(10,logL-log_L0))
    return log*log_phi*frac*exp

def integrand_MHI_blue(M, SFR, MHI, *params):
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    lnh
    Mstar, phistar1, alpha1 = 10.72, 0.71E-3, -1.45
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))
    f = (b1*M*M) + (b2*M) + b3
    P_SFR_given_Mstar = (1/np.sqrt(2*np.pi*np.power(np.exp(lnb),2)))*np.exp((-1/(2*np.power(np.exp(lnb),2)))*np.power((SFR-f),2))
    f2 = (h1*SFR) + h2
    P_MHI_given_SFR = (1/np.sqrt(2*np.pi*np.power(np.exp(lnh),2)))*np.exp((-1/(2*np.power(np.exp(lnh),2)))*np.power((MHI-f2),2))
    return phi_Mstar_double*P_SFR_given_Mstar*P_MHI_given_SFR

def S_error(x_err,y_err):
    N = len(x_err)
    S = np.zeros((N, 2, 2))
    for n in range(N):
        L = np.zeros((2, 2))
        L[0, 0] = np.square(x_err[n])
        if len(y_err) == 1:
            L[1, 1] = np.square(y_err[0])
        else:
            L[1, 1] = np.square(y_err[n])
        S[n] = L
    return S

def integrand_SFR(M, SFR, *params):
    # parameters inferred from emcee
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    # Baldry+11 double Schechter function parameters
    # Mstar = 10.66
    # phistar1 = 3.96E-3
    # phistar2 = 0.79E-3
    # alpha1 = - 0.35
    # alpha2 = - 1.47

    Mstar = 10.78
    phistar1 = 2.93E-3
    phistar2 = 0.63E-3
    alpha1 = - 0.62
    alpha2 = - 1.50
    # probabilities
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    fpass = f_passive(M, alpha, beta, zeta)
    # P_SFR_given_passive
    y_passive = (r1*M) + r2
    P_SFR_given_Mstar_red = (1/np.sqrt(2*np.pi*np.power(np.exp(lnr),2)))*np.exp((-1/(2*np.power(np.exp(lnr),2)))*np.power((SFR-y_passive),2))
    # P_SFR_given_sforming
    y_sforming = (b1*M*M) + (b2*M) + b3
    P_SFR_given_Mstar_blue = (1/np.sqrt(2*np.pi*np.power(np.exp(lnb),2)))*np.exp((-1/(2*np.power(np.exp(lnb),2)))*np.power((SFR-y_sforming),2))
    # P_SFR_total
    P_SFR_given_Mstar_total = fpass*P_SFR_given_Mstar_red + (1-fpass)*P_SFR_given_Mstar_blue
    # return phi_SFR
    return phi_Mstar_double*P_SFR_given_Mstar_total

def integrand_MHI_direct(M, MHI, *params):
    # parameters inferred from emcee
    a1, a2, lna = params

    Mstar = 10.78
    phistar1 = 2.93E-3
    phistar2 = 0.63E-3
    alpha1 = - 0.62
    alpha2 = - 1.50
    # probabilities
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    P_MHI_given_Mstar = (1/np.sqrt(2*np.pi*np.power(np.exp(lna),2)))*np.exp((-1/(2*np.power(np.exp(lna),2)))*np.power((MHI - ((a1*M) + a2)),2))
    # P_SFR_total
    return phi_Mstar_double*P_MHI_given_Mstar

def integrand_SFR_blue(M, SFR, *params):
    # parameters inferred from emcee
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    # Baldry+11 double Schechter function parameters
    # Mstar = 10.66
    # phistar1 = 3.96E-3
    # phistar2 = 0.79E-3
    # alpha1 = - 0.35
    # alpha2 = - 1.47
    Mstar = 10.78
    phistar1 = 2.93E-3
    phistar2 = 0.63E-3
    alpha1 = - 0.62
    alpha2 = - 1.50
    # probabilities
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    fpass = f_passive(M, alpha, beta, zeta)
    # P_SFR_given_passive
    # y_passive = (r1*M) + r2
    # P_SFR_given_Mstar_red = (1/np.sqrt(2*np.pi*np.power(np.exp(lnr),2)))*np.exp((-1/(2*np.power(np.exp(lnr),2)))*np.power((SFR-y_passive),2))
    # P_SFR_given_sforming
    y_sforming = (b1*M*M) + (b2*M) + b3
    P_SFR_given_Mstar_blue = (1/np.sqrt(2*np.pi*np.power(np.exp(lnb),2)))*np.exp((-1/(2*np.power(np.exp(lnb),2)))*np.power((SFR-y_sforming),2))
    # P_SFR_total
    P_SFR_given_Mstar_total = (1-fpass)*P_SFR_given_Mstar_blue
    # return phi_SFR
    return phi_Mstar_double*P_SFR_given_Mstar_total

def integrand_SFR_red(M, SFR, *params):
    # parameters inferred from emcee
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    # Baldry+11 double Schechter function parameters
    # Mstar = 10.66
    # phistar1 = 3.96E-3
    # phistar2 = 0.79E-3
    # alpha1 = - 0.35
    # alpha2 = - 1.47
    Mstar = 10.78
    phistar1 = (0.7**3)*2.93E-3
    phistar2 = (0.7**3)*0.63E-3
    alpha1 = - 0.62
    alpha2 = - 1.50
    # probabilities
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    fpass = f_passive(M, alpha, beta, zeta)
    # P_SFR_given_passive
    y_passive = (r1*M) + r2
    P_SFR_given_Mstar_red = (1/np.sqrt(2*np.pi*np.power(np.exp(lnr),2)))*np.exp((-1/(2*np.power(np.exp(lnr),2)))*np.power((SFR-y_passive),2))
    # P_SFR_given_sforming
    # y_sforming = (b1*M*M) + (b2*M) + b3
    # P_SFR_given_Mstar_blue = (1/np.sqrt(2*np.pi*np.power(np.exp(lnb),2)))*np.exp((-1/(2*np.power(np.exp(lnb),2)))*np.power((SFR-y_sforming),2))
    # P_SFR_total
    P_SFR_given_Mstar_total = fpass*P_SFR_given_Mstar_red
    # return phi_SFR
    return phi_Mstar_double*P_SFR_given_Mstar_total

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

def f_passive(x, a, b, zeta):
    c = 1 + np.tanh(zeta)
    return c + ((1-c)/(1+np.power(np.power(10,x-a), b)))

def plot_samples_3(sampler, ndim, fname):
    fig, axes = plt.subplots(ndim, figsize=(10, 20), sharex=True)
    samples = sampler.get_chain()
    labels = ['h1', 'h2', 'lnh']
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    plt.savefig('img/sampler' + fname + '.pdf')

def plot_samples_full(sampler, ndim, fname):
    fig, axes = plt.subplots(ndim, figsize=(10, 20), sharex=True)
    samples = sampler.get_chain()
    labels = ['b1', 'b2', 'b2', 'lnb', 'r1', 'r2', 'lnr', 'a1', 'a2', 'a3', 'h1', 'h2', 'lnh']
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    plt.savefig('img/sampler' + fname + '.pdf')

def SFR_HI_fit_params(params):
    h1, h2, lnh = params
    if  0.6 < h1 < 1.0 and \
        8.0 < h2 < 11.0 and \
        -2.0 < lnh < 2.0:
        return 0
    return -np.inf

def SFR_HI_fit(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    h1, h2, lnh = params
    x1, x2, y1, y2, S1, S2 = GASS_data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood for the linear SFR-MHI plane
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = np.array([-h1, 1.0])
    deltaN = y1 - (h1 * x1) - h2
    model = (h1 * x2) + h2
    sigma = np.dot(np.dot(S1, v), v) + np.exp(2 * lnh)
    sigma2 = np.dot(np.dot(S2, v), v) + np.exp(2 * lnh)
    sigma2 = sigma2 ** 0.5
    ll1 = -0.5 * np.sum(np.square(deltaN) / sigma + np.log(sigma))
    I = np.zeros(len(x2))
    for i in range(0,len(x2)):
        I[i] = np.log(((2 * np.pi) ** 0.5) * 0.5 * (special.erf((y2[i]-model[i]) / ((2 ** 0.5) * sigma2[i])) + 1))
    ll2 = np.sum(I)
    LL_SFR_MHI = ll1  + ll2
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return LL_SFR_MHI + SFR_HI_fit_params(params)

def MHI_Mstar_fit_params(params):
    a1, a2, lna = params
    if  0.0 < a1 < 2.0 and \
        -5.0 < a2 < 8.0 and \
        -2.0 < lna < 2.0:
        return 0
    return -np.inf

def MHI_Mstar_fit(params):
    a1, a2, lna = params
    x, x2, y, y2, S1, S2 = GASS_data2
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood for the linear SFR-MHI plane
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = np.array([-a1, 1.0])
    deltaN = y - (a1 * x) - a2
    model = (a1 * x2) + a2
    sigma = np.dot(np.dot(S1, v), v) + np.exp(2 * lna)
    sigma2 = np.dot(np.dot(S2, v), v) + np.exp(2 * lna)
    sigma2 = sigma2 ** 0.5
    ll1 = -0.5 * np.sum(np.square(deltaN) / sigma + np.log(sigma))
    I = np.zeros(len(x2))
    for i in range(0,len(x2)):
        I[i] = np.log(((2 * np.pi) ** 0.5) * 0.5 * (special.erf((y2[i]-model[i]) / ((2 ** 0.5) * sigma2[i])) + 1))
    ll2 = np.sum(I)
    LL_SFR_MHI = ll1  + ll2

    return LL_SFR_MHI + MHI_Mstar_fit_params(params)


def log_mainsequence_priors_full(params):
    b1, b2, b3, lnb, r1, r2, r3, lnr, alpha, beta, zeta = params
    if  -0.3 < b1 < 0.0 and \
        2.0 < b2 < 4.0 and \
        -22.0 < b3 < -16.0 and \
        -5.0 < lnb < 5.0 and \
        0.0 < r1 < 1.0 and \
        -14.0 < r2 < -10.0 and \
        50.0 < r3 < 70.0 and \
        -5.0 < lnr < 5.0 and \
        9.0 < alpha < 12.0 and \
        -2.0 < beta < 0.0 and \
        -7.0 < zeta < -2.0:
        return 0
    return -np.inf

def log_marg_mainsequence_full(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    b1, b2, b3, lnb, r1, r2, r3, lnr, alpha, beta, zeta = params
    x, y, xerr, yerr =  GAMA_data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # passive fraction
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    c = 1 + np.tanh(zeta)
    f_pass = c + ((1-c)/(1+np.power(np.power(10,x-alpha), beta)))
    # assume R = 1
    B = (1-f_pass)/f_pass
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # star forming likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_blue = np.square(xerr)*np.square((2*b1*x) + b2) + np.square(yerr) + np.exp(2*lnb)
    DeltaN_blue = y - (b1*x*x) - (b2*x) - b3
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # passive likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_red = np.square(xerr)*np.square((2*r1*x) + r2) + np.square(yerr) + np.exp(2*lnr)
    DeltaN_red = y - (r1*x*x) - (r2*x) - r3
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ll_all = np.sum(np.log((B/Sigma2_blue)*np.exp(-DeltaN_blue**2/(2*Sigma2_blue)) + (1/Sigma2_red)*np.exp(-DeltaN_red**2/(2*Sigma2_red))))
    return ll_all + log_mainsequence_priors_full(params)


def log_mainsequence_priors_full2(params):
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    if  -0.3 < b1 < 0.0 and \
        0.0 < b2 < 4.0 and \
        -16.0 < b3 < -10.0 and \
        -5.0 < lnb < 5.0 and \
        0.3 < r1 < 1.5 and \
        -9.0 < r2 < -5.5 and \
        -5.0 < lnr < 5.0 and \
        9.0 < alpha < 12.0 and \
        -4.0 < beta < 0.0 and \
        -7.0 < zeta < 0.0 and \
        0.6 < h1 < 1.0 and \
        8.0 < h2 < 11.0 and \
        -2.0 < lnh < 2.0:
        return 0
    return -np.inf

def log_marg_mainsequence_full2(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    x, y, yerr = passive_data
    sfr, n = sfr_hist_data
    xb, yb, xerrb, yerrb = GAMA_sforming
    xr, yr, xerrr, yerrr = GAMA_passive
    xt, yt, xerrt, yerrt = GAMA_data
    x1, x2, y1, y2, S1, S2 = GASS_data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # passive fraction likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    c = .5*(1+np.tanh(zeta))
    Sigma2 = np.square(yerr)
    Delta = y - (c + ((1-c)/(1+np.power(np.power(10,x-alpha), beta))))
    ll_pass_frac = -0.5 * np.sum(Delta**2 / Sigma2 + np.log(Sigma2))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # star forming likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_sf = np.square(xerrb)*np.square((2*b1*xb) + b2) + np.square(yerrb) + np.exp(2*lnb)
    DeltaN = yb - (b1*xb*xb) - (b2*xb) - b3
    ll_sf = -0.5 * np.sum(DeltaN**2/Sigma2_sf + np.log(Sigma2_sf))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # passive likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_pass = np.square(xerrr)*np.square(r1) + np.square(yerrr) + np.exp(2*lnr)
    DeltaN = yr - (r1*xr) - r2
    ll_pass = -0.5 * np.sum(DeltaN**2/Sigma2_pass + np.log(Sigma2_pass))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood for the linear SFR-MHI plane
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = np.array([-h1, 1.0])
    deltaN = y1 - (h1 * x1) - h2
    model = (h1 * x2) + h2
    sigma = np.dot(np.dot(S1, v), v) + np.exp(2 * lnh)
    sigma2 = np.dot(np.dot(S2, v), v) + np.exp(2 * lnh)
    sigma2 = sigma2 ** 0.5
    ll1 = -0.5 * np.sum(np.square(deltaN) / sigma + np.log(sigma))
    I = np.zeros(len(x2))
    for i in range(0,len(x2)):
        I[i] = np.log(((2 * np.pi) ** 0.5) * 0.5 * (special.erf((y2[i]-model[i]) / ((2 ** 0.5) * sigma2[i])) + 1))
    ll2 = np.sum(I)
    LL_SFR_MHI = ll1  + ll2
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return ll_sf + ll_pass + ll_pass_frac + LL_SFR_MHI + log_mainsequence_priors_full2(params)

def log_marg_mainsequence_full3(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta = params
    x, y, yerr = passive_data
    sfr, n = sfr_hist_data
    xb, yb, xerrb, yerrb = GAMA_sforming
    xr, yr, xerrr, yerrr = GAMA_passive
    xt, yt, xerrt, yerrt = GAMA_data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # passive fraction likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    c = .5*(1+np.tanh(zeta))
    Sigma2 = np.square(yerr)
    Delta = y - (c + ((1-c)/(1+np.power(np.power(10,x-alpha), beta))))
    ll_pass_frac = -0.5 * np.sum(Delta**2 / Sigma2 + np.log(Sigma2))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # star forming likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_sf = np.square(xerrb)*np.square((2*b1*xb) + b2) + np.square(yerrb) + np.exp(2*lnb)
    DeltaN = yb - (b1*xb*xb) - (b2*xb) - b3
    ll_sf = -0.5 * np.sum(DeltaN**2/Sigma2_sf + np.log(Sigma2_sf))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # passive likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_pass = np.square(xerrr)*np.square(r1) + np.square(yerrr) + np.exp(2*lnr)
    DeltaN = yr - (r1*xr) - r2
    ll_pass = -0.5 * np.sum(DeltaN**2/Sigma2_pass + np.log(Sigma2_pass))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood delta MHI
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # make the means, passive fractions for this set of params
    b_mean = (b1*xt*xt) + (b2*xt) + b3
    r_mean = (r1*xt) + r2
    f_pass = f_passive(xt, alpha, beta, zeta)
    rand = np.random.uniform(0, 1, len(xt))
    sfrs = []
    for idx, element in enumerate(b_mean):
        if rand[idx] <= f_pass[idx]:
            sfrs.append(r_mean[idx])
        else:
            sfrs.append(b_mean[idx])
    n2, sfr5 = np.histogram(sfrs, sfr_bins)
    delta = n - n2
    ll_delta_sfr = -0.5 * np.sum(delta**2/np.sqrt(n) + np.log(np.sqrt(n)))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return ll_sf + ll_pass + ll_pass_frac + ll_delta_sfr + log_mainsequence_priors_full2(params)

def gauss(x,a1,mu1,sigma1):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2))

def double_gauss(x,a1,mu1,sigma1,a2,mu2,sigma2):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-(mu2))**2/(2*sigma2**2))

def triple_gauss(x,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + \
            a2*np.exp(-(x-(mu2))**2/(2*sigma2**2)) + \
            a3*np.exp(-(x-(mu3))**2/(2*sigma3**2))

def bootstrap_GAMA(GAMAb, GAMAr, frac, n):
    blue = np.copy(GAMAb['logM*'].values)
    red = np.copy(GAMAr['logM*'].values)
    x1 = np.linspace(6.0,11.3,40)
    data = np.zeros((n, len(x1) + 7))
    # blue = GAMAb['logM*'].values
    # red = GAMAr['logM*'].values
    print (type(blue))
    # index_blue = np.linspace(0,len(GAMAb['logM*'])-1,len(GAMAb['logM*']))
    # index_red = np.linspace(0,len(GAMAr['logM*'])-1,len(GAMAr['logM*']))
    lim_index_blue = int(len(blue)*frac)
    print (lim_index_blue)
    lim_index_red = int(len(red)*frac)
    for j in range (0, n):
        # randomly select a fraction of the data
        # first randomly shuffle the original data so the order is always different
        np.random.shuffle(blue)
        np.random.shuffle(red)
        # select a fraction of the data
        blue_shuffled = blue[:lim_index_blue]
        red_shuffled = red[:lim_index_red]
        ratio, xnew = [], []
        for i in range(0, len(x1) - 1):
            if x1[i] < 7.5:
                ratio.append(np.random.normal(0, 0.01))
                xnew.append((x1[i+1] + x1[i])/2)
            else:
                sf = blue_shuffled[blue_shuffled > x1[i]]
                sf = sf[sf <= x1[i+1]]
                passive = red_shuffled[red_shuffled > x1[i]]
                passive = passive[passive <= x1[i+1]]
                ratio.append(len(passive)/(len(sf)+len(passive)))
                xnew.append((x1[i+1] + x1[i])/2)
            if x1[i] >= 11.5:
                ratio.append(np.random.normal(0, 0.01))
                xnew.append((x1[i+1] + x1[i])/2)
        xnew.append(11.25)
        xnew.append(11.5)
        xnew.append(11.75)
        xnew.append(12.0)
        xnew.append(12.25)
        xnew.append(12.5)
        xnew.append(12.75)
        xnew.append(13.0)
        ratio.append(np.random.normal(1.0, 0.01))
        ratio.append(np.random.normal(1.0, 0.01))
        ratio.append(np.random.normal(1.0, 0.01))
        ratio.append(np.random.normal(1.0, 0.01))
        ratio.append(np.random.normal(1.0, 0.01))
        ratio.append(np.random.normal(1.0, 0.01))
        ratio.append(np.random.normal(1.0, 0.01))
        ratio.append(np.random.normal(1.0, 0.01))
        data[j,:] = ratio
    std = []
    for idx in range(0, len(x1) + 7):
        std.append(np.std(data[:,idx]))
    # bootstrap for the original full dataset
    xnew, ratio = [], []
    for i in range(0, len(x1) - 1):
        if x1[i] < 7.5:
            ratio.append(0)
            xnew.append((x1[i+1] + x1[i])/2)
        else:
            sf = blue[blue > x1[i]]
            sf = sf[sf <= x1[i+1]]
            passive = red[red > x1[i]]
            passive = passive[passive <= x1[i+1]]
            ratio.append(len(passive)/(len(sf)+len(passive)))
            xnew.append((x1[i+1] + x1[i])/2)
    xnew.append(11.25)
    xnew.append(11.5)
    xnew.append(11.75)
    xnew.append(12.0)
    xnew.append(12.25)
    xnew.append(12.5)
    xnew.append(12.75)
    xnew.append(13.0)
    ratio.append(np.random.normal(1.0, 0.01))
    ratio.append(np.random.normal(1.0, 0.01))
    ratio.append(np.random.normal(1.0, 0.01))
    ratio.append(np.random.normal(1.0, 0.01))
    ratio.append(np.random.normal(1.0, 0.01))
    ratio.append(np.random.normal(1.0, 0.01))
    ratio.append(np.random.normal(1.0, 0.01))
    ratio.append(np.random.normal(1.0, 0.01))
    print (type(xnew), np.shape(xnew))
    print (type(ratio), np.shape(ratio))
    print (type(std), np.shape(std))
    return xnew, std, ratio

def sfr_histogram(GAMA, samples4, samples5, M, phi_Baldry):
    sfr_bins = np.linspace(-3.0, 3.0, 25)
    # Make the plot
    fig, ax = plt.subplots(nrows = 4, ncols = 2, squeeze=False, figsize=(12, 24))
    ax2 = ax[2,1].twiny()
    SFR = np.linspace(-4,3,30)
    Mstar = np.linspace(3,15,1000)
    ############################################################################
    # Plot 1 for observed SFR histogram
    ############################################################################
    # predicted SFR from Kennicut paper
    phi_SFR = np.log(10) * np.exp(-np.power(10,SFR-np.log10(9.2))) * (0.00016*np.power(10,(-1.51+1)*(SFR-np.log10(9.2))))
    ax[0,1].plot(SFR, np.log10(phi_SFR), linestyle = '--', linewidth = 1, color ='k')
    for params in samples4[np.random.randint(len(samples4), size=10)]:
        b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
        SFR2 = (b1*Mstar*Mstar) + (b2*Mstar) + b3
        ax[3,1].plot(Mstar, SFR2)
        phi, phib, phir = [], [], []
        b1ave, b2ave, b3ave = [], [], []
        for idx, element in enumerate(SFR):
            # print (b1, b2, b3, element, np.roots([b1, b2, b3-element])[1])
            phi.append(quad(integrand_SFR, 0, 12, args=(element, *params))[0])
            phib.append(quad(integrand_SFR_blue, 6, 12, args=(element, *params))[0])
            phir.append(quad(integrand_SFR_red, 0, 12, args=(element, *params))[0])
            b1ave.append(b1)
            b2ave.append(b2)
            b3ave.append(b3)
        b1ave = np.mean(b1ave)
        b2ave = np.mean(b2ave)
        b3ave = np.mean(b3ave)
        ax[0,1].plot(SFR,np.log10(phi), color = 'k', alpha = 0.1, linewidth = 0.1)
        ax2.plot(SFR,np.log10(phi), color = 'k', alpha = 0.1, linewidth = 0.1)
        ax2.set_xlim(b1ave*36 + b2ave*6 + b3ave, b1ave*144 + b2ave*12 + b3ave)
        ax[2,1].set_xlim(6,12)

        ax[0,1].plot(SFR,np.log10(phib), color = 'b', alpha = 0.1, linewidth = 0.1)
        ax[0,1].plot(SFR,np.log10(phir), color = 'r', alpha = 0.1, linewidth = 0.1)


        # mmin.append(third_order(6, b1, b2, b3))
        # mmax.append(third_order(12, b1, b2, b3))
        # make the means, passive fractions for this set of params
        GAMA['b_mean'] = (b1*GAMA['logM*']*GAMA['logM*']) + (b2*GAMA['logM*']) + b3
        GAMA['r_mean'] = (r1*GAMA['logM*']) + r2
        GAMA['f_pass1'] = f_passive(GAMA['logM*'], alpha, beta, zeta)
        # GAMA['f_pass2'] = exp_function2(GAMA['logM*'], 2.15695552e-04,  1.03762864e+00, -3.23331742e+00,  11.2)
        # print (GAMA[['logM*', 'f_pass']])
        GAMA['rand'] = np.random.uniform(0, 1, len(GAMA))
        GAMA['sfr_model1'] = -9.9
        GAMA['sfr_model2'] = -9.9
        # calculate the model sfrs for this set of params
        for idx, row in GAMA.iterrows():
            # if random number is less than f_pass its a red galaxy
            if row['rand'] <= row['f_pass1']:
                GAMA.loc[idx,'sfr_model1'] = row['r_mean'] + np.random.normal(0, np.exp(lnr))
            # else its a blue galaxy
            else:
                GAMA.loc[idx,'sfr_model1'] = row['b_mean'] + np.random.normal(0, np.exp(lnb))
        # for idx, row in GAMA.iterrows():
        #     # if random number is less than f_pass its a red galaxy
        #     if row['rand'] <= row['f_pass2']:
        #         GAMA.loc[idx,'sfr_model2'] = row['r_mean'] + np.random.normal(0, np.exp(lnr))
        #     # else its a blue galaxy
        #     else:
        #         GAMA.loc[idx,'sfr_model2'] = row['b_mean'] + np.random.normal(0, np.exp(lnb))
        # bin the predicted sfrs for this set of params and then plot the sfr hist
        n, sfr, n2, n3 = [], [], [], []
        for idx in range(0, len(sfr_bins) - 1):
            sfr_model = GAMA[GAMA['sfr_model1'] > sfr_bins[idx]]
            sfr_model = sfr_model[sfr_model['sfr_model1'] <= sfr_bins[idx + 1]]
            n2.append(len(sfr_model))
            # sfr_model = GAMA[GAMA['sfr_model2'] > sfr_bins[idx]]
            # sfr_model = sfr_model[sfr_model['sfr_model2'] <= sfr_bins[idx + 1]]
            # n3.append(len(sfr_model))
            sfr.append((sfr_bins[idx] + sfr_bins[idx + 1])/2)
            if idx == 23:
                # the number is the volume area in Mpc of the GAMA survey from z = 0.005 to 0.08
                n2 = n2/((sfr[1] - sfr[0])*1317162.8627450706)
                # n3 = n3/((sfr[1] - sfr[0])*1317162.8627450706)
                ax[0,0].plot(sfr, np.log10(n2), linewidth = 0.1, alpha = 0.2, color = 'b')
                ax[0,1].plot(sfr, np.log10(n2), linewidth = 0.1, alpha = 0.2, color = 'b')
                # ax[0,0].plot(sfr, np.log10(n3), linewidth = 0.1, alpha = 0.2, color = 'g')

    # plot the histogram for the observed data
    for idx in range(0, len(sfr_bins) - 1):
        slice = GAMA[GAMA['logSFR'] > sfr_bins[idx]]
        slice = slice[slice['logSFR'] <= sfr_bins[idx + 1]]
        n.append(len(slice))
    n = n/((sfr[1] - sfr[0])*1317162.8627450706)
    err = (np.sqrt(n)/(np.array(n)*np.log(10.0)))/((sfr[1] - sfr[0])*1317162.8627450706)
    ax[0,0].errorbar(sfr, np.log10(n), yerr = err, mfc='r', mec='r', markersize = 4, linewidth=1, markeredgewidth=1, capthick=3)
    ax[0,1].errorbar(sfr, np.log10(n), yerr = err, mfc='r', mec='r', markersize = 4, linewidth=1, markeredgewidth=1, capthick=3)

    # read in the ALFA ALFA datasets from the 40% paper
    ALFAALFA = pd.read_csv('ALFAALFA.csv', comment = '#', header = None, sep=",")
    ALFAALFA.columns = ['x', 'y', 'dy', 'MHI', 'phi', 'err', 'phi_err']
    ALFAALFA = ALFAALFA[np.isfinite(ALFAALFA['phi_err'])]
    MHI_alfa, phi_alfa, phi_err_alfa = np.round(ALFAALFA['MHI'].values,2), ALFAALFA['phi'].values, ALFAALFA['phi_err'].values
    N, n = 1, 20
    best_fits = np.zeros((N*2,n))
    best_fits_1 = np.zeros((N*2,n))
    best_fits_2 = np.zeros((N*2,n))
    best_fits_3 = np.zeros((N*2,n))
    best_fits_4 = np.zeros((N*2,n))
    MHI = np.linspace(5,13,n)
    i=0
    x = np.linspace(-3,3,100)
    for params in samples4[np.random.randint(len(samples4), size = N)]:
        for idx, element in enumerate(MHI):
            b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
            best_fits[i,idx] = dblquad(integrand_MHI_blue, -8.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, *params))[0]

            best_fits_1[1,idx] = dblquad(integrand_MHI_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1+0.1, h2+0.1, np.log(0.1)))[0]
            best_fits_2[1,idx] = dblquad(integrand_MHI_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1+0.1, h2+0.1, np.log(0.2)))[0]
            best_fits_3[1,idx] = dblquad(integrand_MHI_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1+0.1, h2+0.1, np.log(0.3)))[0]
            best_fits_4[1,idx] = dblquad(integrand_MHI_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1+0.1, h2+0.1, np.log(0.4)))[0]
        i+=1
    for i in range(0, N):
        ax[1,0].plot(MHI, np.log10(best_fits[i,:]), alpha = 1, color = 'c', label = str(round(np.exp(lnh), 2)))
        ax[2,1].plot(MHI, np.log10(best_fits[i,:]), alpha = 1, color = 'c', label = str(round(np.exp(lnh), 2)))
        ax[1,1].plot(x, ((h1+0.1)*x) + h2+0.1, alpha = 1, color = 'r')
        ax[1,0].plot(MHI, np.log10(best_fits_1[1,:]), alpha = 1, color = 'b', linestyle = '--', label = '0.1')
        ax[1,0].plot(MHI, np.log10(best_fits_2[1,:]), alpha = 1, color = 'r', linestyle = '--',  label = '0.2')
        ax[1,0].plot(MHI, np.log10(best_fits_3[1,:]), alpha = 1, color = 'k', linestyle = '--',  label = '0.3')
        ax[1,0].plot(MHI, np.log10(best_fits_4[1,:]), alpha = 1, color = 'm', linestyle = '--',  label = '0.4')
    # ax[1,0].plot(MHI,np.log10(phi), color = 'k', alpha = 0.1, linewidth = 0.1)
    ax[1,0].legend(loc="upper right")
    # ax[1,0].plot(MHI, np.log10(best_fits[0,:]), alpha = 0.1, color = 'g')
    ax[1,0].errorbar(MHI_alfa, phi_alfa, yerr = phi_err_alfa, fmt='o', capsize = 2, markersize = 3, linewidth=2, markeredgewidth=2, capthick=2, mfc='gray', mec='gray', ecolor = 'gray')
    y3 = log_schechter_true(MHI, 4.8E-3, 9.96, -1.33)
    # ax[1,0].plot(MHI, np.log10(y3), color = 'g')
    ax[1,0].set_xlim(6,13)
    ax[1,0].set_ylim(-6,0)


    ax[0,0].set_xlabel(r"$\mathrm{log \, SFR}$")
    ax[0,0].set_ylabel(r"$\mathrm{log \, phi(SFR) \, [dex^{-1} \, Mpc^{-3}]}$")

    ax[0,1].set_xlabel(r"$\mathrm{log \, SFR}$")
    ax[0,1].set_ylabel(r"$\mathrm{log \, phi(SFR) \, [dex^{-1} \, Mpc^{-3}]}$")

    ax[1,0].set_xlabel(r"$\mathrm{log \, M_{HI}}$")
    ax[1,0].set_ylabel(r"$\mathrm{log \, phi(M_{HI}) \, [dex^{-1} \, Mpc^{-3}]}$")

    ax[1,1].set_xlabel(r"$\mathrm{log \, SFR}$")
    ax[1,1].set_ylabel(r"$\mathrm{log \, M_{HI}}$")

    ax[0,0].set_ylim(-6, -2.0)
    ax[0,1].set_ylim(-10, 0.0)
    ax[1,1].set_xlim(-4, 3.0)
    ax[1,1].set_ylim(6, 13)
    ax[1,0].set_ylim(-10, 0.0)

    xxGASS, det, nondet = read_GASS()
    ax[2,0].scatter(xxGASS['lgMstar'], xxGASS['lgMHI'], s = 3)
    # for params in samples4[np.random.randint(len(samples4), size=10)]:
    #     b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    #     xxGASS['b_mean'] = (b1*xxGASS['lgMstar']*xxGASS['lgMstar']) + (b2*xxGASS['lgMstar']) + b3
    #     xxGASS['r_mean'] = (r1*xxGASS['lgMstar']) + r2
    #     xxGASS['f_pass1'] = f_passive(xxGASS['lgMstar'], alpha, beta, zeta)
    #     xxGASS['rand'] = np.random.uniform(0, 1, len(xxGASS))
    #     xxGASS['sfr_model'] = -9.9
    #     # calculate the model sfrs for this set of params
    #     for idx, row in xxGASS.iterrows():
    #         # if random number is less than f_pass its a red galaxy
    #         if row['rand'] <= row['f_pass1']:
    #             xxGASS.loc[idx,'sfr_model'] = row['r_mean'] + np.random.normal(0, np.exp(lnr))
    #         # else its a blue galaxy
    #         else:
    #             xxGASS.loc[idx,'sfr_model'] = row['b_mean'] + np.random.normal(0, np.exp(lnb))
    # These are the new arguments that I used
    scatter_kwargs = {"zorder":100, "vmin":min(xxGASS['lgMstar']), "vmax":max(xxGASS['lgMstar']), "cmap":'cubehelix'}
    error_kwargs = {"lw":.5, "zorder":0}

    detections = xxGASS[xxGASS['HIconf_flag'] == 0]
    non_detections = xxGASS[xxGASS['HIconf_flag'] == -99]
    ax[1,1].scatter(np.log10(detections['SFR_best']), detections['lgMHI'], c = detections['lgMstar'], s = 3, **scatter_kwargs)
    # ax[1,1].errorbar(np.log10(detections['SFR_best']), detections['lgMHI'], xerr = detections['SFRerr_best'], yerr = .2, mfc='k', mec='k', markersize = 0.3, linewidth=.3, markeredgewidth=.3, capthick=.3, fmt = 'o', **error_kwargs)
    ax[3,0].errorbar(np.log10(detections['SFR_best']), detections['lgMHI'], xerr = detections['SFRerr_best'], yerr = .2, mfc='b', mec='b', markersize = 0.3, linewidth=.3, markeredgewidth=.3, capthick=.3, fmt = 'o')
    # ax[1,1].errorbar(np.log10(non_detections['SFR_best']), non_detections['lgMHI'], xerr = non_detections['SFRerr_best'], yerr = 0.2, uplims = True, mfc='k', mec='k', markersize = 0.3, linewidth=.3, markeredgewidth=.3, capthick=.3, fmt = 'o', **error_kwargs)
    sc = ax[1,1].scatter(np.log10(non_detections['SFR_best']), non_detections['lgMHI'], c = non_detections['lgMstar'], s = 3, **scatter_kwargs)
    plt.colorbar(sc)
    for params in samples4[np.random.randint(len(samples4), size = 100)]:
        b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
        ax[1,1].plot(x, (h1*x) + h2, color = 'g', alpha = 0.1)
        ax[3,0].plot(x, (h1*x) + h2, color = 'g', alpha = 0.1)
    for params in samples5[np.random.randint(len(samples5), size = 100)]:
        h1, h2, lnh = params
        ax[3,0].plot(x, (h1*x) + h2, color = 'r', alpha = 0.1)
    ax[1,1].set_xlim(-3,2)
    ax[1,1].set_ylim(7,11)
    ax[2,1].plot(M, np.log10(phi_Baldry))
    ax[2,1].set_ylim(-10,0)
    # ax[2,0].set_ylim(-10,0)
    # plt.legend()
    plt.tight_layout()
    plt.savefig('img/GAMA_sfr_hist.pdf')

def plot_trends(samples3):
    x2 = np.linspace(4,13, 300)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    x, y, yerr = passive_data
    # popt, pcov = curve_fit(fifth_order, x, y)
    # ax[0,0].plot(x2, fifth_order(x2, *popt), color = 'b')
    ax[0,0].errorbar(x, y, yerr, color = 'r')
    # ax[0,0].errorbar(x[:-7], y[:-7], yerr[:-7], color = 'k')
    # popt, pcov = curve_fit(fourth_order, x[:-7], y[:-7])
    popt2, pcov = curve_fit(exp_function, x[:-7], y[:-7])
    popt3, pcov = curve_fit(exp_function2, x, y, p0 = [popt2[0], popt2[1], popt2[2], 11.2])
    print (popt2)
    print (popt3)
    # ax[0,0].plot(x2, fourth_order(x2, *popt), color = 'b')
    # ax[0,0].plot(x2, exp_function(x2, *popt2), color = 'c')
    ax[0,0].plot(x2, exp_function2(x2, *popt3), color = 'm')
    # for b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta in samples3[np.random.randint(len(samples3), size=100)]:
    #     ax[0,0].plot(x2, f_passive(x2, alpha, beta, zeta), color = 'g', alpha = 0.1)
    ax[0,0].set_ylim(0,1.1)
    plt.savefig('img/trends_double_gaussian.pdf')

def sfrmplane(GAMA_sf, GAMA_pass, samples3):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    x = np.linspace(7,12,300)
    print (GAMA_sf['logM*'])
    ax[0,0].scatter(GAMA_sf['logM*'], GAMA_sf['logSFR'], s = 0.1, color = 'b')
    ax[0,0].scatter(GAMA_pass['logM*'], GAMA_pass['logSFR'], s = 0.1, color = 'r')
    for b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta in samples3[np.random.randint(len(samples3), size=100)]:
        ax[0,0].plot(x, third_order(x, b1, b2, b3), alpha = 0.1, color = 'b')
        ax[0,0].plot(x, second_order(x, r1, r2), alpha = 0.1, color = 'r')
    ax[0,0].plot(x, third_order(x, -0.06, +1.8, -12.0))
    # ax[0,0].plot(x, third_order(x, -0.06, +1.95, -14.5))
    # ax[0,0].plot(x, second_order(x, 0.9, -11))
    plt.savefig('img/test.pdf')

def second_order(x, a1, a2):
    return (a1*x) + a2

def third_order(x, a1, a2, a3):
    return (a1*x*x) + (a2*x) + a3

def fourth_order(x, a1, a2, a3, a4):
    return (a1*x*x*x) + (a2*x*x) + (a3*x) + a4

def fifth_order(x, a1, a2, a3, a4, a5):
    return (a1*x*x*x*x) + (a2*x*x*x) + (a3*x*x) + (a4*x) + a5

def exp_function(x, a1, a2, a3):
    return a1*np.exp((a2*x) + a3)

def exp_function2(x, a1, a2, a3, a4):
    y = np.zeros(x.shape)
    y[x <= a4] = a1*np.exp((a2*x[x <= a4]) + a3)
    y[x > a4] = 1
    return y

def mass_functions():
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))

    xbaldry = [7.10, 7.30, 7.5, 7.7, 7.9, 8.1, 8.3, 8.5, 8.7, 8.9, 9.1, 9.3, 9.5, 9.7, 9.9, 10.1, 10.3, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5, 11.7, 11.9]
    baldry = [17.9, 43.1, 31.6, 34.8, 27.3, 28.3, 23.5, 19.2, 18.0, 14.3, 10.2, 9.59, 7.42, 6.21, 5.71, 5.51, 5.48, 5.12, 3.55, 2.41, 1.27, 0.338, 0.042, 0.021, 0.042]
    baldry_err = [5.7, 8.7, 9.0, 8.4, 4.2, 2.8, 3.0, 1.2, 2.6, 1.7, 0.6, 0.55, 0.41, 0.37, 0.35, 0.34, 0.34, 0.33, 0.27, 0.23, 0.16, 0.085, 0.030, 0.021, 0.030]
    baldry = np.array(baldry)/1000
    baldry_err = (np.array(baldry_err)/1000)/(baldry*np.log(10))

    Baldry = {'Mstar': 10.66, 'phistar1': 3.96E-3, 'phistar2': 0.79E-3, 'alpha1': - 0.35, 'alpha2': - 1.47}
    GAMA18 = {'Mstar': 10.78, 'phistar1': 2.93E-3, 'phistar2': 0.63E-3, 'alpha1': - 0.62, 'alpha2': - 1.50}

    M = np.linspace(6,12,100)

    phi_Mstar_Baldry = np.log(10) * np.exp(-np.power(10,M-Baldry['Mstar'])) * \
    (Baldry['phistar1']*np.power(10,(Baldry['alpha1']+1)*(M-Baldry['Mstar'])) + \
    Baldry['phistar2']*np.power(10,(Baldry['alpha2']+1)*(M-Baldry['Mstar'])))

    phi_Mstar_GAMA18 = np.log(10) * np.exp(-np.power(10,M-GAMA18['Mstar'])) * \
    (GAMA18['phistar1']*np.power(10,(GAMA18['alpha1']+1)*(M-GAMA18['Mstar'])) + \
    GAMA18['phistar2']*np.power(10,(GAMA18['alpha2']+1)*(M-GAMA18['Mstar'])))

    ax[0,0].plot(M, np.log10(phi_Mstar_Baldry), label = 'Baldry')
    ax[0,0].plot(M, np.log10(phi_Mstar_GAMA18), label = 'GAMA18')
    ax[0,0].errorbar(xbaldry, np.log10(baldry), yerr = baldry_err, label = 'data', fmt = 'o')
    ax[0,0].set_xlim(6.8,11.6)
    ax[0,0].set_ylim(-5, -1)
    plt.legend()
    plt.savefig('img/mass_functions.pdf')
    return M, phi_Mstar_Baldry, phi_Mstar_GAMA18, xbaldry, np.log10(baldry), baldry_err

def read_GAMA():
    GAMA = pd.read_csv('data/GAMA_sample.dat', comment = '#', header = None, sep = "\s+")
    GAMA.columns = ['CATAID', 'z', 'logM*', 'logM*err', 'logSFR', 'logSFRerr', 'ColorFlag']
    GAMA = GAMA[np.isfinite(GAMA['logSFR'])]
    GAMA = GAMA[np.isfinite(GAMA['logM*'])]
    GAMA = GAMA[GAMA['logM*']>7.0]
    GAMA = GAMA[GAMA['logM*']<12]
    # GAMA = GAMA[GAMA['logSFR']<1.5]
    GAMA = GAMA[GAMA['logSFR'] > - 6.0]
    GAMAb = GAMA[GAMA['ColorFlag']==1]
    GAMAr = GAMA[GAMA['ColorFlag']==2]
    print ('GAMA min and maz zs')
    print (np.min(GAMA['z']))
    print (np.max(GAMA['z']))
    return GAMA, GAMAb, GAMAr

def bin_sfrs(GAMA, sfr_bins):
    n = []
    sfr = []
    for idx in range(0, len(sfr_bins) - 1):
        slice = GAMA[GAMA['logSFR'] > sfr_bins[idx]]
        slice = slice[slice['logSFR'] <= sfr_bins[idx + 1]]
        n.append(len(slice))
        sfr.append((sfr_bins[idx] + sfr_bins[idx + 1])/2)
    return sfr, n

def m_gas_ratio(det):
    # read in the ALFA ALFA datasets from the 40% paper
    ALFAALFA = pd.read_csv('ALFAALFA.csv', comment = '#', header = None, sep=",")
    ALFAALFA.columns = ['x', 'y', 'dy', 'MHI', 'phi', 'err', 'phi_err']
    ALFAALFA = ALFAALFA[np.isfinite(ALFAALFA['phi_err'])]
    MHI_alfa, phi_alfa, phi_err_alfa = np.round(ALFAALFA['MHI'].values,2), ALFAALFA['phi'].values, ALFAALFA['phi_err'].values

    ndim, nwalkers = 3, 100
    g = [1.0, 0.0, -1.1]
    pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    sampler5 = emcee.EnsembleSampler(nwalkers, ndim, MHI_Mstar_fit, pool = pool)
    sampler5.run_mcmc(pos, 1500, progress=True)
    plot_samples_3(sampler5, ndim, 'MHI_scaling')
    samples5 = sampler5.chain[:, 800:, :].reshape((-1, ndim))
    # np.savetxt('data/samples5.txt', samples5)
    fig, ax = plt.subplots(nrows = 1, ncols = 3, squeeze=False, figsize=(18,6))
    ax[0,0].errorbar(det['lgMstar'], det['lgMHI'], xerr  = 0.0, yerr = det['lgMHI_err'], fmt = 'o', markersize = 0.1)
    ax[0,1].errorbar(MHI_alfa, phi_alfa, yerr = phi_err_alfa, fmt='o', capsize = 2, markersize = 3, linewidth=2, markeredgewidth=2, capthick=2, mfc='gray', mec='gray', ecolor = 'gray')
    ax[0,2].errorbar(det['lgMstar'], det['lgMHI']/det['lgMstar'], xerr  = 0, yerr = 0, fmt = 'o')
    x = np.linspace(9,11.5,200)
    MHI = np.linspace(6,13,20)
    for params in samples5[np.random.randint(len(samples5), size=100)]:
        h1, h2, lnh = params
        ax[0,0].plot(x, h1*x + h2, color = 'g', alpha = 0.1)
        phi = []
        for idx, element in enumerate(MHI):
            phi.append(quad(integrand_MHI_direct, 0, 12, args=(element, h1, h2, -10))[0])
        ax[0,1].plot(MHI, np.log10(phi), color = 'g', alpha = 0.1)
    ax[0,0].set_xlabel(r"$\mathrm{log \, M_{*}}$")
    ax[0,1].set_xlabel(r"$\mathrm{log \, M_{HI}}$")
    ax[0,0].set_ylabel(r"$\mathrm{log \, M_{HI}}$")
    ax[0,1].set_ylabel(r"$\mathrm{log \, \phi}$")
    ax[0,1].set_ylim(-10,0)
    plt.savefig('img/atomic_gas_fraction.pdf')

popts = pd.read_csv('bestfits.csv')
mstars = np.linspace(7.6,11.4,39)
bins = np.linspace(-3.5,1.5,51)
sfr_bins = np.linspace(-3.0, 0.6, 19)
GAMA, GAMAb, GAMAr = read_GAMA()
xxGASS, det, nondet = read_GASS()
# calculate error matrices etc
S1 = S_error(det['SFRerr_best'].values, [0.2])
S2 = S_error(nondet['SFRerr_best'].values, [0.2])
x1, y1 = det['SFR_best'].values, det['lgMHI'].values
x2, y2 = nondet['SFR_best'].values, nondet['lgMHI'].values

GAMA_pass = GAMA[GAMA['logSFR'] < third_order(GAMA['logM*'], -0.06, +1.95, -14.5)]
GAMA_sf = GAMA[GAMA['logSFR'] >= third_order(GAMA['logM*'], -0.06, +1.95, -14.5)]
# GAMA_pass = GAMA[GAMA['logSFR'] < (0.83*GAMA['logM*']) + -9.5]
# GAMA_sf = GAMA[GAMA['logSFR'] >= (0.83*GAMA['logM*']) + -9.5]
xnew, std, ratio = bootstrap_GAMA(GAMA_sf, GAMA_pass, 0.5, 1000)
n, sfr = np.histogram(GAMA['logSFR'], sfr_bins)
sfr = ((sfr[1:] + sfr[:-1])/2)
global GASS_data
global GAMA_data
global GAMA_passive
global GAMA_sforming
global passive_data
global sfr_hist_data
# GAMA = GAMA[GAMA['logM*']>9.0]
GASS_data = x1, x2, y1, y2, S1, S2
GAMA_data = GAMA['logM*'].values, GAMA['logSFR'].values, GAMA['logM*err'].values, GAMA['logSFRerr'].values
GAMA_passive = GAMA_pass['logM*'], GAMA_pass['logSFR'], GAMA_pass['logM*err'], GAMA_pass['logSFRerr']
GAMA_sforming = GAMA_sf['logM*'], GAMA_sf['logSFR'], GAMA_sf['logM*err'], GAMA_sf['logSFRerr']
passive_data = xnew, ratio, std
sfr_hist_data = sfr, n

xxGASS, det, nondet = read_GASS()
xxGASS_final = fits.open('data/xGASS_RS_final_Serr_180903.fits')
xxGASS_final = Table(xxGASS_final[1].data).to_pandas()
xxGASS['SNR'] = xxGASS_final['SNR']
xxGASS['MHI_err'] = np.power(10, xxGASS['lgMHI'])/xxGASS['SNR']
xxGASS['lgMHI_err'] = xxGASS['MHI_err']/(np.power(10,xxGASS['lgMHI'])*np.log(10))
xxGASS['lgMstar_err'] = 0.0
det = xxGASS[xxGASS['HIconf_flag']==0]
nondet = xxGASS[xxGASS['HIconf_flag']==-99]
print (xxGASS)
# det = xxGASS[xxGASS['SNR']>0]
global GASS_data2
global GASS_data3
S1 = S_error(det['lgMHI_err'].values, [0.0])
S2 = S_error(nondet['lgMHI_err'].values, [0.0])

GASS_data2 = det['lgMstar'], nondet['lgMstar'], det['lgMHI'], nondet['lgMHI'], S1, S2


# jump = 4
# data = np.zeros((len(mstars) - jump, 7))
# data2 = np.zeros((len(mstars) - jump, 7))
# fig, ax = plt.subplots(nrows = len(mstars) - jump, ncols = 2, squeeze=False, figsize=(12,35), sharex=True)
# offset = np.linspace(0.5,2.0,len(mstars) - jump)
# ms = ((mstars[4:] + mstars[:-4])/2)
# print (ms)
# passive_ratio = (0.02030656*ms*ms*ms) - (0.3111481*ms*ms) + (0.30672944*ms) + 7.95966901
# for idx, row in popts.iterrows():
#     slice = GAMA[GAMA['logM*'] > mstars[idx]]
#     slice = slice[slice['logM*'] <= mstars[idx + jump]]
#     n, bins2, patches = ax[idx,0].hist(slice['logSFR'], bins=bins, color ='r', alpha = 0.3)
#     n, bins2, patches = ax[idx,1].hist(slice['logSFR'], bins=bins, color ='r', alpha = 0.3)
#     ax[idx,0].text(1,0, str(idx) + '    ' + str(round(ms[idx], 1)) )
#     bins2 = (bins2[1:] + bins2[:-1]) / 2
#     x = np.linspace(-3.5,1.5,200)
#     m = (mstars[idx] + (mstars[idx + jump]))/2
#     print (m, row['M'])
#
#     # print (mstars[idx], (mstars[idx + jump]))
#     # popt = [ 0.06162647, -1.14437429, 4.85330247]
#     popt_old = [ -0.06, 1.80, -11.8]
#     poptb = [ -0.13715367,   3.20227314, -18.37979575]
#     poptr = [  0.63334498, -12.93447438,  64.41146806]
#     # popt2 =  [ 0.08746716, -1.8441705,   8.08515159]
#     sfr = (popt_old[0]*ms[idx]*ms[idx]) + (popt_old[1]*ms[idx]) + popt_old[2]
#     # sfr2 = (popt2[0]*m*m) + (popt2[1]*m) + popt2[2]
#     # print (m, sfr)
#     # if idx > 0:
#     #     sfr = popt[1] + 0.1
#     if m <=9.2:
#         popt, pcov = curve_fit(gauss, bins2, n, p0 = [row['B1'], row['Bmean'], row['Bsigma']], maxfev=5000)
#         # ax[idx,1].plot(x, double_gauss(x, *popt), color = 'k')
#         ax[idx,1].plot(x, gauss(x, *popt[:3]), color = 'b')
#         ax[idx,1].plot(x, gauss(x, popt[0], (poptb[0]*m*m) + (poptb[1]*m) + poptb[2], .4), color = 'b', alpha = 0.2, linewidth = 3)
#         # ax[idx,1].plot(x, gauss(x, *popt[3:]), color = 'r')
#     else:
#         popt, pcov = curve_fit(double_gauss, bins2, n, p0 = [row['B1'], row['Bmean'], row['Bsigma'], row['R1'], row['Rmean'], row['Rsigma']], maxfev=5000)
#         # ax[idx,1].plot(x, double_gauss(x, *popt), color = 'k')
#         # ax[idx,1].plot(x, gauss(x, *popt[:3]), color = 'b')
#         # ax[idx,1].plot(x, gauss(x, *popt[3:]), color = 'r')
#         ax[idx,1].plot(x, gauss(x, popt[0], (poptb[0]*m*m) + (poptb[1]*m )+ poptb[2], .4), color = 'b', alpha = 0.2, linewidth = 3)
#         ax[idx,1].plot(x, gauss(x, popt[3], (poptr[0]*m*m) + (poptr[1]*m )+ poptr[2], .5), color = 'r', alpha = 0.2, linewidth = 3)
#         ax[idx,1].plot(x, double_gauss(x, popt[0], (poptb[0]*m*m) + (poptb[1]*m )+ poptb[2], .4, popt[3], (poptr[0]*m*m) + (poptr[1]*m )+ poptr[2], .5), color = 'k', alpha = 0.2, linewidth = 3)
#         data[idx,3], data[idx,4], data[idx,5] = popt[3], popt[4], popt[5]
#     # data2[idx,1:] = popt
#     # data2[idx,0] = ms[idx]
#     data[idx,0], data[idx,1], data[idx,2] = popt[0], popt[1], popt[2]
#     data[idx,6] = ms[idx]
#     # print (popt)
#     # popt2, pcov2 = curve_fit(triple_gauss, bins2, n, p0 = [200.0, -0.34, 0.2, 42.0, -2.1, 0.3, 30.0, -1.21, 0.3], maxfev=2000)
#     ax[idx,0].plot(x, double_gauss(x, *row[['B1', 'Bmean', 'Bsigma', 'R1', 'Rmean', 'Rsigma']].values), color = 'k')
#     ax[idx,0].plot(x, gauss(x, *row[['B1', 'Bmean', 'Bsigma']].values), color = 'b')
#     ax[idx,0].plot(x, gauss(x, *row[['R1', 'Rmean', 'Rsigma']].values), color = 'r')
#
#
#     ax[idx,0].axvline(sfr)
#     ax[idx,0].axvline(sfr-offset[idx])
#     # print (popt)
#     # popt[1] = popt[1] + 0.1
#     # popt[4] = popt[4] + 0.1
#     # print (popt)
#     # plt.plot(x, triple_gauss(x, *popt2))
#     # plt.plot(x, double_gauss(x, 200.0, -0.34, 0.2, 42.0, -2.1, 0.3))
#     # plt.plot(x, triple_gauss(x, 200.0, -0.34, 0.2, 42.0, -2.1, 0.3, 30.0, -1.21, 0.3))
# # np.savetxt('bestfits.csv', data2, delimiter = ',')
# plt.savefig('img/double_gaussian.pdf')




# ndim, nwalkers = 13, 100
# g = [-0.06, +1.8, -12.0, -0.9, .64, -8.23, -1.1, 10.6, -0.96, -2.2, 0.8, 10.0, -1.1]
# pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
pool = Pool(2)
#
# # # emcee using delta from observed SFR histogram
# # sampler3 = emcee.EnsembleSampler(nwalkers, ndim, log_marg_mainsequence_full3, pool = pool)
# # sampler3.run_mcmc(pos, 1000, progress=True)
# # samples3 = sampler3.chain[:, 500:, :].reshape((-1, ndim))
# # plot_samples_full(sampler3, ndim, 'mainsequence_full')
# # np.savetxt('data/samples3.txt', samples3)
#
# # emcee without using delta from observed SFR histogram
# sampler4 = emcee.EnsembleSampler(nwalkers, ndim, log_marg_mainsequence_full2, pool = pool)
# sampler4.run_mcmc(pos, 1500, progress=True)
# plot_samples_full(sampler4, ndim, 'mainsequence_full')
# samples4 = sampler4.chain[:, 800:, :].reshape((-1, ndim))
# np.savetxt('data/samples4.txt', samples4)
#
# ndim, nwalkers = 3, 100
# g = [0.8, 10.0, -1.1]
# pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
#
# sampler5 = emcee.EnsembleSampler(nwalkers, ndim, SFR_HI_fit, pool = pool)
# sampler5.run_mcmc(pos, 1500, progress=True)
# plot_samples_full(sampler5, ndim, 'mainsequence_full')
# samples5 = sampler5.chain[:, 800:, :].reshape((-1, ndim))
# np.savetxt('data/samples5.txt', samples5)

three_dim_plot(det)
# m_gas_ratio(det)
# do the calculation of the galaxy stellar mass functions
# M, phi_Baldry, phi_GAMA18, xbaldry, ybaldry, baldry_err = mass_functions()
# read in the already run and cut chains
# samples4 = np.loadtxt('data/samples4.txt')
# samples5 = np.loadtxt('data/samples5.txt')
# make 8 panel plots howing all the different trends and parameter estimation fits
# sfr_histogram(GAMA, samples4, samples5, M, phi_Baldry)
# plot_trends(samples3)
# sfrmplane(GAMA_sf, GAMA_pass, samples3)
