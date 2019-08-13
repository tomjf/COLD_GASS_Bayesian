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
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
import random
from scipy.optimize import minimize
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import emcee
import corner
from scipy.integrate import quad, dblquad, nquad
from scipy import special
import random
from integrand import integrand_MHI, integrand_MHI_var_sigma, integrand_MHI_double
import os
import time
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
pd.options.mode.chained_assignment = None  # default='warn'
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
import nose

import schechter
import read_files
import utilities
import plots

def read_Lagos_data(fname):
     keres = pd.read_csv(fname, sep=",", header = None)
     keres.columns = ["x", "y", "erup", "erdn"]
     hobs=0.7
     Hubble_h=0.6777
     keres['lgCOB'] = keres['x'] + np.log10((hobs**2)/(Hubble_h**2))
     keres['lgdndlgLCOB'] = keres['y'] + np.log10((hobs**3)/(Hubble_h**3))
     keres['errupB'] = abs(keres['y']-keres['erup'])
     keres['errdnB'] = abs(keres['erdn']-keres['y'])
     X_CO = 2.0
     keres['MH2_x'] = keres['lgCOB'] + np.log10(580.0*X_CO) + (2*np.log10(2.6)) - np.log10(4*np.pi)
    #  print (keres)
     return keres


def extrapolate(data, up, down, n):
    x = np.linspace(down,up,n)
    newdata = np.zeros((len(x),2))
    for j in range(0,len(x)):
        for i in range(0,len(data)-1):
            dx = data[i+1,0] - data[i,0]
            dy = data[i+1,1] - data[i,1]
            y = data[i,1]
            if x[j] < data[i+1,0] and x[j]>=data[i,0]:
                DX = x[j] - data[i,0]
                if DX == 0:
                    newdata[j,1] = y
                    newdata[j,0] = x[j]
                else:
                    ans = y + ((DX/dx)*dy)
                    newdata[j,1] = ans
                    newdata[j,0] = x[j]
            elif x[j]<data[0,0]:
                dx = data[1,0] - data[0,0]
                dy = data[1,1] - data[0,1]
                DX = x[j] - data[0,0]
                y = data[0,1]
                ans = y + ((DX/dx)*dy)
                newdata[j,1] = ans
                newdata[j,0] = x[j]
            elif x[j]>data[-1,0]:
                dx = data[-1,0] - data[-2,0]
                dy = data[-1,1] - data[-2,1]
                DX = x[j] - data[-1,0]
                y = data[-1,1]
                ans = y + ((DX/dx)*dy)
                newdata[j,1] = ans
                newdata[j,0] = x[j]
    return newdata

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value



# Defining error shading for fit on scaling relation plot:
def shading_linear(sampler_input, samples_input, x_input, chain_discard):
   lnprob = sampler_input.lnprobability # taking all log probabilities (except first 200)
   # print (np.shape(lnprob))
   lnprob = sampler_input.lnprobability[500:, :].reshape(-1)
   # print (np.shape(lnprob))
   # print (lnprob)
   posterior_percentile = np.percentile(lnprob, 31.7) # taking only probabilities within 1sigma
   # print (posterior_percentile)
   onesigma = samples_input[np.where(lnprob > posterior_percentile)] # taking samples from these 1sigma probabilities
   # print (np.shape(onesigma))
   # print (onesigma)
   # Building error region for shading fit
   y_fits = []
   for i in range(len(onesigma)):
       # print (onesigma[i])
       params = onesigma[i][1], onesigma[i][0], onesigma[i][2]
       y_fits.append(schechter.single_schechter2(x_input, params))
   y_fits = np.array(y_fits)
   y_max = []
   y_min = []
   for i in range(len(x_input)): # for each x interval, find max and min of fits to shade between
       y_max.append(max(y_fits[:, i]))
       y_min.append(min(y_fits[:, i]))
   y_max = np.array(y_max)
   y_min = np.array(y_min)
   return y_max, y_min

def plot_samples_full(sampler, ndim, fname, l):
    fig, axes = plt.subplots(ndim, figsize=(10, 20), sharex=True)
    samples = sampler.get_chain()
    labels = l
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    plt.savefig('img/sampler' + fname + '.pdf')

def plot_corner_full(samples_input, fname):
    corner.corner(  samples_input,
                    labels=[r"$\rho$", r"$M_{*}$", r"$\alpha$"],
                    truths=(np.median(samples_input[:, 0]),
                    np.median(samples_input[:, 1]),
                    np.median(samples_input[:, 2])),
                  truth_color="k",
                  quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('img/corner/' + fname)


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

def OmegaH2(bins, yrho):
    rhocrit = 9.2*(10**(-27))
    dMH2 = bins[1] - bins[0]
    rhoH2 = (np.sum((10**yrho)*dMH2)*(2*(10**30)))/((3.086*(10**22))**3)
    OmegaH2 = (rhoH2/rhocrit)
    return OmegaH2

def schechfunc(M, rhostar, Mstar, alpha):
    y = np.log(10.0) * rhostar * ((M / Mstar) ** (alpha + 1)) * np.exp(-1 * M / Mstar)
    return y

def log_marg_mainsequence_full2(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Mstar, phistar, alpha, lnsigma = params
    x, y, yerr = passive_data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # star forming likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_sf = np.square(xerrb)*np.square((2*b1*xb) + b2) + np.square(yerrb) + np.exp(2*lnb)
    DeltaN = yb - (b1*xb*xb) - (b2*xb) - b3
    ll_sf = -0.5 * np.sum(DeltaN**2/Sigma2_sf + np.log(Sigma2_sf))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return ll_sf + log_mainsequence_priors_full2(params)

def bootstrap_with_replacement(df, n, bins):
    minm, maxm = min(bins), max(bins)
    dm = bins[1] - bins[0]
    rows = df.index.tolist()
    countTOT = []
    for i in range(n):
        countTOT_j = np.zeros(len(bins) - 1, dtype=int)
        bootstrapped_rows = np.random.choice(rows, len(rows), replace = True)
        # print (np.sort(bootstrapped_rows))
        bootstrapped = df.ix[np.sort(bootstrapped_rows)]
        # print (bootstrapped)
        Nm = bootstrapped['MH2'].values
        weightNEW = bootstrapped['WEIGHT'].values
        for j in range(len(bins) - 1):
            indE = np.where((Nm > minm + j * dm) & (Nm < minm + (j + 1) * dm))
            if len(Nm[indE]) == 0:  # if no indices fitting the criteria for the bin i were found, add 0 to count at i
                countTOT_j[j] = 0
            else:
                countTOT_j[j] = np.sum(weightNEW[indE])  # sum all weights found and add to index i in count
        countTOT.append(countTOT_j)
    countTOT = np.array(countTOT)
    countNorm = countTOT / dm
    countNorm = countNorm / df['V_m'].values[0]
    error1 = []
    for i in range(len(bins) - 1):
        error1_i = []
        for j in range(n):
            error1_i.append(countNorm[j, i])
        error1.append(np.std(error1_i))
    error1 = np.array(error1)
    # std = np.std(datarr, axis = 0)
    return error1

def bootstrap_with_replacement2(df, n, bins):
    minm, maxm = min(bins), max(bins)
    dm = bins[1] - bins[0]
    rows = df.index.tolist()
    countTOT = []
    for i in range(n):
        countTOT_j = np.zeros(len(bins) - 1, dtype=int)
        bootstrapped_rows = np.random.choice(rows, len(rows), replace = True)
        # print (np.sort(bootstrapped_rows))
        bootstrapped = df.ix[np.sort(bootstrapped_rows)]
        # print (bootstrapped)
        Nm = bootstrapped['LCO_COR'].values
        weightNEW = bootstrapped['WEIGHT'].values
        for j in range(len(bins) - 1):
            indE = np.where((Nm > minm + j * dm) & (Nm < minm + (j + 1) * dm))
            if len(Nm[indE]) == 0:  # if no indices fitting the criteria for the bin i were found, add 0 to count at i
                countTOT_j[j] = 0
            else:
                countTOT_j[j] = np.sum(weightNEW[indE])  # sum all weights found and add to index i in count
        countTOT.append(countTOT_j)
    countTOT = np.array(countTOT)
    countNorm = countTOT / dm
    countNorm = countNorm / df['V_m'].values[0]
    error1 = []
    for i in range(len(bins) - 1):
        error1_i = []
        for j in range(n):
            error1_i.append(countNorm[j, i])
        error1.append(np.std(error1_i))
    error1 = np.array(error1)
    # std = np.std(datarr, axis = 0)
    return error1

def bootstrap_with_replacement3(df, n, bins):
    minm, maxm = min(bins), max(bins)
    dm = bins[1] - bins[0]
    rows = df.index.tolist()
    countTOT = []
    for i in range(n):
        countTOT_j = np.zeros(len(bins) - 1, dtype=int)
        bootstrapped_rows = np.random.choice(rows, len(rows), replace = True)
        # print (np.sort(bootstrapped_rows))
        bootstrapped = df.ix[np.sort(bootstrapped_rows)]
        # print (bootstrapped)
        Nm = bootstrapped['LCO_estimated'].values
        weightNEW = bootstrapped['WEIGHT'].values
        for j in range(len(bins) - 1):
            indE = np.where((Nm > minm + j * dm) & (Nm < minm + (j + 1) * dm))
            if len(Nm[indE]) == 0:  # if no indices fitting the criteria for the bin i were found, add 0 to count at i
                countTOT_j[j] = 0
            else:
                countTOT_j[j] = np.sum(weightNEW[indE])  # sum all weights found and add to index i in count
        countTOT.append(countTOT_j)
    countTOT = np.array(countTOT)
    countNorm = countTOT / dm
    countNorm = countNorm / df['V_m'].values[0]
    error1 = []
    for i in range(len(bins) - 1):
        error1_i = []
        for j in range(n):
            error1_i.append(countNorm[j, i])
        error1.append(np.std(error1_i))
    error1 = np.array(error1)
    # std = np.std(datarr, axis = 0)
    return error1

def throw_error(df, bins, frac):
    throws = len(df)
    minm, maxm = 7.69, 10.15
    dm = bins[1] - bins[0]
    num = int(throws*frac)
    datarr = np.zeros((num, len(bins)-1))
    rows = df.index.tolist()
    countTOT = []
    for i in range(0, num):
        countTOT_j = np.zeros(len(bins) - 1, dtype=int)
        newrows = random.sample(rows, num)
        newdf = df.ix[newrows]
        for index, row in newdf.iterrows():
            if row['LOGMH2_ERR'] == 0:
                newdf.ix[index, 'new_LOGMH2'] = row['MH2']
            else:
                # newdf.ix[index, 'new_LOGMH2'] = np.random.normal(loc = row['LOGMH2'], scale = row['LOGMH2_ERR'])
                gaus = np.random.normal(row['LOGMH2'], row['LOGMH2_ERR'], num)
                newdf.ix[index, 'new_LOGMH2'] = np.random.choice(gaus, 1)
        # rho = Schechter(newdf, bins)
        Nm = newdf['new_LOGMH2'].values
        weightNEW = newdf['WEIGHT'].values / 1.252
        for i in range(len(bins) - 1):
            # massE[i] = minm + (i + 0.5) * dm  # 0.5 because we are taking the middle of the bin
            indE = np.where((Nm > minm + i * dm) & (Nm < minm + (i + 1) * dm))
            if len(Nm[indE]) == 0:  # if no indices fitting the criteria for the bin i were found, add 0 to count at i
                countTOT_j[i] = 0
            else:
                countTOT_j[i] = np.sum(weightNEW[indE])  # sum all weights found and add to index i in count
        countTOT.append(countTOT_j)
        # datarr[i,:] = rho[1]
    countTOT = np.array(countTOT)
    # print (np.sum(countTOT[0,:]))
    # print (countTOT)
    countNorm = countTOT / dm
    countNorm = countNorm / newdf['V_m2'].values[0]
    error1 = []
    for i in range(len(bins) - 1):
        error1_i = []
        for j in range(num):
            error1_i.append(countNorm[j, i])
        error1.append(np.std(error1_i))
    error1 = np.array(error1)
    # std = np.std(datarr, axis = 0)
    return error1

def Schechter(data, bins):
    l = data['MH2'].values
    w = data['WEIGHT'].values
    Vm = data['V_m'].values
    # l = np.log10(l)
    rho, N, xbins, sigma, rhoH2 = [], [], [], [], []
    for i in range (1,len(bins)):
        p, Num, o, pH2 = 0, 0, 0, 0
        for j in range(0,len(l)):
            if l[j] >= bins[i-1] and l[j] < bins[i]:
                p += w[j]/Vm[j]
                o += 1/(Vm[j]**2)
                pH2 += l[j]/Vm[j]
                Num+=1
        N.append(Num)
        xbins.append((bins[i]+bins[i-1])/2)
        rho.append(p/(bins[1]-bins[0]))
        sigma.append(math.sqrt(o))
        rhoH2.append(pH2/(bins[1]-bins[0]))
    # return the Number of gals, log10(density), centre pt of each bin
    return [N, rho, xbins, np.log10(sigma), np.log10(rhoH2)]

def Schechter2(data, bins):
    l = data['new_LOGMH2'].values
    v = data['V_m'].values
    # l = np.log10(l)
    rho, N, xbins, sigma, rhoH2 = [], [], [], [], []
    for i in range (1,len(bins)):
        p, Num, o, pH2 = 0, 0, 0, 0
        for j in range(0,len(l)):
            if l[j] >= bins[i-1] and l[j] < bins[i]:
                p += 1/v[j]
                o += 1/(v[j]**2)
                pH2 += l[j]/v[j]
                Num+=1
        N.append(Num)
        xbins.append((bins[i]+bins[i-1])/2)
        rho.append(p/(bins[1]-bins[0]))
        sigma.append(math.sqrt(o))
        rhoH2.append(pH2/(bins[1]-bins[0]))
    # return the Number of gals, log10(density), centre pt of each bin
    return [N, np.log10(rho), xbins, np.log10(sigma), np.log10(rhoH2)]

def Schechter3(data, bins):
    l = data['LCO_COR'].values
    v = data['V_m'].values
    w = data['WEIGHT'].values
    # l = np.log10(l)
    rho, N, xbins, sigma, rhoH2 = [], [], [], [], []
    for i in range (1,len(bins)):
        p, Num, o, pH2 = 0, 0, 0, 0
        for j in range(0,len(l)):
            if l[j] >= bins[i-1] and l[j] < bins[i]:
                p += w[j]/v[j]
                o += 1/(v[j]**2)
                pH2 += l[j]/v[j]
                Num+=1
        N.append(Num)
        xbins.append((bins[i]+bins[i-1])/2)
        rho.append(p/(bins[1]-bins[0]))
        sigma.append(math.sqrt(o))
        rhoH2.append(pH2/(bins[1]-bins[0]))
    # return the Number of gals, log10(density), centre pt of each bin
    return [N, rho, xbins, np.log10(sigma), np.log10(rhoH2)]

def Schechter4(data, bins):
    l = data['LCO_estimated'].values
    v = data['V_m'].values
    w = data['WEIGHT'].values
    # l = np.log10(l)
    rho, N, xbins, sigma, rhoH2 = [], [], [], [], []
    for i in range (1,len(bins)):
        p, Num, o, pH2 = 0, 0, 0, 0
        for j in range(0,len(l)):
            if l[j] >= bins[i-1] and l[j] < bins[i]:
                p += w[j]/v[j]
                o += 1/(v[j]**2)
                pH2 += l[j]/v[j]
                Num+=1
        N.append(Num)
        xbins.append((bins[i]+bins[i-1])/2)
        rho.append(p/(bins[1]-bins[0]))
        sigma.append(math.sqrt(o))
        rhoH2.append(pH2/(bins[1]-bins[0]))
    # return the Number of gals, log10(density), centre pt of each bin
    return [N, rho, xbins, np.log10(sigma), np.log10(rhoH2)]

def log_schechter(logL, log_phi, log_L0, alpha):
    log = np.log(10)
    frac = np.power(10,logL-log_L0)
    exp = np.exp(-frac)
    phibit1 = (log_phi*(frac**(alpha+1)))
    return exp*log*phibit1
    # schechter = log_phi
    # schechter += (alpha+1)*(logL-log_L0)*np.log(10)
    # schechter -= pow(10,logL-log_L0)
    # return schechter

def log_double_schechter_true(logL, log_phi1, log_phi2, log_L0, alpha1, alpha2):
    log = np.log(10)
    frac1 = np.power(10,(alpha1+1)*(logL-log_L0))
    frac2 = np.power(10,(alpha2+1)*(logL-log_L0))
    exp = np.exp(-np.power(10,logL-log_L0))
    return log*exp*(log_phi1*frac1 + log_phi2*frac2)

def log_schechter_true(logL, log_phi, log_L0, alpha):
    # print (log_phi, log_L0, alpha)
    log = np.log(10)
    frac = np.power(10,(alpha+1)*(logL-log_L0))
    exp = np.exp(-np.power(10,logL-log_L0))
    return log*log_phi*frac*exp

# Functions for the ML fitting ##################################################
def log_likelihood(theta, x, y, yerr):
    rhostar, logM, alpha = theta
    model = np.log(10.0) * rhostar * np.power((np.power(10, x) / np.power(10, logM)), (alpha + 1)) * np.exp(-1 * np.power(10, x) / np.power(10, logM))
    #inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
    #return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))

def log_prior(theta):
    rhostar, logM, alpha = theta
    if 0.000001 < rhostar < 0.005 and 7.0 < logM < 10.5 and -2.0 < alpha < 1.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

# Building x- and y-error matrix (in our case it is diagonal since errors aren't correlated)
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

def log_prior_SFR_MH2(params):
    c1, c2, lnf3 = params
    if 0.4 < c1 < 1.0 and \
    8.0 < c2 < 10.0 and \
    -5.0 < lnf3 < 5.0:
        return 0.0
    return -np.inf

def log_prob_SFR_MH2(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    c1, c2, lnf3 = params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read in the data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    CGx1, CGx2, CGy1, CGy2, CGS1, CGS2 = COLD_GASS_data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood for the linear SFR-MH2 plane
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = np.array([-c1, 1.0])
    sigma = np.dot(np.dot(CGS1, v), v) + np.exp(2 * lnf3)
    sigma2 = np.dot(np.dot(CGS2, v), v) + np.exp(2 * lnf3)
    sigma2 = sigma2 ** 0.5
    deltaN = CGy1 - (c1 * CGx1) - c2
    model = (c1 * CGx2) + c2
    ll1 = -0.5 * np.sum(np.square(deltaN) / sigma + np.log(sigma))
    I = np.zeros(len(CGx2))
    for i in range(0,len(CGx2)):
        I[i] = np.log(((2 * np.pi) ** 0.5) *
                      0.5 * (special.erf((CGy2[i]-model[i]) / ((2 ** 0.5) * sigma2[i])) + 1))
    ll2 = np.sum(I)
    ll_SFR_MH2 = ll1  + ll2
    return ll_SFR_MH2 + log_prior_SFR_MH2(params)

def log_prior_SFR_MHI(params):
    c1, c2, lnf3 = params
    if 0.4 < c1 < 1.4 and \
    8.0 < c2 < 12.0 and \
    -5.0 < lnf3 < 5.0:
        return 0.0
    return -np.inf

def log_prob_SFR_MHI(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    c1, c2, lnf3 = params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read in the data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Gx1, Gx2, Gy1, Gy2, GS1, GS2 = GASS_data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood for the linear SFR-MH2 plane
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = np.array([-c1, 1.0])
    sigma = np.dot(np.dot(GS1, v), v) + np.exp(2 * lnf3)
    sigma2 = np.dot(np.dot(GS2, v), v) + np.exp(2 * lnf3)
    sigma2 = sigma2 ** 0.5
    deltaN = Gy1 - (c1 * Gx1) - c2
    model = (c1 * Gx2) + c2
    ll1 = -0.5 * np.sum(np.square(deltaN) / sigma + np.log(sigma))
    I = np.zeros(len(Gx2))
    for i in range(0,len(Gx2)):
        I[i] = np.log(((2 * np.pi) ** 0.5) *
                      0.5 * (special.erf((Gy2[i]-model[i]) / ((2 ** 0.5) * sigma2[i])) + 1))
    ll2 = np.sum(I)
    ll_SFR_MHI = ll1  + ll2
    return ll_SFR_MHI + log_prior_SFR_MHI(params)

def LCO_plot(prods_LCO, prods_LCOd, prods_LCO_e):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(5, 5))
    # ax2 = ax[0,0].twinx()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read in and process the scraped data from the keres 2003 paper
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    keres2003 = pd.read_csv('data/keres2003.csv')
    keres2003['y2'] = np.power(10, keres2003['y'])
    keres2003['dyu'] = keres2003['yu'] - keres2003['y']
    keres2003['dlogyu'] = keres2003['dyu'] * keres2003['y2'] * np.log(10)
    keres2003['dyl'] = np.abs(keres2003['yl'] - keres2003['y'])
    keres2003['dlogyl'] = keres2003['dyl'] * keres2003['y2'] * np.log(10)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # work out the factors for the conversion between axis units
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Mx = np.linspace(0, 12.0, 500)
    L = np.linspace(6, 10.5, 300)
    keres_params = np.log10((2.81/(0.7**2))*np.power(10,9)), 0.0089*(0.7**3), -1.18
    keres_params_LCO = 7.0, 0.00072, -1.30
    keres_params_LCO1 = 7.0 + np.log10((0.75**2)/(0.7**2)), 0.00072*((0.7**3)/(0.75**3)), -1.30
    LCO = Mx - 0.477#np.log10(6.421)
    LCO_K = (3.25*np.power(10,7)*L)/(4*np.pi*np.square(115.3))
    factor = np.log10((3.25*np.power(10,7))/(np.square(115.3)*4*np.pi))
    factor_sq = (0.70**2)/(0.75**2)
    factor_cube = (0.70**3)/(0.75**3)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # do the emcee parameter estimation for the LCO Schechter fit
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nwalkers, ndim = 200, 3
    g = [0.001, 8.5, -1.1]
    pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    param_names = ['rhostar', 'logM', 'alpha']
    M = np.linspace(6.0, 10.5, 300)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # emcee infer Schechter parameters for all the data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start_i = np.where(np.array(prods_LCO[2]) >= 7.5)[0][0]
    sampler5 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (prods_LCO[2][start_i:], prods_LCO[1][start_i:], error_arr_LCO[start_i:]))
    sampler5.run_mcmc(pos, 1000)
    samples5 = sampler5.chain[:, 500:, :].reshape((-1, ndim))
    phistar = (np.median(samples5[:, 0]))
    Mstar = (np.median(samples5[:, 1]))
    alpha = (np.median(samples5[:, 2]))
    print (phistar, Mstar, alpha)
    ymin, ymax = shading_linear(sampler5, samples5, M, 500)
    ax[0,0].fill_between(M, ymin, ymax, color = 'lightcoral', alpha = 0.2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # emcee infer Schechter parameters for just the detections
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start_i = np.where(np.array(prods_LCOd[2]) >= 7.5)[0][0]
    sampler5 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (prods_LCOd[2][start_i:], prods_LCOd[1][start_i:], error_arr_LCOd[start_i:]))
    sampler5.run_mcmc(pos, 1000)
    samples5 = sampler5.chain[:, 500:, :].reshape((-1, ndim))
    phistar1 = (np.median(samples5[:, 0]))
    Mstar1 = (np.median(samples5[:, 1]))
    alpha1 = (np.median(samples5[:, 2]))
    print (phistar1, Mstar1, alpha1)
    # np.savetxt('data/samples5.txt', samples5)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # emcee infer Schechter parameters for estimated LCO data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start_i = np.where(np.array(prods_LCO_e[2]) >= 7.5)[0][0]
    sampler6 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (prods_LCO_e[2][start_i:], prods_LCO_e[1][start_i:], error_arr_LCO_e[start_i:]))
    sampler6.run_mcmc(pos, 1000)
    samples6 = sampler6.chain[:, 500:, :].reshape((-1, ndim))
    plot_corner_full(samples6, 'corner_LCO')
    phistar2 = (np.median(samples6[:, 0]))
    Mstar2 = (np.median(samples6[:, 1]))
    alpha2 = (np.median(samples6[:, 2]))
    print ('estimated LCOs', phistar2, Mstar2, alpha2)
    pcs_phistar = np.percentile(samples6[:, 0], [16, 50, 84], axis=0)
    pcs_Mstar = np.percentile(samples6[:, 1], [16, 50, 84], axis=0)
    pcs_alpha = np.percentile(samples6[:, 2], [16, 50, 84], axis=0)
    print (round(pcs_phistar[1], 5), round(pcs_phistar[0]-pcs_phistar[1], 5), round(pcs_phistar[2]-pcs_phistar[1], 5))
    print (round(pcs_Mstar[1], 2), round(pcs_Mstar[0]-pcs_Mstar[1], 2), round(pcs_Mstar[2]-pcs_Mstar[1], 2))
    print (round(pcs_alpha[1], 2), round(pcs_alpha[0]-pcs_alpha[1], 2), round(pcs_alpha[2]-pcs_alpha[1], 2))
    # np.savetxt('data/samples5.txt', samples5)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # make the plot
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # keres data
    ax[0,0].plot(Mx+factor, schechter.single_schechter2(Mx, keres_params_LCO1), color = 'k', linestyle = '--', linewidth = .5, label = "$\mathrm{K+03}$", zorder = 1)
    # ax2.errorbar(   keres2003['x'] + factor - np.log10(factor_sq), keres2003['y'] + np.log10(factor_cube),
    #                     yerr = [keres2003['dyl'], keres2003['dyu']],
    #                     fmt="s", ecolor = 'k', mec = 'k', ms = 4,
    #                     mfc ='gray', zorder = 2, capsize=2, label="$\mathrm{Keres+03}$")
    # ax[0,0].errorbar(   0, 0,
    #                     yerr = 1,
    #                     fmt="s", ecolor = 'k', mec = 'k', ms = 4,
    #                     mfc ='gray', zorder = 2, capsize=2, label="$\mathrm{Keres+03}$")
    # plot the xCOLD GASS bins
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar*np.power(10,(alpha+1)*(M-Mstar)))
    ax[0,0].plot(M, phi_Mstar_double, color = 'crimson', zorder = 96)
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar1)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar1)))
    ax[0,0].plot(M, phi_Mstar_double, color = 'royalblue', zorder = 97)
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar2)) * (phistar2*np.power(10,(alpha2+1)*(M-Mstar2)))
    ax[0,0].plot(M, phi_Mstar_double, color = 'darkgreen', zorder = 95)
    ymin, ymax = shading_linear(sampler5, samples5, M, 500)
    ax[0,0].fill_between(M, ymin, ymax, color = 'lightskyblue', alpha = 0.2)
    ymin, ymax = shading_linear(sampler6, samples6, M, 500)
    ax[0,0].fill_between(M, ymin, ymax, color = 'mediumseagreen', alpha = 0.2)

    ax[0,0].errorbar(prods_LCO[2], prods_LCO[1], yerr = error_arr_LCO, fmt="h",
                        ecolor = 'crimson', mec = 'crimson', mfc ='lightcoral',
                        zorder = 98, capsize=2, label = "$\mathrm{xCOLD \, GASS \, detections \, and \, non-detections}$")
    ax[0,0].errorbar(prods_LCOd[2], prods_LCOd[1], yerr = error_arr_LCOd, fmt="^",
                        ecolor = 'royalblue', mec = 'royalblue', mfc ='lightskyblue',
                        zorder = 99, capsize=2, label = "$\mathrm{xCOLD \, GASS \, detections}$")
    ax[0,0].errorbar(prods_LCO_e[2], prods_LCO_e[1], yerr = error_arr_LCOd, fmt="^",
                        ecolor = 'darkgreen', mec = 'darkgreen', mfc ='mediumseagreen',
                        zorder = 99, capsize=2, label = "$\mathrm{xCOLD \, GASS \, estimated}$")
    # vallini data
    data = pd.read_csv('data/vallini/test33a.csv').values
    data[:,1] = np.power(10,data[:,1])
    uperr = pd.read_csv('data/vallini/uperr.csv').values
    lowerr = pd.read_csv('data/vallini/lowerr.csv').values
    lowerr = lowerr[lowerr[:,0].argsort()]
    lowerr2 = extrapolate(lowerr, 5.5, 11, 200)
    uperr2 = extrapolate(uperr, 5.5, 11, 200)
    ax[0,0].plot(data[:,0], data[:,1], label = "$\mathrm{Vallini+16}$", color = 'k', linewidth=1, zorder = 2, linestyle='-')
    # completeness limits
    ax[0,0].axvline(7.5, color = 'k', linestyle = '-.', linewidth = .5, label = "$\mathrm{xCOLD \, GASS \, \log M_{*} \, completeness \, limit}$", zorder = 0)
    ax[0,0].axvline(8.6, color = 'k', linestyle = ':', linewidth = .5, label = "$\mathrm{xCOLD \, GASS \, integration \, completeness \, limit}$", zorder = 0)
    # setting all the axes parameters
    ax[0,0].set_yscale('log')
    ax[0,0].set_xlim(6.0, 10.5)
    ax[0,0].set_ylim(0.0000001, 0.1)
    # ax2.set_xlim(6,10.5)
    # ax2.set_ylim(-7,-1)
    ax[0,0].set_xlabel("$\mathrm{\log_{10}\, L_{CO} \, [K\, km\, s^{-1} \, pc^{2}]}$")
    ax[0,0].set_ylabel("$\mathrm{\log_{10}\phi(L_{CO}) \, [Mpc^{-3} dex^{-1}]}$")
    ax[0,0].legend(fontsize = 8, facecolor = 'w', framealpha = 1)
    # ax2.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig('img/LCO_final.pdf')

def MH2_plot(prods, prods_d, prods_d2, samples5, samples6, samples7, sampler5, sampler6, sampler7):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, squeeze=False, figsize=(10, 5))
    keres_params = np.log10((2.81/(0.7**2))*np.power(10,9)), 0.0089*(0.7**3), -1.18
    omegas, omegas2, omegas3 = [], [], []
    N = 10000
    omegas_all = np.zeros((N,3))
    i = 0
    for m1, b, f1 in samples5[np.random.randint(len(samples5), size=N)]:
        om1 = OmegaH2(M2, np.log10(np.power(10,M2)*schechfunc(np.power(10,M2),m1, 10**b,f1)))*np.power(10,5)*0.7
        omegas.append(om1)
    for m1, b, f1 in samples6[np.random.randint(len(samples6), size=N)]:
        om2 = OmegaH2(M2, np.log10(np.power(10,M2)*schechfunc(np.power(10,M2),m1, 10**b,f1)))*np.power(10,5)*0.7
        omegas2.append(om2)
    for m1, b, f1 in samples7[np.random.randint(len(samples7), size=N)]:
        om3 = OmegaH2(M2, np.log10(np.power(10,M2)*schechfunc(np.power(10,M2),m1, 10**b,f1)))*np.power(10,5)*0.7
        omegas3.append(om3)
        i+=1
    omegas_all[:,0] = omegas2
    omegas_all[:,1] = omegas3
    omegas_all[:,2] = omegas
    M3 = np.linspace(7.5,10.5, 100)
    ymin5, ymax5 = shading_linear(sampler5, samples5, M3, 500)
    ymin6, ymax6 = shading_linear(sampler6, samples6, M3, 500)
    ymin7, ymax7 = shading_linear(sampler7, samples7, M3, 500)
    Mstar = 9.35
    phistar1 = 4.29E-3
    alpha1 = -1.03
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))
    phi_Mstar_double2 = np.log(10) * np.exp(-np.power(10,M-Mstar2)) * (phistar2*np.power(10,(alpha2+1)*(M-Mstar2)))
    phi_Mstar_double3 = np.log(10) * np.exp(-np.power(10,M-Mstar3)) * (phistar3*np.power(10,(alpha3+1)*(M-Mstar3)))
    phi_Mstar_double4 = np.log(10) * np.exp(-np.power(10,M-Mstar4)) * (phistar4*np.power(10,(alpha4+1)*(M-Mstar4)))
    obreschkow_params = np.log10(7.5/(0.7**2)*np.power(10,8)), 0.0243*(0.7**3), -1.07

    Mx = np.linspace(0, 12.0, 500)
    keres_params_LCO1 = 7.0 + np.log10((0.75**2)/(0.7**2)), 0.00072*((0.7**3)/(0.75**3)), -1.30
    factor = np.log10((3.25*np.power(10,7))/(np.square(115.3)*4*np.pi))
    factor_sq = (0.70**2)/(0.75**2)
    factor_cube = (0.70**3)/(0.75**3)
    print (np.log10(factor_sq))
    print ('factor cube', factor_cube)
    ax[0,0].plot(Mx + factor + np.log10(4.76), schechter.single_schechter2(Mx, keres_params_LCO1), color = 'g', linestyle = '--', linewidth = .5, label = "$\mathrm{K+03}$", zorder = 1)
    ax[0,0].plot(Mx + factor + np.log10(4.76) - np.log10(factor_sq) , schechter.single_schechter2(Mx, keres_params_LCO1)*factor_cube, color = 'g', linestyle = '-', linewidth = .5, label = "$\mathrm{K+03}$", zorder = 1)

    ax[0,0].fill_between(M3, ymin5, ymax5, color = 'lightcoral', alpha = 0.2)
    ax[0,0].fill_between(M3, ymin6, ymax6, color = 'lightskyblue', alpha = 0.2)
    ax[0,0].fill_between(M3, ymin7, ymax7, color = 'mediumseagreen', alpha = 0.2)
    ax[0,0].plot(M, phi_Mstar_double2, color = 'crimson')
    ax[0,0].plot(M, phi_Mstar_double3, color = 'royalblue')
    ax[0,0].plot(M, phi_Mstar_double4, color = 'darkgreen')
    ax[0,0].plot(M, schechter.single_schechter2(M, keres_params), color = 'k', linestyle = '-', linewidth = .5, label = "$\mathrm{K+03}$")
    ax[0,0].plot(M, schechter.single_schechter2(M, obreschkow_params), color = 'k', linestyle = '--', linewidth = .5, label = "$\mathrm{O+09}$")
    ax[0,0].errorbar(   prods[2], prods[1], yerr = error_arr, fmt="^",
                        ecolor = 'crimson', mec = 'crimson', mfc ='lightcoral',
                        zorder = 99, capsize=2,
                        label = "$\mathrm{xCOLD \, GASS \, detections \, and \, non-detections}$")
    ax[0,0].errorbar(   prods_d[2], prods_d[1], yerr = error_arr_d, fmt="h",
                        ecolor = 'royalblue', mec = 'royalblue', mfc ='lightskyblue',
                        zorder = 99, capsize=2,
                        label = "$\mathrm{xCOLD \, GASS \, detections \, only}$")
    ax[0,0].errorbar(   prods_d2[2], prods_d2[1], yerr = error_arr_d2, fmt=".",
                        ms = 10, ecolor = 'darkgreen', mec = 'darkgreen',
                        mfc ='mediumseagreen', zorder = 99, capsize=2,
                        label = "$\mathrm{xCOLD \, GASS \, detections \, and \, estimates}$")
    ax[0,0].axvline(8.2, color = 'k', linestyle = '-.', linewidth = .5, label = "$\mathrm{xCOLD \, GASS \, \log M_{*} \, completeness \, limit}$", zorder = 0)
    ax[0,0].axvline(9.2, color = 'k', linestyle = ':', linewidth = .5, label = "$\mathrm{xCOLD \, GASS \, integration \, completeness \, limit}$", zorder = 0)
    ax[0,0].set_ylim(0.00001, 0.1)
    ax[0,0].set_xlim(7.5, 10.5)
    ax[0,0].set_yscale('log')
    ax[0,0].set_xlabel("$\mathrm{\log_{10}(M_{H2}) \, [M_\odot]}$")
    ax[0,0].set_ylabel("$\mathrm{\log_{10}\phi(M_{H2}) \, [Mpc^{-3} dex^{-1}]}$")
    ax[0,0].legend(fontsize = 8)

    ax[0,1].fill_between(M3, np.power(10,M3)*ymin5, np.power(10,M3)*ymax5, color = 'lightcoral', alpha = 0.2)
    ax[0,1].fill_between(M3, np.power(10,M3)*ymin6, np.power(10,M3)*ymax6, color = 'lightskyblue', alpha = 0.2)
    ax[0,1].fill_between(M3, np.power(10,M3)*ymin7, np.power(10,M3)*ymax7, color = 'mediumseagreen', alpha = 0.2)
    ax[0,1].plot(M, np.power(10,M)*phi_Mstar_double2, color = 'crimson')
    ax[0,1].plot(M, np.power(10,M)*phi_Mstar_double3, color = 'royalblue')
    ax[0,1].plot(M, np.power(10,M)*phi_Mstar_double4, color = 'darkgreen')
    ax[0,1].plot(M, np.power(10,M)*schechter.single_schechter2(M, keres_params), color = 'k', linestyle = '-', linewidth = .5, label = "$\mathrm{K+03}$")
    ax[0,1].plot(M, np.power(10,M)*schechter.single_schechter2(M, obreschkow_params), color = 'k', linestyle = '--', linewidth = .5, label = "$\mathrm{O+09}$")
    ax[0,1].errorbar(   prods_d[2], np.power(10,prods_d[2])*prods_d[1], yerr = np.power(10,prods_d[2])*error_arr_d, fmt="h",
                        ecolor = 'royalblue', mec = 'royalblue', mfc ='lightskyblue',
                        zorder = 99, capsize=2,
                        label = "$\mathrm{xCOLD \, GASS \, detections \, only}$")
    ax[0,1].errorbar(   prods[2], np.power(10,prods[2])*prods[1], yerr = np.power(10,prods[2])*error_arr,
                        fmt="^",  ecolor = 'crimson', mec = 'crimson',
                        mfc ='lightcoral', zorder = 99, capsize=2,
                        label = "$\mathrm{xCOLD \, GASS \, detections \, and \, non-detections}$")
    ax[0,1].errorbar(   prods_d2[2], np.power(10,prods_d2[2])*prods_d2[1], yerr = np.power(10,prods_d2[2])*error_arr_d2, fmt=".",
                        ms = 10, ecolor = 'darkgreen', mec = 'darkgreen',
                        mfc ='mediumseagreen', zorder = 99, capsize=2,
                        label = "$\mathrm{xCOLD \, GASS \, detections \, and \, estimates}$")
    ax[0,1].axvline(8.2, color = 'k', linestyle = '-.', linewidth = .5, label = "$\mathrm{xCOLD \, GASS \, \log M_{*} \, completeness \, limit}$", zorder = 0)
    ax[0,1].axvline(9.2, color = 'k', linestyle = ':', linewidth = .5, label = "$\mathrm{xCOLD \, GASS \, integration \, completeness \, limit}$", zorder = 0)
    ax[0,1].set_xlim(7.5, 10.5)
    ax[0,1].set_ylim(10000, np.power(10,7.5))
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlabel("$\mathrm{\log_{10}(M_{H2}) \, [M_\odot]}$")
    ax[0,1].set_ylabel("$\mathrm{\log_{10} \\rho (M_{H2}) \, [M_{\odot} Mpc^{-3} dex^{-1}]}$")
    ax[0,1].legend(fontsize = 8)
    plt.tight_layout()
    plt.savefig('img/MH2_paper_final.pdf')
    violin_plot(omegas_all)

def violin_plot(omegas_all):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(5, 5))
    parts = ax[0,0].violinplot(omegas_all, showmeans=False, showmedians=False, showextrema=False)
    facecolors = ['lightskyblue', 'mediumseagreen', 'lightcoral']
    i = 0
    for pc in parts['bodies']:
        pc.set_facecolor(facecolors[i])
        pc.set_alpha(1)
        i+=1
    quartile1, medians, quartile3 = np.percentile(omegas_all, [16, 50, 84], axis=0)
    print (medians)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(omegas_all, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    medians = np.append(medians, [6.9, 10.2])
    inds = np.arange(1, len(medians) + 1)
    quartile1 = np.append(quartile1, [6.9-2.7, 10.2-3.9])
    quartile3 = np.append(quartile3, [6.9+2.7, 10.2+3.9])
    print (inds)
    print (medians)
    print (quartile1)
    print (quartile3)
    ax[0,0].scatter(inds, medians, marker='o', edgecolors = 'k', color='white', s=40, zorder=3)
    ax[0,0].vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    labels = ["$\mathrm{DET}$", "$\mathrm{EST}$", "$\mathrm{ALL}$", "$\mathrm{O+09}$", "$\mathrm{K+03}$"]
    set_axis_style(ax[0,0], labels)

    ax[0,0].set_ylabel("$\mathrm{\Omega_{H2} \, [10^{-5}h^{-1}]}$")
    ax[0,0].set_xlabel("")
    ax[0,0].set_ylim(3, 15)

    plt.tight_layout()
    plt.savefig('img/violinplot.pdf')

def LCO_vs_SFR(xCOLDGASS_data):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(5, 5))
    ax[0,0].scatter(xCOLDGASS_data['LOGSFR_BEST'], xCOLDGASS_data['LCO_COR'])
    plt.savefig('img/LCO_vs_SFR.pdf')

# nose.run(argv=[__file__, 'nose_tests.py'])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# read in the COLD GASS data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
xCOLDGASS_data = read_files.read_COLD_GASS()
xCOLDGASS_data['LCO_COR'] = np.log10(xCOLDGASS_data['LCO_COR'])
LCO_vs_SFR(xCOLDGASS_data)
detections = xCOLDGASS_data[xCOLDGASS_data['LCO_COR_ERR']>0]
print ('length of arrays before binning', len(detections), len(xCOLDGASS_data))
print (min(xCOLDGASS_data['LCO_COR']), max(xCOLDGASS_data['LCO_COR']))
bins_LCO = np.linspace(min(xCOLDGASS_data['LCO_COR']), max(xCOLDGASS_data['LCO_COR']), 13)
prods_LCO = Schechter3(xCOLDGASS_data, bins_LCO)
prods_LCOd = Schechter3(detections, bins_LCO)
error_arr_LCO = bootstrap_with_replacement2(xCOLDGASS_data, int(len(xCOLDGASS_data)*.8), bins_LCO)
error_arr_LCOd = bootstrap_with_replacement2(detections, int(len(detections)*.8), bins_LCO)
xxGASS, det, nondet = read_files.read_GASS(True)
detections = xCOLDGASS_data[xCOLDGASS_data['LIM_LOGMH2'] == 0]
xCOLDGASS_data = xCOLDGASS_data[np.isfinite(xCOLDGASS_data['LOGSFR_BEST'])]
xCOLDGASS_data = xCOLDGASS_data[np.isfinite(xCOLDGASS_data['MH2'])]
xCOLDGASS_nondet = xCOLDGASS_data[xCOLDGASS_data['LOGMH2_ERR']==0]
xCOLDGASS_data = xCOLDGASS_data[xCOLDGASS_data['LOGMH2_ERR']>0]
xCOLDGASS_nondet['LOGMH2_ERR'] = 1/(5*np.log(10))
CGx1, CGy1, CGxerr, CGyerr = xCOLDGASS_data['LOGSFR_BEST'].values, xCOLDGASS_data['MH2'].values, xCOLDGASS_data['LOGSFR_ERR'].values, xCOLDGASS_data['LOGMH2_ERR'].values
CGx2, CGy2, CGxerr2, CGyerr2 = xCOLDGASS_nondet['LOGSFR_BEST'].values, xCOLDGASS_nondet['MH2'].values, xCOLDGASS_nondet['LOGSFR_ERR'].values, xCOLDGASS_nondet['LOGMH2_ERR'].values
CGS1 = S_error(CGxerr, CGyerr)
CGS2 = S_error(CGxerr2, [1/(5*np.log(10))])
global COLD_GASS_data
COLD_GASS_data = CGx1, CGx2, CGy1, CGy2, CGS1, CGS2

xxGASS, det, nondet = read_files.read_GASS(True)
xxGASS_final = fits.open('data/xGASS_RS_final_Serr_180903.fits')
xxGASS_final = Table(xxGASS_final[1].data).to_pandas()
xxGASS['SNR'] = xxGASS_final['SNR']
xxGASS['MHI_err'] = np.power(10, xxGASS['lgMHI'])/xxGASS['SNR']
xxGASS['lgMHI_err'] = xxGASS['MHI_err']/(np.power(10,xxGASS['lgMHI'])*np.log(10))
# xxGASS['lgMstar_err'] = 0.0
xxGASS = xxGASS[xxGASS['SFR_best'] > 0.00001]
xxGASS = xxGASS[xxGASS['SFR_best'] < np.power(10,4)]
xxGASS = xxGASS[xxGASS['lgMHI'] > 7]
xxGASS = xxGASS[xxGASS['lgMHI'] < 13]
det = xxGASS[xxGASS['HIconf_flag']==0]
nondet = xxGASS[xxGASS['HIconf_flag']==-99]
# det = xxGASS[xxGASS['SNR']>0]
global GASS_data
S1 = S_error(np.log10(det['SFR_best'].values), det['lgMHI_err'].values)
S2 = S_error(np.log10(nondet['SFR_best'].values), nondet['lgMHI_err'].values)
GASS_data = np.log10(det['SFR_best'].values), np.log10(nondet['SFR_best'].values), det['lgMHI'].values, nondet['lgMHI'].values, S1, S2
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # parameter estimation for the SFR-MHI scaling relation
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# nwalkers, ndim = 200, 3
# g = [.95, 9.8, -1.3]
# pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
# sampler0 = emcee.EnsembleSampler(nwalkers, ndim, log_prob_SFR_MHI)
# sampler0.run_mcmc(pos, 500, progress=True)
# samples0 = sampler0.chain[:, 250:, :].reshape((-1, ndim))
# plot_samples_3(sampler0, ndim, '_sfr_mhI_fit')
# np.savetxt('data/SFR_MHI_chain.txt', samples0)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# parameter estimation for the SFR-MH2 scaling relation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# nwalkers, ndim = 200, 3
# g = [.85, 8.92, -1.3]
# pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
# sampler1 = emcee.EnsembleSampler(nwalkers, ndim, log_prob_SFR_MH2)
# sampler1.run_mcmc(pos, 500, progress=True)
# samples1 = sampler1.chain[:, 250:, :].reshape((-1, ndim))
# print ('sfr_mh2_params', np.median(samples1[:, 0]), np.median(samples1[:, 1]), np.median(samples1[:, 2]))
# plot_samples_3(sampler1, ndim, '_sfr_mh2_fit')
# np.savetxt('data/SFR_MH2_chain.txt', samples1)
# read from file
samples1 = np.loadtxt('data/SFR_MH2_chain.txt')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# do the binning and bootstrapping for the Schechter parameter estimation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bins = np.linspace(7.69,10.15,15)
# bins = np.linspace(7.69, 10.15, 11)
M = np.linspace(7.5, 10.5, 300)
M2 = np.linspace(4.0, 11.5, 500)
xCOLDGASS_data = read_files.read_COLD_GASS()
prods = Schechter(xCOLDGASS_data, bins)
prods_d = Schechter(detections, bins)
dm = prods[2][1] - prods[2][0]
start_i = np.where(np.array(prods[2]) > 8.2)[0][0]
error_arr = bootstrap_with_replacement(xCOLDGASS_data, int(len(xCOLDGASS_data)*.8), bins)
error_arr_d = bootstrap_with_replacement(detections, int(len(detections)*.8), bins)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate the H2 masses for the non-detections
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
xCOLDGASS_data = utilities.estimate_H2_masses(xCOLDGASS_data, samples1, 200)
# now combine the detections and predicted masses for the non-detections
xCOLDGASS_data['MH2'] = xCOLDGASS_data['LOGMH2'] + xCOLDGASS_data['MH2_estimated']
xCOLDGASS_data['MH2_est_err'] = xCOLDGASS_data['LOGMH2_ERR'] + xCOLDGASS_data['MH2_estimated_err']
# plots.MH2_varying_with_Mstar(xCOLDGASS_data)
prods_d2 = Schechter(xCOLDGASS_data, bins)
error_arr_d2 = bootstrap_with_replacement(xCOLDGASS_data, int(len(xCOLDGASS_data)*.8), bins)
prods_LCO_e = Schechter4(xCOLDGASS_data, bins_LCO)
error_arr_LCO_e = bootstrap_with_replacement3(xCOLDGASS_data, int(len(xCOLDGASS_data)*.8), bins_LCO)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# do the emcee parameter estimation for the two schechter fits
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nwalkers, ndim = 200, 3
g = [0.0001, 9.5, -1.1]
pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
param_names = ['rhostar', 'logM', 'alpha']
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# all the data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sampler5 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (prods[2][start_i:], prods[1][start_i:], error_arr[start_i:]))
sampler5.run_mcmc(pos, 1000)
plot_samples_full(sampler5, ndim, 'MH2_observed_all', param_names)
samples5 = sampler5.chain[:, 500:, :].reshape((-1, ndim))
np.savetxt('data/H2_MF_params_all.txt', samples5)
# samples5 = np.loadtxt('data/H2_MF_params_all.txt')
# plot_corner_full(samples5, 'MH2_observed_all')
phistar2 = (np.median(samples5[:, 0]))
Mstar2 = (np.median(samples5[:, 1]))
alpha2 = (np.median(samples5[:, 2]))
print (round(phistar2, 2), round(Mstar2, 2), round(alpha2, 2))
# np.savetxt('data/samples5.txt', samples5)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# detections only
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sampler6 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (prods_d[2][start_i:], prods_d[1][start_i:], error_arr_d[start_i:]))
sampler6.run_mcmc(pos, 1000)
plot_samples_full(sampler6, ndim, 'MH2_observed_detections', param_names)
samples6 = sampler6.chain[:, 500:, :].reshape((-1, ndim))
np.savetxt('data/H2_MF_params_detections.txt', samples6)
# samples6 = np.loadtxt('data/H2_MF_params_detections.txt')
# plot_corner_full(samples6, 'MH2_observed_detections')
phistar3 = (np.median(samples6[:, 0]))
Mstar3 = (np.median(samples6[:, 1]))
alpha3 = (np.median(samples6[:, 2]))
print (round(phistar3, 2), round(Mstar3, 2), round(alpha3, 2))
# np.savetxt('data/samples5.txt', samples5)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimated masses for non-detections
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sampler7 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (prods_d2[2][start_i:], prods_d2[1][start_i:], error_arr_d2[start_i:]))
sampler7.run_mcmc(pos, 1000)
plot_samples_full(sampler7, ndim, 'MH2_observed_estimated', param_names)
samples7 = sampler7.chain[:, 500:, :].reshape((-1, ndim))
np.savetxt('data/H2_MF_params_scaling_estimated.txt', samples7)
# samples7 = np.loadtxt('data/H2_MF_params_scaling_estimated.txt')
# plot_corner_full(samples7, 'MH2_observed_estimated')
phistar4 = (np.median(samples7[:, 0]))
Mstar4 = (np.median(samples7[:, 1]))
alpha4 = (np.median(samples7[:, 2]))
print (round(phistar4, 2), round(Mstar4, 2), round(alpha4, 2))
# np.savetxt('data/samples5.txt', samples5)

# p_m = np.array([7.813, 8.059, 8.305, 8.551, 8.797, 9.043, 9.289, 9.535, 9.781, 10.027])
# p_phi = np.array([0.00286376, 0.00668211, 0.01011862, 0.01107321, 0.00840036, 0.00429564, 0.00334105, 0.00200463, 0.00124096, 0.00028638])
# p_error = np.array([0.00026085, 0.00032034, 0.00050953, 0.00054069, 0.00057808, 0.00055566, 0.00050332, 0.00044624, 0.00033308, 0.00022391])
# sampler6 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (p_m[1:], p_phi[1:], p_error[1:]))
# sampler6.run_mcmc(pos, 1000, progress=True)
# samples6 = sampler6.chain[:, 800:, :].reshape((-1, ndim))
# phistar3 = (np.median(samples6[:, 0]))
# Mstar3 = (np.median(samples6[:, 1]))
# alpha3 = (np.median(samples6[:, 2]))
MH2_plot(prods, prods_d, prods_d2, samples5, samples6, samples7, sampler5, sampler6, sampler7)
LCO_plot(prods_LCO, prods_LCOd, prods_LCO_e)
# print (prods)
