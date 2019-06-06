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
from integrand import integrand_MHI, integrand_MHI_var_sigma, integrand_MHI_double
import os
import time
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
pd.options.mode.chained_assignment = None  # default='warn'
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

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
    OmegaH2 = (rhoH2/rhocrit)*0.7
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
    minm, maxm = 7.69, 10.15
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

def throw_error(df, bins, frac):
    throws = len(df)
    minm, maxm = 7.69, 10.15
    dm = bins[1] - bins[0]
    num = int(throws*frac)
    print ('NUM', num)
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
    print (np.sum(countTOT[0,:]))
    print (countTOT)
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

def lumdistance(data):
    omega_m = 0.3                          # from Planck
    omega_l = 0.7                       # from Planck
    c = 3*math.pow(10,5)                    # in km/s
    Ho = 70                                 # in km/(s Mpc)
    f = lambda x : (((omega_m*((1+z)**3))+omega_l)**-0.5)
    Dlvals = np.zeros((len(data),1))
    for idx,row in data.iterrows():
        z = row['Z_SDSS']
        integral = integrate.quad(f, 0.0, z)    # numerically integrate to calculate luminosity distance
        Dm = (c/Ho)*integral[0]
        Dl = (1+z)*Dm                           # calculate luminosity distance
        #DH = (c*z)/Ho                          # calculate distance from Hubble law for comparison
        Dlvals[idx,0] = Dl
    return Dlvals

def calcVm(data, numtot, bootstrap):
    answer, M, baldry = doubleschec(min(data['LOGMSTAR']), max(data['LOGMSTAR']))
    # print (len(answer), len(M))
    # print (np.array(answer))
    V_arb = 1E2
    N_arb = np.sum(np.multiply(answer, V_arb * (M[1] - M[0])))
    print (N_arb, numtot)
    V_CG = V_arb*(numtot/N_arb)
    print (V_CG)
    V_CG2 = V_arb*(int(numtot/(1/bootstrap))/N_arb)
    return V_CG, V_CG2

def doubleschec(m1, m2):
    xbaldry = [7.10, 7.30, 7.5, 7.7, 7.9, 8.1, 8.3, 8.5, 8.7, 8.9, 9.1, 9.3, 9.5, 9.7, 9.9, 10.1, 10.3, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5, 11.7, 11.9]
    baldry = [17.9, 43.1, 31.6, 34.8, 27.3, 28.3, 23.5, 19.2, 18.0, 14.3, 10.2, 9.59, 7.42, 6.21, 5.71, 5.51, 5.48, 5.12, 3.55, 2.41, 1.27, 0.338, 0.042, 0.021, 0.042]
    baldry = np.divide(baldry, 1000.0)
    M1 = np.linspace(9,13.5,25)
    M = np.linspace(m1,m2,100)
    M2 = np.linspace(6,12,100)
    # print (M)
    # print (M)
    dM = M[1]-M[0]
    Mstar = 10.66
    phi1 = 3.96E-3
    phi2 = 0.79E-3
    alph1 = -0.35
    alph2 = -1.47

    b_Mstar = 10.72
    b_phi1 = 0.71E-3
    b_alpha1 = -1.45

    r_Mstar = 10.72
    r_phi1 = 3.25E-3
    r_phi2 = 0.08E-3
    r_alpha1 = -0.45
    r_alpha2 = -1.45

    phis = []
    phib = []
    phir = []

    bphi = []
    rphi = []
    for i in range(0,len(M)):
        frac = 10**(M[i]-Mstar)
        exp = math.exp(-frac)
        phibit1 = (phi1*(frac**(alph1+1)))
        phibit2 = (phi2*(frac**(alph2+1)))

        b_frac2 = 10**(M[i]-b_Mstar)
        b_exp = math.exp(-b_frac2)
        b_phibit1 = (b_phi1*(b_frac2**(b_alpha1+1)))

        r_frac2 = 10**(M[i]-r_Mstar)
        r_exp = math.exp(-r_frac2)
        r_phibit1 = (r_phi1*(r_frac2**(r_alpha1+1)))
        r_phibit2 = (r_phi2*(r_frac2**(r_alpha2+1)))

        log = np.log(10)
        phis.append(exp*log*(phibit1+phibit2))
        phib.append(exp*log*(phibit1))
        phir.append(exp*log*(phibit2))

        rphi.append(r_exp*log*(r_phibit1+r_phibit2))
        bphi.append(b_exp*log*(b_phibit1))

    y=log_schechter(M, b_phi1, b_Mstar, b_alpha1)
    y2 = log_schechter_true(M, b_phi1, b_Mstar, b_alpha1)
    y3 = log_double_schechter_true(M, phi1, phi2, Mstar, alph1, alph2)

    Mstar = 10.66
    phistar1 = 3.96E-3
    phistar2 = 0.79E-3
    alpha1 = - 0.35
    alpha2 = - 1.47
    y4 = np.log(10) * np.exp(-np.power(10,M2-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M2-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M2-Mstar)))

    xmajorLocator   = MultipleLocator(1)
    xminorLocator   = MultipleLocator(0.2)
    ymajorLocator   = MultipleLocator(1)
    yminorLocator   = MultipleLocator(0.2)
    # fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    # ax[0,0].tick_params(axis='both', which='major', labelsize=15)
    # ax[0,0].scatter(xbaldry, np.log10(baldry), color = 'k', label = 'baldry')
    # ax[0,0].plot(M, np.log10(phis), color = 'g', label ='fit', linewidth = 10)
    # ax[0,0].plot(M, np.log10(phib), color = 'r', label ='fit')
    # ax[0,0].plot(M, np.log10(phir), color = 'b', label ='fit')
    # ax[0,0].plot(M, np.log10(rphi), color = 'r', label ='fit', linestyle = '--')
    # ax[0,0].plot(M, np.log10(bphi), color = 'b', label ='fit', linestyle = '--')
    # ax[0,0].plot(M, np.log10(y), color = 'k', label ='fit', linestyle = ':')
    # ax[0,0].plot(M, np.log10(y2), color = 'g', label ='fit', linestyle = ':')
    # ax[0,0].plot(M, np.log10(y3), color = 'm', label ='fit', linestyle = ':')
    # ax[0,0].plot(M2, np.log10(y4), color = 'k', label ='fit', linewidth = '4')
    # ax[0,0].set_ylim(-6,0)
    # plt.xlabel(r'$\mathrm{log M_{*}}$', fontsize = 20)
    # plt.ylabel(r'$\mathrm{log \phi}$', fontsize = 20)
    # plt.savefig('img/baldry.pdf')
    # plt.legend()
    return phis, M, baldry

def read_COLD_GASS():
    bootstrap = 0.8
    # Reading in the COLD GASS file and converting to pandas df
    xCOLDGASS = fits.open('data/xCOLDGASS_PubCat.fits')
    xCOLDGASS_data = Table(xCOLDGASS[1].data).to_pandas()
    # Calculate lumdist, Vm, MH2 including limits
    V_CG, V_CG2 = calcVm(xCOLDGASS_data, len(xCOLDGASS_data), bootstrap)
    xCOLDGASS_data['D_L'] = lumdistance(xCOLDGASS_data)
    xCOLDGASS_data['V_m'] = V_CG
    xCOLDGASS_data['V_m2'] = V_CG2
    xCOLDGASS_data['MH2'] = xCOLDGASS_data['LOGMH2'] + xCOLDGASS_data['LIM_LOGMH2']
    xCOLDGASS_data['new_LOGMH2'] = 0
    return xCOLDGASS_data

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
    if 0.000001 < rhostar < 0.005 and 9.0 < logM < 10.5 and -2.0 < alpha < 1.0:
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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# read in the COLD GASS data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
xCOLDGASS_data = read_COLD_GASS()
detections = xCOLDGASS_data[xCOLDGASS_data['LIM_LOGMH2'] == 0]
xCOLDGASS_data = xCOLDGASS_data[np.isfinite(xCOLDGASS_data['LOGSFR_BEST'])]
xCOLDGASS_data = xCOLDGASS_data[np.isfinite(xCOLDGASS_data['MH2'])]
xCOLDGASS_nondet = xCOLDGASS_data[xCOLDGASS_data['LOGMH2_ERR']==0]
xCOLDGASS_data = xCOLDGASS_data[xCOLDGASS_data['LOGMH2_ERR']>0]
xCOLDGASS_nondet['LOGMH2_ERR'] = 0.14
CGx1, CGy1, CGxerr, CGyerr = xCOLDGASS_data['LOGSFR_BEST'].values, xCOLDGASS_data['MH2'].values, xCOLDGASS_data['LOGSFR_ERR'].values, xCOLDGASS_data['LOGMH2_ERR'].values
CGx2, CGy2, CGxerr2, CGyerr2 = xCOLDGASS_nondet['LOGSFR_BEST'].values, xCOLDGASS_nondet['MH2'].values, xCOLDGASS_nondet['LOGSFR_ERR'].values, xCOLDGASS_nondet['LOGMH2_ERR'].values
CGS1 = S_error(CGxerr, CGyerr)
CGS2 = S_error(CGxerr2, [0.14])
global COLD_GASS_data
COLD_GASS_data = CGx1, CGx2, CGy1, CGy2, CGS1, CGS2
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# parameter estimation for the SFR-MH2 scaling relation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nwalkers, ndim = 200, 3
g = [.85, 8.92, -1.3]
pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler1 = emcee.EnsembleSampler(nwalkers, ndim, log_prob_SFR_MH2)
sampler1.run_mcmc(pos, 500, progress=True)
samples1 = sampler1.chain[:, 250:, :].reshape((-1, ndim))
plot_samples_3(sampler1, ndim, '_sfr_mh2_fit')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate the H2 masses for the non-detections
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N = 200
xCOLDGASS_nondet['MH2_estimated'] = 0
xCOLDGASS_nondet['MH2_estimated_err'] = 0
for idx, row in xCOLDGASS_nondet.iterrows():
    masses = []
    for c1, c2, lnc in samples1[np.random.randint(len(samples1), size=N)]:
        # print (row['LOGMSTAR'], c1, c2, c1*row['LOGMSTAR'] + c2 + np.random.normal(0, np.exp(lnc)))
        masses.append(c1*row['LOGSFR_BEST'] + c2 + np.random.normal(0, np.exp(lnc)))
    xCOLDGASS_nondet.loc[idx, 'MH2_estimated'] = np.mean(masses)
    xCOLDGASS_nondet.loc[idx, 'MH2_estimated_err'] = np.std(masses)
print (xCOLDGASS_nondet)
x11 = np.linspace(7,9,100)
plt.scatter(xCOLDGASS_nondet['MH2'], xCOLDGASS_nondet['MH2_estimated'])
plt.plot(x11,x11)
plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# do the binning and bootstrapping for the Schechter parameter estimation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bins = np.linspace(7.69,10.15,15)
# bins = np.linspace(7.69, 10.15, 11)
M = np.linspace(7.5, 10.5, 300)
M2 = np.linspace(4.0, 11.5, 500)
prods = Schechter(xCOLDGASS_data, bins)
prods_d = Schechter(detections, bins)
dm = prods[2][1] - prods[2][0]
start_i = np.where(np.array(prods[2]) > 8.2)[0][0]
error_arr = bootstrap_with_replacement(xCOLDGASS_data, int(len(xCOLDGASS_data)*.8), bins)
error_arr_d = bootstrap_with_replacement(detections, int(len(detections)*.8), bins)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# do the emcee parameter estimation for the two schechter fits
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nwalkers, ndim = 200, 3
g = [0.0001, 9.5, -1.1]
pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# all the data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sampler5 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (prods[2][start_i:], prods[1][start_i:], error_arr[start_i:]))
sampler5.run_mcmc(pos, 1000, progress=True)
# plot_samples_full(sampler5, ndim, 'mainsequence_full')
samples5 = sampler5.chain[:, 500:, :].reshape((-1, ndim))
phistar2 = (np.median(samples5[:, 0]))
Mstar2 = (np.median(samples5[:, 1]))
alpha2 = (np.median(samples5[:, 2]))
# np.savetxt('data/samples5.txt', samples5)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# detections only
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sampler6 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (prods_d[2][start_i:], prods_d[1][start_i:], error_arr_d[start_i:]))
sampler6.run_mcmc(pos, 1000, progress=True)
# plot_samples_full(sampler5, ndim, 'mainsequence_full')
samples6 = sampler6.chain[:, 500:, :].reshape((-1, ndim))
phistar3 = (np.median(samples6[:, 0]))
Mstar3 = (np.median(samples6[:, 1]))
alpha3 = (np.median(samples6[:, 2]))
# np.savetxt('data/samples5.txt', samples5)

p_m = np.array([7.813, 8.059, 8.305, 8.551, 8.797, 9.043, 9.289, 9.535, 9.781, 10.027])
p_phi = np.array([0.00286376, 0.00668211, 0.01011862, 0.01107321, 0.00840036, 0.00429564, 0.00334105, 0.00200463, 0.00124096, 0.00028638])
p_error = np.array([0.00026085, 0.00032034, 0.00050953, 0.00054069, 0.00057808, 0.00055566, 0.00050332, 0.00044624, 0.00033308, 0.00022391])
# sampler6 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (p_m[1:], p_phi[1:], p_error[1:]))
# sampler6.run_mcmc(pos, 1000, progress=True)
# samples6 = sampler6.chain[:, 800:, :].reshape((-1, ndim))
# phistar3 = (np.median(samples6[:, 0]))
# Mstar3 = (np.median(samples6[:, 1]))
# alpha3 = (np.median(samples6[:, 2]))


# print (prods)
fig, ax = plt.subplots(nrows = 1, ncols = 3, squeeze=False, figsize=(15,5))
ax[0,0].errorbar(prods[2], prods[1], yerr = error_arr, fmt=".k", capsize=0, zorder = 99)
ax[0,0].errorbar(prods_d[2], prods_d[1], yerr = error_arr_d, fmt=".b", capsize=0, zorder = 99)
# ax[0,0].errorbar(p_m, p_phi, yerr = p_error, fmt=".b", capsize=0, zorder = 20)
omegas, omegas2 = [], []
for m1, b, f1 in samples5[np.random.randint(len(samples5), size=500)]:
    ax[0,0].plot(M, schechfunc(np.power(10,M),m1, 10**b,f1), color="g", alpha=0.1)
    ax[0,1].plot(M2, np.log10(np.power(10,M2)*schechfunc(np.power(10,M2),m1, 10**b,f1)), color="g", alpha=0.1)
    # print (OmegaH2(M2, np.log10(np.power(10,M2)*schechfunc(np.power(10,M2),m1, 10**b,f1))))
    omegas.append(OmegaH2(M2, np.log10(np.power(10,M2)*schechfunc(np.power(10,M2),m1, 10**b,f1)))*1E5)
for m1, b, f1 in samples6[np.random.randint(len(samples6), size=500)]:
    ax[0,0].plot(M, schechfunc(np.power(10,M),m1, 10**b,f1), color="c", alpha=0.1)
    ax[0,1].plot(M2, np.log10(np.power(10,M2)*schechfunc(np.power(10,M2),m1, 10**b,f1)), color="c", alpha=0.1)
    # print (OmegaH2(M2, np.log10(np.power(10,M2)*schechfunc(np.power(10,M2),m1, 10**b,f1))))
    omegas2.append(OmegaH2(M2, np.log10(np.power(10,M2)*schechfunc(np.power(10,M2),m1, 10**b,f1)))*1E5)
Mstar = 9.35
phistar1 = 4.29E-3
alpha1 = -1.03
phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))
phi_Mstar_double2 = np.log(10) * np.exp(-np.power(10,M-Mstar2)) * (phistar2*np.power(10,(alpha2+1)*(M-Mstar2)))
# phi_Mstar_double3 = np.log(10) * np.exp(-np.power(10,M-Mstar3)) * (phistar3*np.power(10,(alpha3+1)*(M-Mstar3)))
# ax[0,0].plot(M, phi_Mstar_double, color = 'c')
# plt.plot(M, schechfunc(np.power(10,M), phistar2, 10**Mstar2, alpha2), color = 'c', linestyle = '--', linewidth = 5)
ax[0,0].plot(M, phi_Mstar_double2, color = 'k')
ax[0,2].hist(omegas, bins = 30, alpha = 0.5, color = 'g')
ax[0,2].hist(omegas2, bins = 30, alpha = 0.5, color = 'c')
ax[0,2].text(4,10,str(round(np.median(omegas2),2)) + r'$\pm$' + str(round(np.std(omegas2),2)))
ax[0,2].text(6,10,str(round(np.median(omegas),2)) + r'$\pm$' + str(round(np.std(omegas),2)))
# plt.plot(M, phi_Mstar_double3, color = 'b')
ax[0,0].axvline(8.2)
ax[0,0].set_ylim(0.00005, 0.05)
ax[0,1].set_ylim(0, 7)
ax[0,0].set_xlim(7.5, 10.5)
ax[0,0].set_yscale('log')
ax[0,0].set_xlabel("$\log(M_{H2}) \quad [M_\odot]$")
ax[0,0].set_ylabel("$\log(\phi) \quad [Mpc^{-3} dex^{-1}]$")
ax[0,1].set_xlabel("$\log(M_{H2}) \quad [M_\odot]$")
ax[0,1].set_ylabel("$\log(p) \quad [M_{\odot} Mpc^{-3} dex^{-1}]$")
ax[0,2].set_xlabel("$\Omega_{H2} \, [10^{-5}h^{-1}]$")
plt.savefig('img/omega_h2_estimates.pdf')
# print (xCOLDGASS_data)
