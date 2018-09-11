import pyximport
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
from scipy.integrate import quad, dblquad
from scipy import special
import random
from integrand import integrand_MHI
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

def log_schechter1(logL, log_rho, log_Lstar, alpha):
    schechter = log_rho*((logL-log_Lstar)**(alpha+1))*math.exp(-(L-Lstar))*np.log(10)
    return schechter

def log_schechter_true(logL, log_phi, log_L0, alpha):
    # print (log_phi, log_L0, alpha)
    log = np.log(10)
    frac = np.power(10,(alpha+1)*(logL-log_L0))
    exp = np.exp(-np.power(10,logL-log_L0))
    return log*log_phi*frac*exp

def log_double_schechter_true(logL, log_phi1, log_phi2, log_L0, alpha1, alpha2):
    log = np.log(10)
    frac1 = np.power(10,(alpha1+1)*(logL-log_L0))
    frac2 = np.power(10,(alpha2+1)*(logL-log_L0))
    exp = np.exp(-np.power(10,logL-log_L0))
    return log*exp*(log_phi1*frac1 + log_phi2*frac2)

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

def baldryMF(x):
    Mstar = np.power(10,10.66)
    phistar1 = 3.96E-3
    phistar2 = 0.79E-3
    alpha1 = - 0.35
    alpha2 = - 1.47

def integrand(x):
    return x*x*x

def p_SFR_Mstar(SFR):
    M = np.linspace(7,12,300)
    # y2=(soln.x[0]*x*x)+(soln.x[1]*x) +(soln.x[2])
    # -0.07983081   2.20158877 -14.06085728   0.23
    sigma = 0.23
    a = -0.07983081
    b = 2.20158877
    c = -14.06085728
    f = (a*M*M) + (b*M) + c
    # print (f)
    y = (1/np.sqrt(2*np.pi*np.power(sigma,2)))*np.exp((-1/(2*np.power(sigma,2)))*np.power((SFR-f),2))
    return M,y

def plot_P_SFR_Mstar():
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    M,y = p_SFR_Mstar(-0.3)
    # print (y)
    ax[0,0].plot(M,y)
    plt.savefig('probs.pdf')

def p_MH2_SFR(MH2):
    SFR = np.linspace(-3,2,300)
    sigma = 0.2
    a = 0.69
    b = 9.01
    f = (a*SFR) + (b)
    # print (f)
    y = (1/np.sqrt(2*np.pi*np.power(sigma,2)))*np.exp((-1/(2*np.power(sigma,2)))*np.power((MH2-f),2))
    return SFR,y

def plot_MH2_SFR():
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    SFR,y = p_MH2_SFR(9.0)
    # print (y)
    ax[0,0].plot(SFR,y)
    plt.savefig('probs2.pdf')

def integrand_stellar_mass(M):
    # Baldry+11 double Schechter function parameters
    Mstar = 10.66
    phistar1 = 3.96E-3
    phistar2 = 0.79E-3
    alpha1 = - 0.35
    alpha2 = - 1.47
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    return phi_Mstar_double

def integrand_SFR(M,SFR):
    # SFR = -0.3
    # Baldry+11 double Schechter function parameters
    Mstar = 10.66
    phistar1 = 3.96E-3
    phistar2 = 0.79E-3
    alpha1 = - 0.35
    alpha2 = - 1.47
    # parameters from SFR-M* plane fit
    sigma = 0.23
    a = -0.07983081
    b = 2.20158877
    c = -14.06085728
    # phi_Mstar = np.log(10)*phistar1*np.power(10,(alpha1+1)*(M-Mstar))*np.exp(-np.power(10,M-Mstar))
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    # phi_Mstar = np.exp(-M/Mstar)*(1/Mstar)*((phistar1*np.power(M/Mstar, alpha1)) + (phistar2*np.power(M/Mstar, alpha2)))
    f = (a*M*M) + (b*M) + c
    P_SFR_given_Mstar = (1/np.sqrt(2*np.pi*np.power(sigma,2)))*np.exp((-1/(2*np.power(sigma,2)))*np.power((SFR-f),2))
    return phi_Mstar_double*P_SFR_given_Mstar

def integrand_MH2_blue(M, SFR, MH2, *params):
    if len(params) > 0:
        a, b, c, sigma = params[0][0]
        m1, m2, sigma2 = params[1][0]
        # sigma, sigma2 = 0.2, 0.1
    else:
        # parameters from SFR-M* plane fit
        sigma = 0.23
        a = -0.07983081
        b = 2.20158877
        c = -14.06085728
        # parameters from the MH2-SFR plane fit
        sigma2 = 0.2
        m1 = 0.69
        m2 = 9.01
    # Baldry+11 double Schechter function parameters
    Mstar = 10.72
    phistar1 = 0.71E-3
    alpha1 = - 1.45
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar))) #+ phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    f = (a*M*M) + (b*M) + c
    P_SFR_given_Mstar = (1/np.sqrt(2*np.pi*np.power(sigma,2)))*np.exp((-1/(2*np.power(sigma,2)))*np.power((SFR-f),2))
    f2 = (m1*SFR) + m2
    P_MH2_given_SFR = (1/np.sqrt(2*np.pi*np.power(sigma2,2)))*np.exp((-1/(2*np.power(sigma2,2)))*np.power((MH2-f2),2))
    return phi_Mstar_double*P_SFR_given_Mstar*P_MH2_given_SFR

def integrand_MHI_blue(M, SFR, MHI, *params):
    a1, a2, a3, lnf, b1, b2, lnf1 = params
    Mstar, phistar1, alpha1 = 10.72, 0.71E-3, -1.45
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))
    f = (a1*M*M) + (a2*M) + a3
    P_SFR_given_Mstar = (1/np.sqrt(2*np.pi*np.power(lnf,2)))*np.exp((-1/(2*np.power(lnf,2)))*np.power((SFR-f),2))
    f2 = (b1*SFR) + b2
    P_MHI_given_SFR = (1/np.sqrt(2*np.pi*np.power(lnf1,2)))*np.exp((-1/(2*np.power(lnf1,2)))*np.power((MHI-f2),2))
    return phi_Mstar_double*P_SFR_given_Mstar*P_MHI_given_SFR

def integrand_MH2_red(M,SFR,MH2):
    # Baldry+11 double Schechter function parameters
    Mstar = 10.72
    phistar1 = 3.25E-3
    phistar2 = 0.08E-3
    alpha1 = - 0.45
    alpha2 = - 1.45
    # parameters from SFR-M* plane fit
    sigma = 0.3
    a = -0.07983081
    b = 2.20158877
    c = -15.06085728
    # parameters from the MH2-SFR plane fit
    sigma2 = 0.2
    m1 = 0.69
    m2 = 9.01
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    f = (a*M*M) + (b*M) + c
    P_SFR_given_Mstar = (1/np.sqrt(2*np.pi*np.power(sigma,2)))*np.exp((-1/(2*np.power(sigma,2)))*np.power((SFR-f),2))
    f2 = (m1*SFR) + m2
    P_MH2_given_SFR = (1/np.sqrt(2*np.pi*np.power(sigma2,2)))*np.exp((-1/(2*np.power(sigma2,2)))*np.power((MH2-f2),2))
    return phi_Mstar_double*P_SFR_given_Mstar*P_MH2_given_SFR

def gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def plot_scatter(GAMA):
    GAMA = GAMA[(GAMA['logM*']>9.0) & (GAMA['logM*']<9.2)]

    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    n, bins, patches = ax[0,0].hist(GAMA['logSFR'], 50, facecolor='green', alpha=0.75)
    bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
    popt,pcov = curve_fit(gauss,bins_mean,n,p0=[70,-0.5,0.2])
    print (popt)
    x = np.linspace(-3.0,1.0,500)
    ax[0,0].plot(x,gauss(x,*popt))
    plt.savefig('SFR_scatter.pdf')

def plot_samples(sampler, ndim, fname):
    fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["m", "b", "c", "log(f)"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    plt.savefig('img/sampler' + fname + '.pdf')

def plot_samples7(sampler, ndim, fname):
    fig, axes = plt.subplots(10, figsize=(10, 20), sharex=True)
    samples = sampler.get_chain()
    labels = ["a1", "a2", "a3", "log(f)", "b1", "b2", "log(f2)", "c1", "c2", "log(f3)"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    plt.savefig('img/sampler' + fname + '.pdf')

def plot_SFR_M_plane(GAMA, GAMAr, soln, x1, y1, std, sampler):
    ndim = 4
    # samples = sampler.chain[:, 250:, :].reshape((-1, ndim))
    # sampler = sampler.flatchain[250:, :]
    print (GAMA.columns)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    # plt.scatter(GAMA['logM*'], GAMA['logSFR'])
    x=np.linspace(7,12,500)
    y=(-.01828*x*x*x)+(0.4156*x*x) -(2.332*x)
    y2=(soln.x[0]*x*x)+(soln.x[1]*x) +(soln.x[2])
    # y3=(soln2.x[0]*x)+(soln2.x[1])
    # y4=(soln3.x[0]*x*x)+(soln3.x[1]*x) +(soln3.x[2])
    # ax[0,0].errorbar(x1,y1, yerr=std, fmt='h', capsize = 4, markersize = 7, linewidth=2, markeredgewidth=2, capthick=3,  mfc='w', mec='k', ecolor = 'k')
    ax[0,0].plot(x,y, color = 'red', label = 'Saintonge+16')
    ax[0,0].plot(x,y2, color = 'g', label = 'GAMA 2nd order')
    # ax[0,0].plot(x,y2+1.0, color = 'g', label = 'GAMA 2nd order')
    # ax[0,0].plot(x,y2-1.0, color = 'g', label = 'GAMA 2nd order')
    # ax[0,0].plot(x,y3, color = 'b', label = 'GAMA 1st order')
    # ax[0,0].plot(x,y4, color = 'k', label = 'Binned')
    # ax[0,0].hist2d(GAMA['logM*'], GAMA['logSFR'], bins=100, cmap = 'Blues', vmin=1,vmax =8)
    ax[0,0].errorbar(GAMA['logM*'], GAMA['logSFR'], xerr = GAMA['logM*err'], yerr = GAMA['logSFRerr'], fmt='o', capsize = .1, markersize = .4, linewidth=.1, markeredgewidth=.1, capthick=.1, mfc='gray', mec='gray', ecolor = 'gray')
    # ax[0,0].errorbar(GAMAr['logM*'], GAMAr['logSFR'], xerr = GAMAr['logM*err'], yerr = GAMAr['logSFRerr'], fmt='o', capsize = .1, markersize = .4, linewidth=.1, markeredgewidth=.1, capthick=.1, mfc='r', mec='r', ecolor = 'r')

    x0=np.linspace(7,12,300)
    for a, b, c, _ in sampler[np.random.randint(len(sampler), size=100)]:
        ax[0,0].plot(x0, a*x0*x0 + b*x0 + c, lw=1, alpha=0.1, color="b")
    ax[0,0].set_xlim(7,12)
    # plt.xlim(8,11 .5)
    ax[0,0].set_ylim(-3.5,1.5)
    plt.legend()
    plt.savefig('img/sfrmplane.png', dpi = 800)
    # print (GAMA)

def plot_emcee_result(GAMA, samples, soln2):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    x=np.linspace(7,12,500)
    for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
        ax[0,0].plot(x, m*x+b, color="k", alpha=0.1)
    ax[0,0].plot(x, soln2.x[0]*x+soln2.x[1], color="r", lw=2, alpha=0.8)
    ax[0,0].hist2d(GAMA['logM*'], GAMA['logSFR'], bins=100, cmap = 'Blues', vmin = 1, vmax = 8)
    ax[0,0].set_xlim(7,12)
    ax[0,0].set_ylim(-3.5,1.5)
    plt.legend()
    plt.savefig('emcee_fits.pdf')

def lnprior2(theta):
    a, b, c, lnf = theta
    if -0.2 < a < 0 and 1.5 < b < 2.5 and -16 < c < -12 and 0.2 < lnf < 0.4:
        return 0.0
    return -np.inf

def lnprob2(theta, x, y, yerr):
    lp = lnprior2(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_SFR_M2(theta, x, y, yerr)

def lnprior(theta):
    a, b, lnf = theta
    if 0.4 < a < 0.8 and -8 < b < -5 and 0.05 < lnf < 0.4:
        return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_SFR_M(theta, x, y, yerr)

def log_prior(theta):
    m, b, llamb = theta
    if 0.4 < m < 0.8 and 8 < b < 10 and -5.0 < llamb < 5.0:
        return 0.0
    return -np.inf

def log_marg_prob(params, x, y, S):
    m, b, llamb = params
    v = np.array([-m, 1.0])
    Sigma2 = np.dot(np.dot(S, v), v) + np.exp(2*llamb)
    DeltaN = y - (m*x) - b
    ll = -0.5 * np.sum(DeltaN**2/Sigma2 + np.log(Sigma2))
    return ll + log_prior(params)

def log_prior_HI(theta):
    m, b, llamb = theta
    if 0.6 < m < 1.0 and 8.0 < b < 11.0 and -2 < llamb < 2:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, x2, y2, S, S2):
    m, const, lnf = theta
    v = np.array([-m, 1.0])
    # det = np.where(fl != 4)
    # nondet = np.where(fl == 4)
    sigma = np.dot(np.dot(S, v), v) + np.exp(2 * lnf)
    # print ()
    sigma2 = np.dot(np.dot(S2, v), v) + np.exp(2 * lnf)
    sigma2 = sigma2 ** 0.5
    deltaN = y - (m * x) - const
    model = (m * x2) + const
    ll1 = -0.5 * np.sum(np.square(deltaN) / sigma + np.log(sigma))
    I = np.zeros(len(x2))
    for i in range(0,len(x2)):
        I[i] = np.log(((2 * np.pi) ** 0.5) *
                      0.5 * (special.erf((y2[i]-model[i]) / ((2 ** 0.5) * sigma2[i])) + 1))
    ll2 = np.sum(I)
    return ll1  + ll2 + log_prior_HI(theta) # combining detection & non detection results

def log_prior_MS(params):
    a, b, c, llamb = params
    if -1.0 < a < 1.0 and 0.0 < b < 5.0 and -20.0 < c < -5.0 and  -5.0 < llamb < 5.0:
        return 0.0
    return -np.inf

def log_marg_prob_MS(params, x, y, xerr, yerr):
    a, b, c, llamb = params
    Sigma2 = np.square(xerr)*np.square((2*a*x) + b) + np.square(yerr) + np.exp(2*llamb)
    DeltaN = y - (a*x*x) - (b*x) - c
    ll = -0.5 * np.sum(DeltaN**2/Sigma2 + np.log(Sigma2))
    return ll + log_prior_MS(params)

def sample_log_marg_prob(params, x, S):
    m, b, llamb = params
    v = np.array([-m, 1.0])
    Sigma2 = np.dot(np.dot(S, v), v) + np.exp(2*llamb)
    mu = (m*x) + b
    return np.random.normal(mu, Sigma2, 100)

def SFRM_plane():
    # Reading in the GAMA file and converting to pandas df
    GAMA = fits.open('data/GAMA.fits')
    GAMA = Table(GAMA[1].data).to_pandas()
    GAMA['D_L'] = cosmo.luminosity_distance(GAMA['Z'])
    GAMA['D_L_cm'] = cosmo.luminosity_distance(GAMA['Z']).to(u.cm)
    GAMA['D_L_pc'] = cosmo.luminosity_distance(GAMA['Z']).to(u.pc)
    GAMA['L_NUV'] = 4*np.pi*GAMA['D_L_cm']*GAMA['D_L_cm']*GAMA['BEST_FLUX_NUV']*10E-6*10E-23
    GAMA['SFR_NUV'] = 10**(-28.165) * GAMA['L_NUV']
    # GAMA['f_v_W4'] = 3631*10E-23*(10**(-GAMA['MAG_W4']/2.5))
    # GAMA['f_v_W1'] = 3631*10E-23*(10**(-GAMA['MAG_W1']/2.5))
    # GAMA['L_v_W4'] = 4*np.pi*GAMA['D_L_cm']*GAMA['D_L_cm']*GAMA['f_v_W4']*282.50
    # GAMA['L_v_W1'] = 4*np.pi*GAMA['D_L_cm']*GAMA['D_L_cm']*GAMA['f_v_W1']*22.883
    # GAMA['SFR_IR'] = 7.5E-10*(GAMA['L_v_W4'] -0.044*GAMA['L_v_W1'])
    GAMA['ABS_MAG_W4'] = GAMA['MAG_W4'] - 5*(np.log10(GAMA['D_L_pc'])-1)
    GAMA['ABS_MAG_W1'] = GAMA['MAG_W1'] - 5*(np.log10(GAMA['D_L_pc'])-1)
    GAMA['L_W4'] = 10**(0.4*(3.24-GAMA['ABS_MAG_W4']))
    GAMA['L_W1'] = 10**(0.4*(3.24-GAMA['ABS_MAG_W1']))
    GAMA['SFR_IR'] = 7.5E-10*(GAMA['L_W4'] - (0.044*GAMA['L_W1']))
    GAMA['SFR_TOT'] = GAMA['SFR_IR'] + GAMA['SFR_NUV']
    GAMA['log_SFR_TOT'] = np.log10(GAMA['SFR_TOT'])
    # GAMA.dropna()
    GAMA = GAMA[np.isfinite(GAMA['log_SFR_TOT'])]
    GAMA = GAMA[np.isfinite(GAMA['logmstar'])]
    GAMA = GAMA[GAMA['logmstar']>8.0]
    GAMA = GAMA[GAMA['logmstar']<11.5]
    GAMA = GAMA[GAMA['log_SFR_TOT']<2.5]
    # plt.hist2d(GAMA['logmstar'], GAMA['log_SFR_TOT'], bins=50)
    plt.scatter(GAMA['logmstar'], GAMA['log_SFR_TOT'])
    plt.xlim(7,12)
    # plt.xlim(8,11 .5)
    plt.ylim(-3.5,1.5)
    plt.show()
    # print (GAMA)

def second_order(a,b,c,x):
    return (a*x*x) + (b*x) + c

def read_GAMA():
    GAMA = pd.read_csv('data/GAMA_sample.dat', comment = '#', header = None, sep=r"\s*")
    GAMA.columns = ['CATAID', 'z', 'logM*', 'logM*err', 'logSFR', 'logSFRerr', 'ColorFlag']
    GAMA = GAMA[np.isfinite(GAMA['logSFR'])]
    GAMA = GAMA[np.isfinite(GAMA['logM*'])]
    GAMA = GAMA[GAMA['logM*']>7.0]
    GAMA = GAMA[GAMA['logM*']<12]
    GAMA = GAMA[GAMA['logSFR']<1.5]
    GAMA = GAMA[GAMA['logSFR']>-3.5]
    GAMAb = GAMA[GAMA['ColorFlag']==1]
    GAMAr = GAMA[GAMA['ColorFlag']==2]
    return GAMA, GAMAb, GAMAr

def MainSequence():
    # read in the GAMA data for z<0.08
    GAMA, GAMAb, GAMAr = read_GAMA()
    # binning the SFR-M plane
    bins = np.linspace(8,11,21)
    x1, y1, std = [], [], []
    for i in range (1,len(bins)):
        inbin = GAMAb[(GAMAb['logM*']>=bins[i-1]) & (GAMAb['logM*']< bins[i])]
        x1.append((bins[i]+bins[i-1])/2)
        y1.append(np.median(inbin['logSFR']))
        std.append(np.std(inbin['logSFR']))
    x1, y1, std = np.array(x1), np.array(y1), np.array(std)
    # ML fit to binned data
    nll = lambda *args: -log_likelihood_SFR_M2(*args)
    initial = np.array([ -0.067,   1.905, -30, 0.1])
    soln3 = minimize(nll, initial, args=(x1,y1,std))
    # ML fit to unbinned data
    nll = lambda *args: -log_likelihood_SFR_M2(*args)
    initial = np.array([soln3.x[0], soln3.x[1], soln3.x[2], 0.05])
    bnds = ((-0.1, -0.05), (2.0, 2.4), (-15,-13), (0.23,0.4))
    soln = minimize(nll, initial, args=(GAMAb['logM*'], GAMAb['logSFR'], GAMAb['logSFRerr']), method='TNC', bounds=bnds)
    print ('soln', soln["x"])
    # emcee fit using ML fit as initial guess
    ndim, nwalkers = 4, 100
    guess = [-0.07254516, 2.02271974, -13., 0.23]
    pos = [guess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_marg_prob_MS, args=(GAMAb['logM*'], GAMAb['logSFR'], GAMAb['logM*err'], GAMAb['logSFRerr']))
    sampler.run_mcmc(pos, 500, progress=True)
    plot_samples(sampler, ndim, 'SFR_M*')
    samples = sampler.chain[:, 250:, :].reshape((-1, ndim))
    plot_corner2(samples, 'sfrmplane')
    plot_SFR_M_plane(GAMAb, GAMAr, soln, x1, y1, std, samples)
    return samples

def log_likelihood(theta, x, y, yerr):
    # print (x)
    rhostar, logM, alpha = theta
    model = np.log(10.0) * rhostar * (10 ** x / 10 ** logM) ** (alpha + 1) * np.exp(-1 * 10 ** x / 10 ** logM)
    # print(model)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))

def log_likelihood_SFR_M(theta, x, y, yerr):
    a, b, lnf = theta
    model = (a*x) + (b)
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def log_likelihood_SFR_MH2(theta, x, y, yerr):
    # print (x)
    a, b, lnf = theta
    model = (a*x) + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def log_likelihood_SFR_M2(theta, x, y, yerr):
    a, b, c, lnf = theta
    model = (a*x*x) + (b*x) + c
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def calcVm(data, numtot, bootstrap):
    answer, M, baldry = doubleschec(min(data['LOGMSTAR']), max(data['LOGMSTAR']))
    V_arb = 1E2
    N_arb = np.sum(np.multiply(answer,V_arb*(M[1]-M[0])))
    V_CG = V_arb*(numtot/N_arb)
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
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    ax[0,0].tick_params(axis='both', which='major', labelsize=15)
    ax[0,0].scatter(xbaldry, np.log10(baldry), color = 'k', label = 'baldry')
    ax[0,0].plot(M, np.log10(phis), color = 'g', label ='fit')
    ax[0,0].plot(M, np.log10(phib), color = 'r', label ='fit')
    ax[0,0].plot(M, np.log10(phir), color = 'b', label ='fit')
    ax[0,0].plot(M, np.log10(rphi), color = 'r', label ='fit', linestyle = '--')
    ax[0,0].plot(M, np.log10(bphi), color = 'b', label ='fit', linestyle = '--')
    ax[0,0].plot(M, np.log10(y), color = 'k', label ='fit', linestyle = ':')
    ax[0,0].plot(M, np.log10(y2), color = 'g', label ='fit', linestyle = ':')
    ax[0,0].plot(M, np.log10(y3), color = 'm', label ='fit', linestyle = ':')
    ax[0,0].plot(M2, np.log10(y4), color = 'k', label ='fit', linewidth = '4')
    ax[0,0].set_ylim(-6,0)
    plt.savefig('img/baldry.pdf')
    plt.legend()
    return phis, M, baldry

def log_schechter_fit(schechX, schechY):
    popt, pcov = curve_fit(log_schechter, schechX, schechY)
    return popt

def throw_error(df, bins, frac):
    throws = len(df)
    num = int(throws*frac)
    datarr = np.zeros((num, len(bins)-1))
    rows = df.index.tolist()
    for i in range(0, int(throws*frac)):
        print (i)
        newrows = random.sample(rows, num)
        newdf = df.ix[newrows]
        for index, row in newdf.iterrows():
            if row['LOGMH2_ERR'] == 0:
                newdf.ix[index, 'new_LOGMH2'] = row['MH2']
            else:
                newdf.ix[index, 'new_LOGMH2'] = np.random.normal(loc = row['LOGMH2'], scale = row['LOGMH2_ERR'])
        rho = Schechter2(newdf, bins)
        datarr[i,:] = rho[1]
    std = np.std(datarr, axis = 0)
    return std

def Vm(data, L):
    minz = min(data['Z_SDSS'])
    maxz = max(data['Z_SDSS'])
    # print (minz,maxz)
    # Omega = 0.483979888662
    Omega = 0.427304474238
    if L == 1: # low mass
        Omega = 0.353621392624
        N_COLDGASS = 89.0
        N_SDSS = 764.0
    elif L == 2: # high mass
        N_COLDGASS = 366.0
        N_SDSS = 12006.0
    elif L == 3:
        N_COLDGASS = 500.0
        N_SDSS = 12006.0
    VVmlist = np.zeros((len(data),1))
    Vmlist = np.zeros((len(data),1))
    x = pd.DataFrame(minz, index = [0], columns=['Z_SDSS'])
    y = pd.DataFrame(maxz, index = [0], columns=['Z_SDSS'])
    D_in = float(lumdistance(x))
    D_out = float(lumdistance(y))
    Vm =  (1.0/3.0)*(N_COLDGASS/N_SDSS)*((D_out**3)-(D_in**3))*(Omega)
    for idx,row in data.iterrows():
        Dl = row['D_L']
        V = ((4*math.pi)/3)*Dl*Dl*Dl
        VVmlist[idx,0] = (V/Vm)
        Vmlist[idx,0] = Vm
    # data = np.hstack((data,VVmlist))
    # data = np.hstack((data,Vmlist))
    return Vmlist

def Schechter(data, bins):
    l = data['MH2']
    # l = np.log10(l)
    rho, N, xbins, sigma, rhoH2 = [], [], [], [], []
    for i in range (1,len(bins)):
        p, Num, o, pH2 = 0, 0, 0, 0
        for j in range(0,len(l)):
            if l[j] >= bins[i-1] and l[j] < bins[i]:
                p += 1/data.loc[j,'V_m']
                o += 1/(data.loc[j,'V_m']**2)
                pH2 += data.loc[j,'MH2']/data.loc[j,'V_m']
                Num+=1
        N.append(Num)
        xbins.append((bins[i]+bins[i-1])/2)
        rho.append(p/(bins[1]-bins[0]))
        sigma.append(math.sqrt(o))
        rhoH2.append(pH2/(bins[1]-bins[0]))
    # return the Number of gals, log10(density), centre pt of each bin
    return [N, np.log10(rho), xbins, np.log10(sigma), np.log10(rhoH2)]

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

def SFR_hist():
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    SFR = np.linspace(-3,1,200)
    phi = []
    phi2 = []
    for idx, element in enumerate(SFR):
        phi.append(quad(integrand_SFR, 0, 12, args=(element))[0])
        phi2.append(quad(integrand_SFR, 6, 12, args=(element))[0])
    ax[0,0].plot(SFR,np.log10(phi))
    ax[0,0].plot(SFR,np.log10(phi2))
    ax[0,0].set_xlabel(r'$\rm log \,SFR \, [M_{\odot}\,yr^{-1}]$')
    ax[0,0].set_ylabel(r'$\rm log \,\phi(SFR) \, [Mpc^{-3}]$')
    ax[0,0].set_ylim(-4,-0.5)
    plt.tight_layout()
    plt.savefig('img/SFR_hist.pdf')

def MH2_hist(n, N, SFR_MH2_chain, Mstar_SFR_chain, fname):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    MH2 = np.linspace(5,12,n)
    phi = []
    phi2 = []
    best_fits = np.zeros((N,n))
    for idx, element in enumerate(MH2):
        phi.append(dblquad(integrand_MH2_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element,))[0])
    for i in range(0,N):
        print (i)
        MS_params = Mstar_SFR_chain[np.random.choice(Mstar_SFR_chain.shape[0], size=1, replace=False), :]
        SFR_MH2_params = SFR_MH2_chain[np.random.choice(SFR_MH2_chain.shape[0], size=1, replace=False), :]
        for idx, element in enumerate(MH2):
            best_fits[i,idx] = dblquad(integrand_MH2_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element,MS_params,SFR_MH2_params))[0]

        # phi2.append(dblquad(integrand_MH2_red, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element,))[0])
    ax[0,0].plot(MH2,np.log10(phi), color = 'b')
    for i in range(0,N):
        ax[0,0].plot(MH2, np.log10(best_fits[i,:]), alpha = 0.1, color = 'g')
    # ax[0,0].plot(MH2,np.log10(phi2), color = 'r')
    # ax[0,0].plot(MH2,np.log10(np.array(phi)+np.array(phi2)), color = 'k')
    ax[0,0].set_xlabel(r'$\rm log \,M_{H2} \, [M_{\odot}]$')
    ax[0,0].set_ylabel(r'$\rm log \,\phi(M_{H2}) \, [Mpc^{-3}]$')
    ax[0,0].set_ylim(-5,0)
    ax[0,0].set_xlim(6,10.5)
    plt.tight_layout()
    plt.savefig(fname)

def MHI_hist(n, N, SFR_MH2_chain, Mstar_SFR_chain, fname):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    MH2 = np.linspace(5,12,n)
    phi = []
    phi2 = []
    best_fits = np.zeros((N,n))
    for idx, element in enumerate(MH2):
        phi.append(dblquad(integrand_MH2_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element,))[0])
    for i in range(0,N):
        print (i)
        MS_params = Mstar_SFR_chain[np.random.choice(Mstar_SFR_chain.shape[0], size=1, replace=False), :]
        SFR_MH2_params = SFR_MH2_chain[np.random.choice(SFR_MH2_chain.shape[0], size=1, replace=False), :]
        for idx, element in enumerate(MH2):
            best_fits[i,idx] = dblquad(integrand_MH2_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element,MS_params,SFR_MH2_params))[0]

        # phi2.append(dblquad(integrand_MH2_red, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element,))[0])
    # ax[0,0].plot(MH2,np.log10(phi), color = 'b')
    for i in range(0,N):
        ax[0,0].plot(MH2, np.log10(best_fits[i,:]), alpha = 0.1, color = 'g')

    y2 = log_schechter_true(MH2, 4.8E-3*(0.7**3), 9.96+(2*np.log10(0.7)), -1.33)
    y3 = log_schechter_true(MH2, 4.8E-3, 9.96, -1.33)
    ax[0,0].plot(MH2, np.log10(y2), color = 'k')
    ax[0,0].plot(MH2, np.log10(y3), color = 'g')
    # ax[0,0].plot(MH2,np.log10(phi2), color = 'r')
    # ax[0,0].plot(MH2,np.log10(np.array(phi)+np.array(phi2)), color = 'k')
    ax[0,0].set_xlabel(r'$\rm log \,M_{HI} \, [M_{\odot}]$')
    ax[0,0].set_ylabel(r'$\rm log \,\phi(M_{HI}) \, [Mpc^{-3}]$')
    ax[0,0].set_ylim(-6,0)
    ax[0,0].set_xlim(6,11)
    plt.tight_layout()
    plt.savefig(fname)

def MHI_hist2(n, N, chain, fname):
    # read in the ALFA ALFA datasets from the 40% paper
    ALFAALFA = pd.read_csv('ALFAALFA.csv', comment = '#', header = None, sep=",")
    ALFAALFA.columns = ['x', 'y', 'dy', 'MHI', 'phi', 'err', 'phi_err']
    ALFAALFA = ALFAALFA[np.isfinite(ALFAALFA['phi_err'])]
    MH2, phi_alfa, phi_err_alfa = np.round(ALFAALFA['MHI'].values,2), ALFAALFA['phi'].values, ALFAALFA['phi_err'].values

    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    # MH2 = np.linspace(5,12,n)
    n = len(MH2)
    phi = []
    phi2 = []
    best_fits = np.zeros((N,n))
    # for idx, element in enumerate(MH2):
    #     phi.append(dblquad(integrand_MHI_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element,))[0])
    for i in range(0,N):
        params = chain[np.random.choice(chain.shape[0], size=1, replace=False), :]
        print (i)
        a1, a2, a3, lnf, b1, b2, lnf1, c1, c2, lnf2 = params[0]
        for idx, element in enumerate(MH2):
            integrand = dblquad(integrand_MHI, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, a1, a2, a3, lnf, b1, b2, lnf1))
            # print ('@', integrand)
            best_fits[i,idx] = np.log10(integrand[0])
    for i in range(0,N):
        ax[0,0].plot(MH2, best_fits[i,:], alpha = 0.1, color = 'g')

    y2 = log_schechter_true(MH2, 4.8E-3*(0.7**3), 9.96+(2*np.log10(0.7)), -1.33)
    y3 = log_schechter_true(MH2, 4.8E-3, 9.96, -1.33)
    ax[0,0].plot(MH2, np.log10(y2), color = 'k')
    ax[0,0].plot(MH2, np.log10(y3), color = 'g')
    ax[0,0].errorbar(MH2, phi_alfa, yerr = phi_err_alfa, fmt='o', capsize = 2, markersize = 3, linewidth=2, markeredgewidth=2, capthick=2, mfc='gray', mec='gray', ecolor = 'gray')


    # ax[0,0].plot(MH2,np.log10(phi2), color = 'r')
    # ax[0,0].plot(MH2,np.log10(np.array(phi)+np.array(phi2)), color = 'k')
    ax[0,0].set_xlabel(r'$\rm log \,M_{HI} \, [M_{\odot}]$')
    ax[0,0].set_ylabel(r'$\rm log \,\phi(M_{HI}) \, [Mpc^{-3}]$')
    ax[0,0].set_ylim(-6,0)
    ax[0,0].set_xlim(6,11)
    plt.tight_layout()
    plt.savefig(fname)

def plot_SFR_MH2_fit(xCOLDGASS_data, xCOLDGASS_nondet, sampler_marg, params, SFR, sample_likelihood):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    ax[0,0].errorbar(xCOLDGASS_data['LOGSFR_BEST'], xCOLDGASS_data['MH2'], xerr = xCOLDGASS_data['LOGSFR_ERR'], yerr = xCOLDGASS_data['LOGMH2_ERR'], fmt='o', capsize = 0.1, markersize = 1, linewidth=0.1, markeredgewidth=0.1, capthick=0.1, mfc='darkgray', mec='gray', ecolor = 'gray')
    ax[0,0].errorbar(xCOLDGASS_nondet['LOGSFR_BEST'], xCOLDGASS_nondet['MH2'], xerr = xCOLDGASS_nondet['LOGSFR_ERR'], yerr = 0.2, uplims = True, fmt='o', capsize = 0.1, markersize = 1, linewidth=0.1, markeredgewidth=0.1, capthick=0.1, mfc='b', mec='b', ecolor = 'b')

    # Plot posterior predictions for a few samples.
    samples_marg = sampler_marg.flatchain
    x0 = np.array([-2.5, 1.5])

    for m, b, _ in samples_marg[np.random.randint(len(samples_marg), size=200)]:
        ax[0,0].plot(x0, m*x0 + b, lw=1, alpha=0.1, color="g")

    # for idx, element in enumerate(sample_likelihood):
    #     ax[0,0].scatter(SFR, element, color = 'b', s = 0.5, alpha = 0.8)
    ax[0,0].set_xlabel(r'$\log\,SFR\,[M_{\odot}yr^{-1}]$')
    ax[0,0].set_ylabel(r'$\log\,M_{H2}\,[M_{\odot}]$')
    ax[0,0].set_xlim(-3,2)
    ax[0,0].set_ylim(7,10.5)
    plt.tight_layout()
    plt.savefig('img/SFR_MH2.pdf')

def SFR_MH2_fit(xCOLDGASS_data):
    xCOLDGASS_data = xCOLDGASS_data[np.isfinite(xCOLDGASS_data['LOGSFR_BEST'])]
    xCOLDGASS_data = xCOLDGASS_data[np.isfinite(xCOLDGASS_data['MH2'])]
    nll = lambda *args: -log_likelihood_SFR_MH2(*args)
    initial = np.array([0.7, 9, 0.1])
    soln3 = minimize(nll, initial, args=(xCOLDGASS_data['LOGSFR_BEST'], xCOLDGASS_data['MH2'], xCOLDGASS_data['LOGMH2_ERR']))
    # print ('soln binned')
    # print (soln3.x)
    # print (soln3["x"][0])
    # print (xCOLDGASS_data[['MH2', 'LOGMH2_ERR', 'LOGSFR_BEST', 'LOGSFR_ERR']])
    xCOLDGASS_nondet = xCOLDGASS_data[xCOLDGASS_data['LOGMH2_ERR']==0]
    xCOLDGASS_data = xCOLDGASS_data[xCOLDGASS_data['LOGMH2_ERR']>0]
    xCOLDGASS_nondet['LOGMH2_ERR'] = 0.14
    x, y, xerr, yerr = xCOLDGASS_data['LOGSFR_BEST'].values, xCOLDGASS_data['MH2'].values, xCOLDGASS_data['LOGSFR_ERR'].values, xCOLDGASS_data['LOGMH2_ERR'].values
    x2, y2, xerr2, yerr2 = xCOLDGASS_nondet['LOGSFR_BEST'].values, xCOLDGASS_nondet['MH2'].values, xCOLDGASS_nondet['LOGSFR_ERR'].values, xCOLDGASS_nondet['LOGMH2_ERR'].values
    print (x2,y2)
    S = S_error(xerr, yerr)
    S2 = S_error(xerr2, [0.14])
    # print (S2)

    ndim, nwalkers = 3, 100
    sampler_marg = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (x, y, x2, y2, S, S2))
    pos = [[0.69, 9.01, 0.2] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler_marg.run_mcmc(pos, 500, progress=True)
    plot_samples(sampler_marg, ndim, 'MH2-SFR')
    samples = sampler_marg.chain[:, 150:, :].reshape((-1, ndim))
    # samples2 = np.copy(samples)
    # samples2[:, 2] = np.exp(samples2[:, 2])
    # m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples2, [16, 50, 84],axis=0)))
    # print (m_mcmc, b_mcmc, f_mcmc)
    # params = m_mcmc[0], b_mcmc[0], f_mcmc[0]
    # print (params)
    # SFR = 0.0
    # sample_likelihood = sample_log_marg_prob(params, SFR, L)
    plot_SFR_MH2_fit(xCOLDGASS_data, xCOLDGASS_nondet, sampler_marg, params, SFR, sample_likelihood)
    plot_corner(samples, 'sfrH2corner.pdf')
    return samples

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

def plotGASS(det,nondet, samples_marg):
    # print (np.shape(samples_marg.flatchain))
    # print (samples_marg.flatchain)
    # samples_marg = samples_marg.flatchain[200:, :]
    x0 = np.array([-3, 2])
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))

    for m, b, _ in samples_marg[np.random.randint(len(samples_marg), size=200)]:
        ax[0,0].plot(x0, m*x0 + b, lw=1, alpha=0.1, color="g")

    ax[0,0].errorbar((det['SFR_best']), det['lgMHI'], xerr = (det['SFRerr_best']), yerr = 0.2, fmt='o', capsize = .1, markersize = .4, linewidth=.1, markeredgewidth=.1, capthick=.1, mfc='gray', mec='gray', ecolor = 'gray')
    ax[0,0].errorbar((nondet['SFR_best']), nondet['lgMHI'], xerr = (nondet['SFRerr_best']), yerr = 0.2, uplims = True, fmt='o', capsize = .1, markersize = .4, linewidth=.1, markeredgewidth=.1, capthick=.1, mfc='b', mec='b', ecolor = 'b')
    ax[0,0].set_xlim(-3,2)
    ax[0,0].set_ylim(7,11)
    plt.savefig('img/HI-SFR.pdf')

def read_GASS():
    xxGASS = fits.open('data/xxGASS_MASTER_CO_170620_final.fits')
    xxGASS = Table(xxGASS[1].data).to_pandas()
    xxGASS = xxGASS[xxGASS['SFR_best'] > -80]
    data = xxGASS[['SFR_best', 'lgMHI', 'SFRerr_best', 'HIsrc']]
    det = data[data['HIsrc']!=4]
    nondet = data[data['HIsrc']==4]
    det['SFRerr_best'] = det['SFRerr_best']/(det['SFR_best']*np.log(10))
    det['SFR_best'] = np.log10(det['SFR_best'])
    nondet['SFRerr_best'] = nondet['SFRerr_best']/(nondet['SFR_best']*np.log(10))
    nondet['SFR_best'] = np.log10(nondet['SFR_best'])
    return xxGASS, det, nondet

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

def GASS():
    # read in the xxGASS data
    xxGASS, det, nondet = read_GASS()
    # calculate error matrices etc
    S = S_error(det['SFRerr_best'].values, [0.2])
    S2 = S_error(nondet['SFRerr_best'].values, [0.14])
    x, y = det['SFR_best'].values, det['lgMHI'].values
    x2, y2 = nondet['SFR_best'].values, nondet['lgMHI'].values
    # emcee with initial guess
    ndim, nwalkers = 3, 100
    sampler_marg = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (x, y, x2, y2, S, S2))
    pos = [[0.8, 9.5, -0.4] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler_marg.run_mcmc(pos, 1000, progress=True)
    # plot the walkers
    plot_samples(sampler_marg, ndim, 'MHI-SFR')
    # cut after they converge
    samples = sampler_marg.chain[:, 200:, :].reshape((-1, ndim))
    # plot corner plot after convergence
    plot_corner(samples, 'sfrHIcorner.pdf')
    # plot the SFR-MHI scaling relation with fits after convergence
    plotGASS(det,nondet, samples)
    return samples

def plot_corner(samples_input, fname):
    samples_input[:, 2] = np.exp(samples_input[:, 2])
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    corner.corner(samples_input, labels=["slope", "y-intercept", "scatter"],
                  truths=(np.median(samples_input[:, 0]), np.median(samples_input[:, 1]), np.median(samples_input[:, 2])),
                  truth_color="k",
                  quantiles=[0.16, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('img/corner/' + fname)

def plot_corner2(samples_input, fname):
    samples_input[:, 3] = np.exp(samples_input[:, 3])
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    corner.corner(samples_input, labels=["a", "b", "c", "scatter"],
                  truths=(np.median(samples_input[:, 0]), np.median(samples_input[:, 1]), np.median(samples_input[:, 2]), np.median(samples_input[:, 3])),
                  truth_color="k",
                  quantiles=[0.16, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('img/corner/' + fname)

def plot_corner3(samples_input, fname):
    samples_input[:, 3] = np.exp(samples_input[:, 3])
    samples_input[:, 6] = np.exp(samples_input[:, 6])
    samples_input[:, 9] = np.exp(samples_input[:, 9])
    corner.corner(samples_input, labels=["a", "a1", "a2", "f", "b1", "b2", "f2", "c1", "c2", "f3"],
                  truths=(np.median(samples_input[:, 0]), np.median(samples_input[:, 1]), np.median(samples_input[:, 2]), np.median(samples_input[:, 3]), np.median(samples_input[:, 4]), np.median(samples_input[:, 5]), np.median(samples_input[:, 6]), np.median(samples_input[:, 7]), np.median(samples_input[:, 8]), np.median(samples_input[:, 9])),
                  truth_color="k",
                  quantiles=[0.16, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('img/corner/' + fname)

def log_prior_all(theta):
    a1, a2, a3, lnf, b1, b2, lnf1, c1, c2, lnf2 = theta
    if -1.0 < a1 < 1.0 and 0.0 < a2 < 5.0 and -20.0 < a3 < -5.0 and -5.0 < lnf < 5.0 and 0.6 < b1 < 1.0 and 8.0 < b2 < 11.0 and -2 < lnf1 < 2 and 0.4 < c1 < 1.0 and 8 < c2 < 10 and -5.0 < lnf2 < 5.0:
        return 0.0
    return -np.inf

def all_log_probability(theta, x, y, xerr, yerr, x1, x2, y1, y2, S1, S2, CGx1, CGx2, CGy1, CGy2, CGS1, CGS2, MHI, phi_alfa, phi_err_alfa):
    a1, a2, a3, lnf, b1, b2, lnf1, c1, c2, lnf2 = theta
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood for the linear SFR-MHI plane
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = np.array([-b1, 1.0])
    sigma = np.dot(np.dot(S1, v), v) + np.exp(2 * lnf1)
    sigma2 = np.dot(np.dot(S2, v), v) + np.exp(2 * lnf1)
    sigma2 = sigma2 ** 0.5
    deltaN = y1 - (b1 * x1) - b2
    model = (b1 * x2) + b2
    ll1 = -0.5 * np.sum(np.square(deltaN) / sigma + np.log(sigma))
    I = np.zeros(len(x2))
    for i in range(0,len(x2)):
        I[i] = np.log(((2 * np.pi) ** 0.5) *
                      0.5 * (special.erf((y2[i]-model[i]) / ((2 ** 0.5) * sigma2[i])) + 1))
    ll2 = np.sum(I)
    LL_SFR_MHI = ll1  + ll2
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood for the linear SFR-MH2 plane
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = np.array([-c1, 1.0])
    sigma = np.dot(np.dot(CGS1, v), v) + np.exp(2 * lnf2)
    sigma2 = np.dot(np.dot(CGS2, v), v) + np.exp(2 * lnf2)
    sigma2 = sigma2 ** 0.5
    deltaN = CGy1 - (c1 * CGx1) - c2
    model = (c1 * CGx2) + c2
    ll1 = -0.5 * np.sum(np.square(deltaN) / sigma + np.log(sigma))
    I = np.zeros(len(CGx2))
    for i in range(0,len(CGx2)):
        I[i] = np.log(((2 * np.pi) ** 0.5) *
                      0.5 * (special.erf((CGy2[i]-model[i]) / ((2 ** 0.5) * sigma2[i])) + 1))
    ll2 = np.sum(I)
    LL_SFR_MH2 = ll1  + ll2
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood for the SFR-M* plane
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2 = np.square(xerr)*np.square((2*a1*x) + a2) + np.square(yerr) + np.exp(2*lnf)
    DeltaN = y - (a1*x*x) - (a2*x) - a3
    LL_SFRM = -0.5 * np.sum(DeltaN**2/Sigma2 + np.log(Sigma2))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood delta MHI
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    best_fits = np.zeros((1,len(MHI)))
    for idx, element in enumerate(MHI):
        integral = dblquad(integrand_MHI, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, a1, a2, a3, lnf, b1, b2, lnf1))
        # print (integral)
        best_fits[0,idx] = np.log10(integral[0])
    delta = phi_alfa - best_fits
    LL_Delta_phi = -0.5 * np.sum(delta**2/phi_err_alfa + np.log(phi_err_alfa))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return LL_SFR_MHI + LL_SFRM + LL_SFR_MH2 + LL_Delta_phi + log_prior_all(theta)

def all_fits():
    # read in the GAMA data for z<0.08
    GAMA, GAMAb, GAMAr = read_GAMA()
    x, y, xerr, yerr = GAMAb['logM*'], GAMAb['logSFR'], GAMAb['logM*err'], GAMAb['logSFRerr']
    # read in COLD GASS data
    xCOLDGASS_data = read_COLD_GASS()
    xCOLDGASS_data = xCOLDGASS_data[np.isfinite(xCOLDGASS_data['LOGSFR_BEST'])]
    xCOLDGASS_data = xCOLDGASS_data[np.isfinite(xCOLDGASS_data['MH2'])]
    xCOLDGASS_nondet = xCOLDGASS_data[xCOLDGASS_data['LOGMH2_ERR']==0]
    xCOLDGASS_data = xCOLDGASS_data[xCOLDGASS_data['LOGMH2_ERR']>0]
    xCOLDGASS_nondet['LOGMH2_ERR'] = 0.14
    CGx1, CGy1, CGxerr, CGyerr = xCOLDGASS_data['LOGSFR_BEST'].values, xCOLDGASS_data['MH2'].values, xCOLDGASS_data['LOGSFR_ERR'].values, xCOLDGASS_data['LOGMH2_ERR'].values
    CGx2, CGy2, CGxerr2, CGyerr2 = xCOLDGASS_nondet['LOGSFR_BEST'].values, xCOLDGASS_nondet['MH2'].values, xCOLDGASS_nondet['LOGSFR_ERR'].values, xCOLDGASS_nondet['LOGMH2_ERR'].values
    CGS1 = S_error(CGxerr, CGyerr)
    CGS2 = S_error(CGxerr2, [0.14])
    # read in the xxGASS data
    xxGASS, det, nondet = read_GASS()
        # read in the ALFA ALFA datasets from the 40% paper
    ALFAALFA = pd.read_csv('ALFAALFA.csv', comment = '#', header = None, sep=",")
    ALFAALFA.columns = ['x', 'y', 'dy', 'MHI', 'phi', 'err', 'phi_err']
    ALFAALFA = ALFAALFA[np.isfinite(ALFAALFA['phi_err'])]
    MHI, phi_alfa, phi_err_alfa = np.round(ALFAALFA['MHI'].values,2), ALFAALFA['phi'].values, ALFAALFA['phi_err'].values
    # calculate error matrices etc
    S1 = S_error(det['SFRerr_best'].values, [0.2])
    S2 = S_error(nondet['SFRerr_best'].values, [0.14])
    x1, y1 = det['SFR_best'].values, det['lgMHI'].values
    x2, y2 = nondet['SFR_best'].values, nondet['lgMHI'].values
    # emcee
    ndim, nwalkers = 10, 30
    g_SFRM = [-0.07254516, 2.02271974, -13., -1.2]
    g_MHI = [0.8, 9.5, -0.9]
    g_MH2 = [.69, 9.01, 0.2]
    g = np.append(g_SFRM, g_MHI)
    g = np.append(g, g_MH2)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, all_log_probability, args = (x, y, xerr, yerr, x1, x2, y1, y2, S1, S2, CGx1, CGx2, CGy1, CGy2, CGS1, CGS2, MHI, phi_alfa, phi_err_alfa))
    pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(pos, 500, progress=True)
    # plot the walkers
    plot_samples7(sampler, ndim, 'all')
    # cut after they converge
    samples = sampler.chain[:, 400:, :].reshape((-1, ndim))
    # plot corner plot after convergence
    plot_corner3(samples, 'all.pdf')
    # plot the SFR-MHI scaling relation with fits after convergence
    # plotGASS(det,nondet, samples)
    return samples

def test_schechter():
    x = np.linspace(6,11,100)
    x2 = np.linspace(10E6,10E11,100)
    phi = 4.8E-3*(0.7**3)
    mass = 9.96+(2*np.log10(0.7))
    m2 = np.power(10,mass)
    alpha = -1.33
    y2 = log_schechter_true(x, phi, mass, alpha)
    y3 = (phi/m2)*np.power(x2/m2, alpha)*np.exp(-x2/m2)
    print (y3)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    ax[0,0].plot(x,np.log10(y2), color = 'b', alpha = 0.5)
    ax[0,0].plot(np.log10(x2),np.log10(y3), color = 'k', alpha = 0.5)
    # ax[0,0].set_ylim(-6,0)
    ax[0,0].set_xlim(6,11)
    plt.savefig('img/test_schechter.pdf')

## MAIN ########################################################################
random.seed(42)

# n=10
# MH2 = np.linspace(5,12,n)
# best_fits = np.zeros((1,n))
# # print (type(params))
# for idx, element in enumerate(MH2):
#     # print (idx)
#     best_fits[0,idx] = dblquad(integrand_MHI_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element,-0.07, 1.90, -12.35, 0.31, 0.80, 9.49, 0.44, 0.85, 8.92, 0.26))[0]
# test_schechter()
chain = all_fits()
chain = np.savetxt('converged2.txt', chain)

# chain = np.loadtxt('converged.txt')
# params = chain[np.random.choice(chain.shape[0], size=1, replace=False), :]
# a1, a2, a3, lnf, b1, b2, lnf1, c1, c2, lnf2 = params[0]
# print (a1)
# print (params)
# MHI_hist2(20, 10, chain, 'img/MHI2_hist.pdf')


# SFRM_plane()

# bins = np.linspace(5.5,11,18)
# bootstrap = 0.8
# #
# ##############################################################################
# # Producing the pandas catalogue
#

################################################################################

# Doing the M*-SFR emcee fit and plot
# Mstar_SFR_chain = MainSequence()
# Doing the SFR-MH2 emcee fit and plot
# xCOLDGASS_data = read_COLD_GASS()
# SFR_MH2_chain = SFR_MH2_fit(xCOLDGASS_data)
# Doing the SFR-MHI emcee fit and plot
# SFR_MHI_chain = GASS()

# plotting the H2 histograms sampling the emcee fits
# MH2_hist(30, 100, SFR_MH2_chain, Mstar_SFR_chain, 'img/MH2_hist.pdf')
# plotting the HI histograms sampling the emcee fits
# MHI_hist(30, 10, SFR_MHI_chain, Mstar_SFR_chain, 'img/MHI_hist.pdf')


# read_GAMA_A()
# SFR_hist()
# print (quad(integrand_SFR, 6, 12, args=(-0.3)))
# print (quad(integrand_stellar_mass, 6, 12))
# print (dblquad(integrand_MH2_blue, -5.0, 2.0, lambda SFR: 4.0, lambda SFR: 12.0, args = (9,)))
# doubleschec(7, 11.5)
# # plot_P_SFR_Mstar()
# plot_MH2_SFR()

# x=np.linspace(8,12,500)
# x2=np.linspace(1E8,1E12,1000)
# Mstar = np.power(10,10.66)
# phistar1 = 3.96E-3
# phistar2 = 0.79E-3
# alpha1 = - 0.35
# alpha2 = - 1.47
# y=log_schechter(x,np.log10(phistar1), np.log10(Mstar), alpha1)
# # y3 = log_schechter1(x,np.log10(phistar1), np.log10(Mstar), alpha1)
# y2 = phistar1*((10**x/Mstar)**(alpha1))*np.exp(-10**x/Mstar)*np.log(10)
# print (y2)
# plt.plot(x,y, color = 'k')
# plt.plot(x,np.log10(y2), color = 'b')
# # plt.plot(x,np.log10(y3), color = 'g')
# plt.ylim(-7,0)
# plt.show()




#
# bins = np.linspace(7.5,10.5,20)
# prods = Schechter(xCOLDGASS_data, bins)
# # print (prods[2])
# # print ()
# # print (prods[1])
# # print ()
#
# # print (error_arr)
# # print (len(prods[2]))
# # print (prods[1][10:-2])
# error_arr = [       np.nan, 0.11270569, 0.07213719, 0.08468254, 0.06172631, 0.03854405,
#  0.04042781, 0.0381058,  0.04738866, 0.0587743,  0.06472507, 0.07389062,
#  0.08438962, 0.09023383, 0.11791274, 0.15188817, np.nan, np.nan, np.nan]
# # error_arr = np.array(error_arr)
# # ind = np.where((prods[1] > 0) & (prods[1] < 10000000) & (prods[2] > 8.4) & (prods[2] < 20))
# nll = lambda *args: -log_likelihood(*args)
# initial = np.array([0.0001, 9.5, -1.1])
# # print (type())
# soln = minimize(nll, initial, args=(np.array(prods[2][1:-2]), np.array(prods[1][1:-2]), np.array(error_arr[1:-2])))
# # print("Maximum likelihood estimates:",soln.x[0:3])
#
# popt = log_schechter_fit(prods[2][6:-2], prods[1][6:-2])
# # print (popt)
# # print (xCOLDGASS_data['LOGMH2_ERR'])
# bins2= np.linspace(7.5,10.5,200)
# # error_arr = throw_error(xCOLDGASS_data, bins, bootstrap)
# y = log_schechter(bins2, *popt)
# # print (error_arr)
# plt.scatter(prods[2], prods[1])
# # plt.errorbar(bins,)
# plt.ylim(-5,-1.5)
# plt.plot(bins2,y)
# plt.errorbar(prods[2], prods[1],yerr=error_arr)
# plt.show()
