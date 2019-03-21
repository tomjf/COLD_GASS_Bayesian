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

def draw_from_chain3(n, N, chain, MH2):
    best_fits = np.zeros((N,n))
    for i in range(0,N):
        print (i)
        params = chain[np.random.choice(chain.shape[0], size=1, replace=False), :]
        a1, a2, a3, s1, A1, d1, d2, lnf4 = params[0]
        for idx, element in enumerate(MH2):
            integrand = nquad(integrand_MHI_logistic, [[0,12], [-5,2]], args = (element, a1, a2, a3, s1, A1, d1, d2, lnf4), opts = [{'epsrel': 1e-2}, {'epsabs':0}])
            best_fits[i,idx] = np.log10(integrand[0])
    return best_fits

def MHI_hist2(n, N, chain, fname):
    # read in the ALFA ALFA datasets from the 40% paper
    ALFAALFA = pd.read_csv('ALFAALFA.csv', comment = '#', header = None, sep=",")
    ALFAALFA.columns = ['x', 'y', 'dy', 'MHI', 'phi', 'err', 'phi_err']
    ALFAALFA = ALFAALFA[np.isfinite(ALFAALFA['phi_err'])]
    MH2, phi_alfa, phi_err_alfa = np.round(ALFAALFA['MHI'].values,2), ALFAALFA['phi'].values, ALFAALFA['phi_err'].values

    ALFAALFA2 = pd.read_csv('data/a99_HIMF.dat', comment = '#', sep=",")
    MH22, phi_alfa2, phi_err_alfa2 = np.round(ALFAALFA['MHI'].values,2), ALFAALFA['phi'].values, ALFAALFA['phi_err'].values
    print (ALFAALFA2)

    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    # MH2 = np.linspace(5,12,n)
    n = len(MH2)

    # best_fits3 = draw_from_chain2(n, N, chain3, MH22)
    # print ('chain three done')
    best_fits = draw_from_chain3(n, N, chain, MH22)
    print ('chain one done')
    # best_fits2 = draw_from_chain(n, N, chain2, MH22)
    # print ('chain two done')


    for i in range(0,N):
        ax[0,0].plot(MH2, best_fits[i,:], alpha = 0.1, color = 'g')
        # ax[0,0].plot(MH2, best_fits2[i,:], alpha = 0.1, color = 'b')
        # ax[0,0].plot(MH2, best_fits3[i,:], alpha = 0.1, color = 'r')

    y2 = log_schechter_true(MH2, 4.8E-3*(0.7**3), 9.96+(2*np.log10(0.7)), -1.33)
    y3 = log_schechter_true(MH2, 4.8E-3, 9.96, -1.33)
    # ax[0,0].plot(MH2, np.log10(y2), color = 'k')
    ax[0,0].plot(MH2, np.log10(y3), color = 'g')
    ax[0,0].errorbar(MH2, phi_alfa, yerr = phi_err_alfa, fmt='o', capsize = 2, markersize = 3, linewidth=2, markeredgewidth=2, capthick=2, mfc='gray', mec='gray', ecolor = 'gray')
    ax[0,0].errorbar(MH22, phi_alfa2, yerr = phi_err_alfa2, fmt='o', capsize = 2, markersize = 3, linewidth=2, markeredgewidth=2, capthick=2, mfc='k', mec='k', ecolor = 'k')

    # ax[0,0].plot(MH2,np.log10(phi2), color = 'r')
    # ax[0,0].plot(MH2,np.log10(np.array(phi)+np.array(phi2)), color = 'k')
    ax[0,0].set_xlabel(r'$\rm log \,M_{HI} \, [M_{\odot}]$')
    ax[0,0].set_ylabel(r'$\rm log \,\phi(M_{HI}) \, [Mpc^{-3}]$')
    ax[0,0].set_ylim(-6,0)
    ax[0,0].set_xlim(6,11)
    plt.tight_layout()
    plt.savefig(fname)


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
    plt.xlabel(r'$\mathrm{log M_{*}}$', fontsize = 20)
    plt.ylabel(r'$\mathrm{log \phi}$', fontsize = 20)
    plt.savefig('img/baldry.pdf')
    # plt.legend()
    return phis, M, baldry


def calcVm(data, numtot, bootstrap):
    answer, M, baldry = doubleschec(min(data['LOGMSTAR']), max(data['LOGMSTAR']))
    V_arb = 1E2
    N_arb = np.sum(np.multiply(answer,V_arb*(M[1]-M[0])))
    V_CG = V_arb*(numtot/N_arb)
    V_CG2 = V_arb*(int(numtot/(1/bootstrap))/N_arb)
    return V_CG, V_CG2

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

def plot_corner(samples_input, fname):
    corner.corner(samples_input, labels=[r"$a1$", r"$a2$", r"$a3$", r"$lnf$", r"$A$", r"$d1$", r"$d2$", r"$lnf2$"],
                  truths=(np.median(samples_input[:, 0]), np.median(samples_input[:, 1]), np.median(samples_input[:, 2]), np.median(samples_input[:, 3]), np.median(samples_input[:, 4]), np.median(samples_input[:, 5]), np.median(samples_input[:, 6]), np.median(samples_input[:, 7])),
                  truth_color="k",
                  quantiles=[0.16, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('img/corner/' + fname)

def plot_samples_full(sampler, ndim, fname):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(10,20))
    samples = sampler.get_chain()
    labels = ['sigma']
    ax[0,0].plot(samples[:, :, 0], "k", alpha=0.3)
    ax[0,0].set_xlim(0, len(samples))
    ax[0,0].set_ylabel(labels[0])
    ax[0,0].yaxis.set_label_coords(-0.1, 0.5)
    ax[0,0].set_xlabel("step number");
    plt.savefig('img/sampler' + fname + '.pdf')

def plot_samples(sampler, ndim, fname):
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["a1", "a2", "a3", "c1", "c2", "A", "d1", "d2", "log(f2)",]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    plt.savefig('img/sampler' + fname + '.pdf')

def plot_SFR_M_plane2(GAMA, GAMAb, GAMAr, GAMA_sf, GAMA_pass, samples):
    converged_old = np.loadtxt('data/converged.txt', delimiter = ',')
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    # h, xedges_b, yedges_b, image_b = ax[0,2].hist2d(GAMAb['logM*'], GAMAb['logSFR'], cmap = 'Blues', bins = 30, label = 'GAMA data', cmin = 1, cmax = 300, alpha = 0.3)
    # h, xedges_r, yedges_r, image_r = ax[0,2].hist2d(GAMAr['logM*'], GAMAr['logSFR'], cmap = 'Reds', bins = 30, label = 'GAMA data', cmin = 1, cmax = 300, alpha = 0.3)
    # h, xedges_b, yedges_b, image_b = ax[0,0].hist2d(GAMA_sf['logM*'], GAMA_sf['logSFR'], cmap = 'Blues', bins = 30, label = 'GAMA data', cmin = 1, cmax = 300, alpha = 0.3)
    ax[0,0].scatter(GAMA_sf['logM*'], GAMA_sf['logSFR'], c = 'b', label = 'GAMA data', s = 0.1)

    x0=np.linspace(7,12,300)
    # y = (-0.06*x0*x0) + (1.75*x0) -11.80
    # y2 = (-0.06*x0*x0) + (1.81*x0) -11.80
    # ax[0,0].plot(x0, np.power(10,y), color = 'k')
    # ax[0,0].plot(x0, np.power(10,y2), color = 'b')
    # ax[0,0].plot(x0, (0.83*x0) + -9.5)
    # ax[0,1].plot(x0, (0.83*x0) + -9.5)
    for a1, a2, a3, c1, c2, A, d1, d2, lnf3 in samples[np.random.randint(len(samples), size=100)]:
        ax[0,0].plot(x0, a1*x0*x0 + a2*x0 + a3, lw=1, alpha=0.1, color="b")
    for alpha, beta, zeta, a1, a2, a3, s1, b1, b2, s2, c1, c2, lnf3, d1, d2, lnf4  in converged_old[np.random.randint(len(converged_old), size=100)]:
        ax[0,0].plot(x0, a1*x0*x0 + a2*x0 + a3, lw=1, alpha=0.1, color="g")
        # ax[0,1].plot(x0, a1*x0*x0 + a2*x0 + a3, lw=1, alpha=0.1, color="b")
        # ax[0,2].plot(x0, a1*x0*x0 + a2*x0 + a3, lw=1, alpha=0.1, color="b")
        # ax[0,2].plot(x0, b1*x0 + b2, lw=1, alpha=0.1, color="r")
    #
    # y0 = np.linspace(-3,1,200)
    # Mstar = 9.5
    # mu = (a1*Mstar*Mstar) + (a2*Mstar) + a3
    # x0 = generalised_logistic(y0, A, lnf, mu)
    # ax[0,0].plot((x0*30)+Mstar, y0, lw=3, alpha=0.1, color="b")
    # for a1, a2, a3, lnf, b1, b2, lnf2 in samples2[np.random.randint(len(samples2), size=10)]:
    #     print (a1, a2, a3)
    #     ax[0,0].plot(x0, a1*x0*x0 + a2*x0 + a3, lw=1, alpha=0.1, color="g")
    #     ax[0,0].plot(x0, b1*x0 + b2, lw=1, alpha=0.1, color="r")
    #
    # for a, b, c, _ in sampler2[np.random.randint(len(sampler2), size=100)]:
    #     ax[0,0].plot(x0, a*x0*x0 + b*x0 + c, lw=1, alpha=0.1, color="g")

    # ax[0,0].set_xlim(7,12)
    # # use the emcee run to show simulated SFR-M* plane
    # V = 1000000
    # dM = 0.1
    # # mass function for stellar mass
    # M = np.arange(5,14,dM)
    # Mstar = 10.66
    # phistar1 = 3.96E-3
    # phistar2 = 0.79E-3
    # alpha1 = - 0.35
    # alpha2 = - 1.47
    # # calculate mass function
    # phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    # # given a number of galaxies and volume
    # N = phi_Mstar_double*dM*V
    # # calculate the passive fraction
    # p_frac = f_passive(M,alpha,beta,zeta)
    # N_red = p_frac*N
    # N_blue = (1-p_frac)*N
    # N_red = N_red.astype(int)
    # print (N_red)
    # N_blue = N_blue.astype(int)
    # blue = predict_dist(a1, a2, a3, lnf, M, N_blue)
    # red = predict_dist(0, b1, b2, lnf2, M, N_red)
    # #
    # hb, xedges_b, yedges_b, image_b = ax[0,1].hist2d(blue[:,0], blue[:,1], cmap = 'Blues', bins = [xedges_b, yedges_b], label = 'GAMA data', cmin = 1, cmax = 1000, alpha = 0.3)
    # ax[0,1].contour(hr, extent = [xedges_r.min(), xedges_r.max(), yedges_r.min(), yedges_r.max()], linewidths = 1)
    # ax[0,1].contour(hb, extent = [xedges_b.min(), xedges_b.max(), yedges_b.min(), yedges_b.max()], linewidths = 1)
    # ax[0,0].set_ylim(-3.5,1.5)
    # ax[0,0].set_xlim(7.5,12.0)
    ax[0,0].set_xlabel(r'$\mathrm{log_{10}(M_{*})}$')
    ax[0,0].set_ylabel(r'$\mathrm{log_{10}(SFR)}$')
    ax[0,0].set_title('GAMA data')
    # ax[0,2].set_title('colour cut')
    plt.legend()
    # plt.xlabel(r'$\mathrm{log_{10}(M_{*})}$')
    # plt.ylabel(r'$\mathrm{log_{10}(SFR)}$')
    plt.savefig('img/sfrmplane3.png', dpi = 800)

def read_GAMA():
    GAMA = pd.read_csv('data/GAMA_sample.dat', comment = '#', header = None, sep = "\s+")
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

def log_mainsequence_priors_full(params):
    a1, a2, a3, sigma1, A1, d1, d2, lnf4 = params
    if -1.0 < a1 < 1.0 and \
    0.0 < a2 < 5.0 and \
    -20.0 < a3 < -5.0 and \
    0.0 < sigma1 < 1.5 and \
    0.1 < A1 < 1.0 and \
    0.6 < d1 < 1.0 and \
    8.0 < d2 < 11.0 and \
    -5.0 < lnf4 < 5.0:
        return 0
    return -np.inf

def lognormalslice_scatter():
    xb2, yb2, xerrb2, yerrb2 =  slice_data['M*'], slice_data['SFR'], slice_data['M*err'], slice_data['SFRerr']
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    ax[0,0].errorbar(xb2, yb2, xerr=xerrb2, yerr=yerrb2, fmt='o')
    plt.savefig('img/lognormalcatter.pdf')


def log_marg_mainsequence_full(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    a1, a2, a3, sigma1, A1, d1, d2, lnf4 = params
    CGx1, CGx2, CGy1, CGy2, CGS1, CGS2 = COLD_GASS_data
    x1, x2, y1, y2, S1, S2 = GASS_data
    xb, yb, xerrb, yerrb =  slice_data['logM*'], slice_data['logSFR'], slice_data['logM*err'], slice_data['logSFRerr']
    xb2, yb2, xerrb2, yerrb2 =  slice_data2['logM*'], slice_data2['logSFR'], slice_data2['logM*err'], slice_data2['logSFRerr']
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood for the linear SFR-MHI plane
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = np.array([-d1, 1.0])
    deltaN = y1 - (d1 * x1) - d2
    model = (d1 * x2) + d2
    sigma = np.dot(np.dot(S1, v), v) + np.exp(2 * lnf4)
    sigma2 = np.dot(np.dot(S2, v), v) + np.exp(2 * lnf4)
    sigma2 = sigma2 ** 0.5
    ll1 = -0.5 * np.sum(np.square(deltaN) / sigma + np.log(sigma))
    I = np.zeros(len(x2))
    for i in range(0,len(x2)):
        I[i] = np.log(((2 * np.pi) ** 0.5) *
                      0.5 * (special.erf((y2[i]-model[i]) / ((2 ** 0.5) * sigma2[i])) + 1))
    ll2 = np.sum(I)
    ll_SFR_MHI = ll1  + ll2
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # star forming likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_sf = np.square(xerrb)*np.square((2*a1*xb) + a2) + np.square(yerrb) + np.square(sigma1)
    Sigma_sf = np.sqrt(Sigma2_sf)
    DeltaN_sf = yb - ((a1*xb*xb) + (a2*xb) + a3)
    ll_sforming = np.sum(np.log(Sigma_sf) - (DeltaN_sf/A1) - np.log(A1) - ((1 + Sigma_sf)*np.log(1 + np.exp(-DeltaN_sf/A1))))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # passive galaxies likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sigma2_pass = np.square(xerrb2)*np.square(b1) + np.square(yerrb2) + np.exp(2*sigma2)
    # Sigma2_pass = np.sqrt(Sigma2_pass)
    # DeltaN_pass = yb2 - ((b1*xb2) + b2)
    # ll_passive = np.sum(np.log(Sigma2_pass) - (DeltaN_pass/A2) - np.log(A2) - ((1 + Sigma2_pass)*np.log(1 + np.exp(-DeltaN_pass/A2))))
    return ll_sforming + ll_SFR_MHI + log_mainsequence_priors_full(params)

def generalised_logistic(x, A, c, mu):
    y = (x-mu)/A
    return (c*np.exp(-y))/(A * np.power(1 + np.exp(-y), 1 + c))

def generalised_logistic2(x, c, mu):
    y = (x-mu)
    return (c*np.exp(-y))/np.power(1 + np.exp(-y), 1 + c)

def third_order(x, a1, a2, a3):
    return (a1*x*x) + (a2*x) + a3


def second_order(x, a1, a2):
    return (a1*x) + a2


def double_gauss(x,a1,mu1,sigma1,a2,mu2,sigma2):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-(mu2))**2/(2*sigma2**2))


def sfrm_evolution(GAMA_sf, GAMA_pass, GAMA, samples):
    # a1, a2, a3, lnf, A =
    rand_data = samples[np.random.randint(len(samples), size=10)]
    print (rand_data)
    print (rand_data[0])
    print (rand_data[1])
    print (np.exp(rand_data[0][3]))
    jump = 4
    y = np.linspace(10.0,11.2,10)
    print (y)
    fig, ax = plt.subplots(nrows = len(y) - jump, ncols = 2, squeeze=False, figsize=(12,48))
    data = np.zeros((len(y) - jump, 14))
    for idx in range (0, len(y) - jump):
        # main sequence galaxies
        slice = GAMA_sf[GAMA_sf['logM*'] > y[idx]]
        slice = slice[slice['logM*'] <= y[idx + jump]]
        sliceb = GAMA[GAMA['logM*'] > y[idx]]
        sliceb = sliceb[sliceb['logM*'] <= y[idx + jump]]
        n, bins, patches = ax[idx,0].hist(sliceb['logSFR'], bins=50, normed=1, color ='r', alpha = 0.3)
        # n, bins, patches = ax[idx,0].hist(slice['logSFR'], bins=50, normed=1, color ='b', alpha = 0.3)
        centers = 0.5*(bins[1:]+ bins[:-1])
        shapen, locn, scalen = stats.genlogistic.fit(slice['logSFR'])
        # params, pcov = curve_fit(generalised_logistic, centers, n)
        params, pcov = curve_fit(double_gauss, centers, n, p0 = [0.1,-1.75,0.4,.8,0.0,0.3])
        # params3, pcov3 = curve_fit(generalised_logistic2, centers, n, p0 = [params[1], params[2]])
        data[idx,0], data[idx,1], data[idx,2], data[idx,3] = (y[idx]+y[idx+jump])/2, shapen, locn, scalen
        data[idx,8], data[idx,9], data[idx,10] = params[0], params[1], params[2]
        x2 = np.linspace(-3.0, 1.5, num=400)
        # ax[idx,0].plot(x2, stats.genlogistic.pdf(x2, shapen, loc=locn, scale=scalen), 'b', linestyle = '-', linewidth = 3)
        # ax[idx,0].plot(x2, generalised_logistic(x2, params[0], params[1], params[2]), 'b', linestyle = ':', linewidth = 5)
        ax[idx,0].plot(x2, double_gauss(x2, params[0], params[1], params[2], params[3], params[4], params[5]), 'b', linestyle = ':', linewidth = 5)
        mstar = (y[idx] + y[idx + jump])/2
        for j, p in enumerate(rand_data):
            a1, a2, a3, sigma1, A, d1, d2, lnf4 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
            ax[idx,0].plot(x2, generalised_logistic(x2, A, sigma1, a1*mstar*mstar + a2*mstar + a3), lw=1, color="c", alpha = 0.1)
            ax[idx,0].axvline(a1*mstar*mstar + a2*mstar + a3, color = 'c', linewidth = .1, alpha =.4)
        # ax[idx,0].plot(x2, generalised_logistic2(x2, params3[0], params3[1]), 'g', linewidth = 5)
        ax[idx,0].set_xlim(-3,1.5)
        ax[idx,0].set_title(str(round(y[idx],2)) + ' < M* < ' + str(round(y[idx + jump],2)))
        # passive galaxies
        if y[idx] >= 9.0:
            slice2 = GAMA_pass[GAMA_pass['logM*'] > y[idx]]
            slice2 = slice2[slice2['logM*'] <= y[idx + jump]]
            n, bins, patches = ax[idx,1].hist(slice2['logSFR'], bins=50, normed=1, color ='r', alpha = 0.3)
            centers = 0.5*(bins[1:]+ bins[:-1])
            shapen, locn, scalen = stats.genlogistic.fit(slice2['logSFR'])
            params, pcov = curve_fit(generalised_logistic, centers, n)
            data[idx,4], data[idx,5], data[idx,6] = shapen, locn, scalen
            data[idx,11], data[idx,12], data[idx,13] = params[0], params[1], params[2]
            x2 = np.linspace(-4.0, 0.5, num=400)
            ax[idx,1].plot(x2, stats.genlogistic.pdf(x2, shapen, loc=locn, scale=scalen), 'r', linestyle = '-', linewidth=3)
            ax[idx,1].plot(x2, generalised_logistic(x2, params[0], params[1], params[2]), 'r', linestyle = ':', linewidth = 5)
            ax[idx,1].set_xlim(-4,.5)
            ax[idx,1].set_title(str(round(y[idx],2)) + ' < M* < ' + str(round(y[idx + jump],2)))
    data[:,7] = data[:,3]/data[:,6]
    # fig, ax = plt.subplots(nrows = 1, ncols = 2, squeeze=False, figsize=(18,6))
    # n, bins, patches = ax[0,0].hist(slice1['SFR'], bins=50, normed=1, color ='b', alpha = 0.3) # Plot histogram
    # n, bins, patches = ax[0,1].hist(slice1['logSFR'], bins=50, normed=1, color ='b', alpha = 0.3) # Plot histogram
    #
    #
    # n, bins, patches = ax[1,0].hist(slice3['SFR'], bins=50, normed=1, color ='b', alpha = 0.3) # Plot histogram
    # n, bins, patches = ax[1,1].hist(slice3['logSFR'], bins=50, normed=1, color ='b', alpha = 0.3) # Plot histogram

    # n2, bins2, patches2 = ax[0,1].hist(slice3['SFR'], bins=50, normed=1, color ='g', alpha = 0.3)
    # GAMA_sf['logSFR_flipped'] = (3.0-GAMA_sf['logSFR'])+3.0
    #
    # slice1 = GAMA_sf[GAMA_sf['logM*'] > 9.3]
    # slice1 = slice1[slice1['logM*'] < 9.7]
    # print (min(slice1['logSFR_flipped']))
    # slice1['logSFR_flipped2'] = slice1['logSFR_flipped'] - min(slice1['logSFR_flipped'] + 0.1)
    # print(slice1['logSFR_flipped2'])
    # # rvs = stats.lognorm.rvs(np.log(2), loc=0, scale=4, size=50000) # Generate some random variates as data
    # # n, bins, patches = plt.hist(rvs, bins=100, normed=True) # Plot histogram
    # n, bins, patches = plt.hist(slice1['logSFR_flipped2'], bins=50) # Plot histogram
    #
    #
    #
    # shapes = []
    # x = np.linspace(8,11,10)
    # for i in range (0,len(x)-1):
    #     slice = GAMA_pass[GAMA_pass['logM*'] > x[i]]
    #     slice = slice[slice['logM*'] <= x[i+1]]
    #     shape, loc, scale = stats.lognorm.fit(slice['SFR'], loc=0)
    #     shapes.append(shape)
    #     print (shape)

    # shape, loc, scale = stats.lognorm.fit(slice1['SFR'], loc=0) # Fit a curve to the variates
    # shapen, locn, scalen = stats.genlogistic.fit(slice1['logSFR']) # Fit a curve to the variates
    # # shape2, loc2, scale2 = stats.lognorm.fit(slice3['SFR'], loc=0)
    #
    #
    # shape10, loc10, scale10 = stats.lognorm.fit(slice3['SFR']) # Fit a curve to the variates
    # shape10n, loc10n, scale10n = stats.genlogistic.fit(slice3['logSFR']) # Fit a curve to the variates


    # print (shape, loc, scale)
    # print (shape2, loc2, scale2)
    # mu = np.log(scale) # Mean of log(X)
    # sigma = shape # Standard deviation of log(X)
    # M = np.exp(mu) # Geometric mean == median
    # s = np.exp(sigma) # Geometric standard deviation
    #
    # mu2 = np.log(scale2) # Mean of log(X)
    # sigma2 = shape2 # Standard deviation of log(X)
    # M2 = np.exp(mu2) # Geometric mean == median
    # s2 = np.exp(sigma2) # Geometric standard deviation
    #
    # print (shape, loc, scale)
    #
    # # Plot figure of results
    # x = np.linspace(rvs.min(), rvs.max(), num=400)
    # x = np.linspace(slice1['SFR'].min(), slice1['SFR'].max(), num=400)
    # x2 = np.linspace(slice3['SFR'].min(), slice3['SFR'].max(), num=400)
    # x3 = np.linspace(slice1['logSFR'].min(), slice1['logSFR'].max(), num=400)
    # print (slice3['SFR'].min(), slice3['SFR'].max())
    # x4 = np.linspace(slice3['SFR'].min(), slice3['SFR'].max(), num=400)
    # x5 = np.linspace(slice3['logSFR'].min(), slice3['logSFR'].max(), num=400)

    # ax[0,0].plot(x, stats.lognorm.pdf(x, shape, loc=loc, scale=scale), 'b', linewidth=3) # Plot fitted curve
    # # ax[0,1].plot(x3, stats.genlogistic.pdf(x3, 0.75, loc=-0.5, scale=0.2), 'b', linewidth=3) # Plot fitted curve
    # ax[0,1].plot(x3, stats.genlogistic.pdf(x3, shapen, loc=locn, scale=scalen), 'k', linewidth=3) # Plot fitted curve
    #
    # # ax[0,1].plot(x2, stats.lognorm.pdf(x2, shape2, loc=loc2, scale=scale2), 'g', linewidth=3)
    # ax[0,0].plot(x, stats.lognorm.pdf(x, shape, loc=0, scale=scale), 'b', linewidth=3) # Plot fitted curve
    # ax[0,1].plot(x2, stats.lognorm.pdf(x2, shape2, loc=0, scale=scale2), 'g', linewidth=3)
    # ax[0,1].plot(x2, stats.lognorm.pdf(x2, 0.5, loc=0, scale=scale2), 'g', linewidth=3)
    # ax[0,1].plot(x2, stats.lognorm.pdf(x2, shape2, loc=loc2, scale=scale2), 'g', linewidth=3)
    # ax = plt.gca() # Get axis handle for text positioning
    # ax[0,0].text(0.9, 0.9, 'M = %.4f\ns = %.4f' % (M, s), horizontalalignment='right',
    #                 size='large', verticalalignment='top', transform=ax.transAxes)
    # ax[0,1].text(0.9, 0.9, 'M = %.4f\ns = %.4f' % (M2, s2), horizontalalignment='right',
    #                 size='large', verticalalignment='top', transform=ax.transAxes)
    # ax[0,1].set_xlim(0,0.04)


    # ax[1,0].plot(x4, stats.lognorm.pdf(x4, shape10, loc=loc10, scale=scale10), 'b', linewidth=3) # Plot fitted curve
    # ax[1,1].plot(x5, stats.genlogistic.pdf(x5, shape10n, loc=loc10n, scale=scale10n), 'k', linewidth=3) # Plot
    #
    # ax[0,0].set_xlabel(r"$\mathrm{SFR}$")
    # ax[0,1].set_xlabel(r"$\mathrm{SFR}$")
    # ax[0,0].set_ylabel(r"$\mathrm{N}$")
    # ax[0,1].set_ylabel(r"$\mathrm{N}$")
    # ax[0,0].set_title('Star-forming Galaxies')
    # ax[0,1].set_title('Passive Galaxies')
    # plt.title('Scatter at M*~9.5')
    plt.tight_layout()
    plt.savefig('img/lognormal.png')
    return data

def plot_trends(data):
    converged_old = np.loadtxt('data/converged.txt', delimiter = ',')
    converged_new = np.loadtxt('data/logistic_fit.txt')
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    # x0 = np.linspace(8,11,200)
    # ax[0,0].plot(data[:-3,0], data[:-3,2], color = 'b', linestyle = '-', label = 'Binned, scipy generalised_logistic')
    # ax[0,0].plot(data[:-3,0], data[:-3,10], color = 'b', linestyle = ':', label = 'Binned, curve_fit generalised_logistic')
    # # params, pcov = curve_fit(third_order, data[:-3,0], data[:-3,10])
    # # params2, pcov2 = curve_fit(third_order, data[:-3,0], data[:-3,2])
    # ax[0,0].plot(data[10:,0], data[10:,5], color = 'r', linestyle = '-', label = 'Binned, scipy generalised_logistic')
    # ax[0,0].plot(data[10:,0], data[10:,13], color = 'r', linestyle = ':', label = 'Binned, curve_fit generalised_logistic')
    # for a1, a2, a3, lnf, A, d1, d2, lnf3 in converged_new[np.random.randint(len(converged_new), size=100)]:
    #     ax[0,0].plot(x0, a1*x0*x0 + a2*x0 + a3, lw=.1, alpha=0.1, color="cyan")
    # for alpha, beta, zeta, a1, a2, a3, s1, b1, b2, s2, c1, c2, lnf3, d1, d2, lnf4  in converged_old[np.random.randint(len(converged_old), size=100)]:
    #     ax[0,0].plot(x0, a1*x0*x0 + a2*x0 + a3, lw=.1, alpha=0.1, color="navy")
    # ax[0,0].plot(data[:,0], data[:,1], label = 'A scipy')
    # ax[0,0].plot(data[:,0], data[:,8], label = 'A curve fit')
    # ax[0,0].plot(data[:,0], data[:,3], label = 'sigma scipy')
    ax[0,0].plot(data[:,0], np.log(data[:,10]), label = 'sigma curve fit')
    # popt, pcov = curve_fit(second_order, data[:,0], data[:,10])
    # print (popt)
    # x0 = np.linspace(8.0, 11.0, 200)
    # ax[0,0].plot(x0, second_order(x0, *popt))
    # x = np.linspace(8,11,200)
    # y = (-0.06*x*x)+ (1.81*x) -11.81
    # ax[0,0].plot(x, y)
    # ax[0,0].plot(x, third_order(x, *params))
    # ax[0,0].plot(x, third_order(x, *params2))
        # ax[0,0].plot(data[:,0], data[:,3]/data[:,6])
    # ax[0,0].set_ylim(0,.5)
    ax[0,0].set_xlabel(r'$\mathrm{log\,M_{*}}$')
    ax[0,0].set_ylabel(r'$\mathrm{log\,SFR}$')
    plt.legend()
    plt.savefig('img/scales.pdf')


GAMA, GAMAb, GAMAr = read_GAMA()
GAMA_pass = GAMA[GAMA['logSFR'] < (0.83*GAMA['logM*']) + -9.5]
GAMA_sf = GAMA[GAMA['logSFR'] >= (0.83*GAMA['logM*']) + -9.5]

GAMA_sf['SFR'] = np.power(10,GAMA_sf['logSFR'])
GAMA_sf['SFRerr'] = GAMA_sf['logSFRerr'] * GAMA_sf['SFR'] * np.log(10)
GAMA_sf['M*'] = np.power(10,GAMA_sf['logM*'])
GAMA_sf['M*err'] = GAMA_sf['logM*err'] * GAMA_sf['M*'] * np.log(10)
# print (GAMA_sf)
# slice1 = GAMA_sf[GAMA_sf['logM*'] > 9.3]
# slice1 = slice1[slice1['logM*'] < 9.7]
#
# slice3 = GAMA_sf[GAMA_sf['logM*'] > 8.8]
# slice3 = slice3[slice3['logM*'] < 9.2]

# GAMA_pass['SFR'] = np.power(10,GAMA_pass['logSFR'])
# slice2 = GAMA_pass[GAMA_pass['logM*'] > 9.3]
# slice2 = slice2[slice2['logM*'] < 9.7]

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
# calculate error matrices etc
S1 = S_error(det['SFRerr_best'].values, [0.2])
S2 = S_error(nondet['SFRerr_best'].values, [0.14])
x1, y1 = det['SFR_best'].values, det['lgMHI'].values
x2, y2 = nondet['SFR_best'].values, nondet['lgMHI'].values

# read in the ALFA ALFA datasets from the 40% paper
ALFAALFA = pd.read_csv('ALFAALFA.csv', comment = '#', header = None, sep=",")
ALFAALFA.columns = ['x', 'y', 'dy', 'MHI', 'phi', 'err', 'phi_err']
ALFAALFA = ALFAALFA[np.isfinite(ALFAALFA['phi_err'])]
MHI, phi_alfa, phi_err_alfa = np.round(ALFAALFA['MHI'].values,2), ALFAALFA['phi'].values, ALFAALFA['phi_err'].values

global GASS_data
global COLD_GASS_data
global slice_data
global slide_data2
global ALFALFA_data
ALFALFA_data = MHI, phi_alfa, phi_err_alfa
slice_data = GAMA_sf
slice_data2 = GAMA_pass
COLD_GASS_data = CGx1, CGx2, CGy1, CGy2, CGS1, CGS2
GASS_data = x1, x2, y1, y2, S1, S2

# plt.show()

# lognormalslice_scatter()



ndim, nwalkers = 8, 100
g = [-0.06, 2.0, -11.80, 0.5, 0.2, 0.87, 9.47, -0.75]
pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
pool = Pool(2)
sampler3 = emcee.EnsembleSampler(nwalkers, ndim, log_marg_mainsequence_full, pool = pool)
# sampler3.run_mcmc(pos, 2000, progress=True)
# samples3 = sampler3.chain[:, 1500:, :].reshape((-1, ndim))
# np.savetxt('data/logistic_fit.txt', samples3)
samples3 = np.loadtxt('data/logistic_fit.txt')
# samples3[:,3] = np.exp(samples3[:,3])
# samples3[:,7] = np.exp(samples3[:,7])
# print (samples3)
data = sfrm_evolution(GAMA_sf, GAMA_pass, GAMA, samples3)
# plot_trends(data)
plot_samples(sampler3, ndim, 'test_lognormal')
# samples3 = sampler3.chain[:, 250:, :].reshape((-1, ndim))
plot_SFR_M_plane2(GAMA, GAMAb, GAMAr, GAMA_sf, GAMA_pass, samples3)
plot_corner(samples3, 'lognormal')

# plot the gama data and the first to passive and star forming galaxies
# plot_SFR_M_plane2(GAMA, GAMAb, GAMAr, GAMA_sf, GAMA_pass, samples3)
# calculate the MHI mass function
MHI_hist2(20, 20, samples3, 'img/MHI_hist_passive.pdf')
# print (sampler3)
