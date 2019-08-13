from IPython.display import display, Math
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import math
import numpy as np
import pandas as pd
import emcee
from scipy.integrate import quad, dblquad, nquad
from scipy import special
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import corner
from multiprocessing import Pool
import os
import nose

# import COLD GASS functions
import schechter
import models
import integrands
import plots
import read_files
import likelihoods

os.environ["OMP_NUM_THREADS"] = "1"
pd.options.mode.chained_assignment = None  # default='warn'
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

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

def bootstrap_GAMA(GAMAb, GAMAr, frac, n):
    blue = np.copy(GAMAb['logM*'].values)
    red = np.copy(GAMAr['logM*'].values)
    x1 = np.linspace(6.0,11.3,40)
    data = np.zeros((n, len(x1) + 7))
    # blue = GAMAb['logM*'].values
    # red = GAMAr['logM*'].values
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
    return xnew, std, ratio

def sfr_hist_only(samples1, samples2, min_mass, max_mass, gsmf_params, fname):
    SFR = np.linspace(-4,3,40)
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6, 6))
    sfr_hist_data = pd.read_csv('sfr_hist.csv')
    counter = 0
    for params in samples1[np.random.randint(len(samples1), size=10)]:
        b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
        phi, phib, phir, phi_test = [], [], [], []
        phi_saintonge = []
        for idx, element in enumerate(SFR):
            integration_t = quad(integrands.integrand_SFR1b, min_mass, max_mass,
                                args=(element, params, gsmf_params),
                                epsrel = 1e-8, epsabs = 0)
            integration_b = quad(integrands.integrand_SFR_blue1, min_mass, max_mass,
                                args=(element, params, gsmf_params),
                                epsrel = 1e-8, epsabs = 0)
            integration_r = quad(integrands.integrand_SFR_red1, min_mass, max_mass,
                                args=(element, params, gsmf_params),
                                epsrel = 1e-8, epsabs = 0)
            # integration_Saintonge = quad(integrands.integrand_SFR_Saintonge16, min_mass, max_mass,
            #                     args=(element, params, gsmf_params),
            #                     epsrel = 1e-8, epsabs = 0)
            phi.append(integration_t[0])
            phib.append(integration_b[0])
            phir.append(integration_r[0])
            # phi_saintonge.append(integration_Saintonge[0])
        counter += 1
        if counter == 9:
            ax[0,0].plot(SFR, np.log10(phi), color = 'k', alpha = 0.05, label = 'Total')
            ax[0,0].plot(SFR, np.log10(phib), color = 'b', alpha = 0.05, label = 'Blue only')
            ax[0,0].plot(SFR, np.log10(phir), color = 'r', alpha = 0.05, label = 'Red only')
            # ax[0,0].plot(SFR, np.log10(phi_saintonge), color = 'orange', alpha = 0.05, label = 'Saintonge+16')
            # ax[0,1].plot(SFR, phi*(SFR[1]-SFR[0])*10000, color = 'k', alpha = 0.05, label = 'Total')
            # ax[0,1].plot(SFR, phib*(SFR[1]-SFR[0])*10000, color = 'b', alpha = 0.05, label = 'Blue only')
            # ax[0,1].plot(SFR, phir*(SFR[1]-SFR[0])*10000, color = 'r', alpha = 0.05, label = 'Red only')
            # ax[0,0].plot(SFR, np.log10(phi_test), color = 'g', alpha = 0.05, label = 'test')
        else:
            ax[0,0].plot(SFR, np.log10(phi), color = 'k', alpha = 0.05)
            ax[0,0].plot(SFR, np.log10(phib), color = 'b', alpha = 0.05)
            ax[0,0].plot(SFR, np.log10(phir), color = 'r', alpha = 0.05)
            # ax[0,0].plot(SFR, np.log10(phi_saintonge), color = 'orange', alpha = 0.05)
            # ax[0,1].plot(SFR, np.array(phi)*(SFR[1]-SFR[0])*10000, color = 'k', alpha = 0.05)
            # ax[0,1].plot(SFR, np.array(phib)*(SFR[1]-SFR[0])*10000, color = 'b', alpha = 0.05)
            # ax[0,1].plot(SFR, np.array(phir)*(SFR[1]-SFR[0])*10000, color = 'r', alpha = 0.05)
    # ax[0,1].text(0,0, str(np.sum(np.array(phi)*(SFR[1]-SFR[0])*10000)))
            # ax[0,0].plot(SFR, np.log10(phi_test), color = 'g', alpha = 0.05)
    # for params in samples2[np.random.randint(len(samples2), size=100)]:
    #     print (params)
    #     b1, b2, b3, lnb, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    #     phi, phib, phir = [], [], []
    #     for idx, element in enumerate(SFR):
    #         phi.append(quad(integrands.integrand_SFR2, min_mass, max_mass, args=(element, params, gsmf_params))[0])
    #         phib.append(quad(integrands.integrand_SFR_blue2, min_mass, max_mass, args=(element, params, gsmf_params))[0])
    #         phir.append(quad(integrands.integrand_SFR_red2, min_mass, max_mass, args=(element, params, gsmf_params))[0])
    #     ax[0,0].plot(SFR, np.log10(phi), color = 'k', alpha = 0.05)
    #     ax[0,0].plot(SFR, np.log10(phib), color = 'b', alpha = 0.05)
    #     ax[0,0].plot(SFR, np.log10(phir), color = 'r', alpha = 0.05)
    phi_SFR = np.log(10) * np.exp(-np.power(10,SFR-np.log10(9.2))) * (0.0015*np.power(10,(-1.51+1)*(SFR-np.log10(9.2))))
    phi_SFR2 = np.log(10) * np.exp(-np.power(10,SFR-np.log10(9.0))) * (0.0015*np.power(10,(-1.48+1)*(SFR-np.log10(9.0))))
    dbin = sfr_hist_data['sfr_3dhst'][1] - sfr_hist_data['sfr_3dhst'][0]
    ax[0,0].plot(SFR, np.log10(phi_SFR), linestyle = '-', linewidth = 1, color ='b', label = 'Kennicutt maximum likelihood')
    ax[0,0].plot(SFR, np.log10(phi_SFR2), linestyle = '-', linewidth = 1, color ='k', label = 'Kennicutt least-squares')
    ax[0,0].scatter(sfr_hist_data['sfr_3dhst'], sfr_hist_data['phi_3dhst'], color = 'g', label ='3D-HST')
    ax[0,0].scatter(sfr_hist_data['sf_gama'], sfr_hist_data['phi_gama'], color = 'r', label = 'GAMA')
    ax[0,0].scatter(sfr_hist_data['sf_gama'], sfr_hist_data['phi_cosmos'], color = 'b', label = 'COSMOS')
    ax[0,0].set_ylim(-10,0)
    ax[0,0].set_xlim(-4,3)
    ax[0,0].set_xlabel(r"$\mathrm{\log_{10} SFR \, [M_{\odot} \, yr^{-1}]}$")
    ax[0,0].set_ylabel(r"$\mathrm{\log_{10} \phi(SFR) \, [Mpc^{-3} \, dex^{-1}]}$")
    plt.legend()
    plt.savefig('img/' + fname + '.pdf')

def MHI_mf_only(samples1, samples2, min_sfr, max_sfr, min_mass, max_mass, gsmf_params):
    # read in the ALFA ALFA datasets from the 40% paper
    ALFAALFA = pd.read_csv('ALFAALFA.csv', comment = '#', header = None, sep=",")
    ALFAALFA.columns = ['x', 'y', 'dy', 'MHI', 'phi', 'err', 'phi_err']
    ALFAALFA = ALFAALFA[np.isfinite(ALFAALFA['phi_err'])]
    MHI_alfa, phi_alfa, phi_err_alfa = np.round(ALFAALFA['MHI'].values,2), ALFAALFA['phi'].values, ALFAALFA['phi_err'].values

    MHI = np.linspace(6,12,30)
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6, 6))
    ax[0,0].errorbar(MHI_alfa, phi_alfa, yerr = phi_err_alfa, fmt='o', capsize = 2, markersize = 3, linewidth=2, markeredgewidth=2, capthick=2, mfc='gray', mec='gray', ecolor = 'gray', label = 'ALFAALFA')
    counter = 0
    for params in samples1[np.random.randint(len(samples1), size=10)]:
        print (counter, params)
        b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
        phi, phib, phir = [], [], []
        for idx, element in enumerate(MHI):
            phi.append(dblquad(integrands.integrand_MHI_total, min_sfr, max_sfr,
                                lambda SFR: min_mass, lambda SFR: max_mass,
                                args=(element, params, gsmf_params),
                                epsrel = 1e-8, epsabs = 0)[0])
            phib.append(dblquad(integrands.integrand_MHI_blue, min_sfr, max_sfr,
                                lambda SFR: min_mass, lambda SFR: max_mass,
                                args=(element, params, gsmf_params),
                                epsrel = 1e-8, epsabs = 0)[0])
            phir.append(dblquad(integrands.integrand_MHI_red, min_sfr, max_sfr,
                                lambda SFR: min_mass, lambda SFR: max_mass,
                                args=(element, params, gsmf_params),
                                epsrel = 1e-8, epsabs = 0)[0])
        counter += 1
        if counter == 9:
            ax[0,0].plot(MHI, np.log10(phi), color = 'k', alpha = 0.05, label = 'Total')
            ax[0,0].plot(MHI, np.log10(phib), color = 'b', alpha = 0.05, label = 'Blue only')
            ax[0,0].plot(MHI, np.log10(phir), color = 'r', alpha = 0.05, label = 'Red only')
        else:
            ax[0,0].plot(MHI, np.log10(phi), color = 'k', alpha = 0.05)
            ax[0,0].plot(MHI, np.log10(phib), color = 'b', alpha = 0.05)
            ax[0,0].plot(MHI, np.log10(phir), color = 'r', alpha = 0.05)

    # phi_SFR = np.log(10) * np.exp(-np.power(10,SFR-np.log10(9.2))) * (0.00016*np.power(10,(-1.51+1)*(SFR-np.log10(9.2))))
    # ax[0,0].plot(SFR, np.log10(phi_SFR), linestyle = '--', linewidth = 1, color ='k', label = 'Kennicutt')
    # ax[0,0].scatter(sfr_hist_data['sfr_3dhst'], sfr_hist_data['phi_3dhst'], color = 'g', label ='3D-HST')
    # ax[0,0].scatter(sfr_hist_data['sf_gama'], sfr_hist_data['phi_gama'], color = 'r', label = 'GAMA')
    # ax[0,0].scatter(sfr_hist_data['sf_gama'], sfr_hist_data['phi_cosmos'], color = 'b', label = 'COSMOS')
    ax[0,0].set_ylim(-6,0)
    ax[0,0].set_xlim(6,12)
    ax[0,0].set_xlabel(r"$\mathrm{\log_{10} MHI \, [M_{\odot}]}$")
    ax[0,0].set_ylabel(r"$\mathrm{\log_{10} \phi(MHI) \, [Mpc^{-3} \, dex^{-1}]}$")
    plt.legend()
    plt.savefig('img/MHI_hist_dbl_only.pdf')


def sfr_histogram(GAMA, samples4, samples5, M, phi_Baldry):
    sfr_hist_data = pd.read_csv('sfr_hist.csv')
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
            phi.append(quad(integrands.integrand_SFR, 0, 12, args=(element, *params))[0])
            phib.append(quad(integrands.integrand_SFR_blue, 6, 12, args=(element, *params))[0])
            phir.append(quad(integrands.integrand_SFR_red, 0, 12, args=(element, *params))[0])
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
        ax[0,1].scatter(sfr_hist_data['sfr_3dhst'], sfr_hist_data['phi_3dhst'], color = 'g')
        ax[0,1].scatter(sfr_hist_data['sf_gama'], sfr_hist_data['phi_gama'], color = 'r')
        ax[0,1].scatter(sfr_hist_data['sf_gama'], sfr_hist_data['phi_cosmos'], color = 'b')


        # mmin.append(models.second_order(6, b1, b2, b3))
        # mmax.append(models.second_order(12, b1, b2, b3))
        # make the means, passive fractions for this set of params
        GAMA['b_mean'] = (b1*GAMA['logM*']*GAMA['logM*']) + (b2*GAMA['logM*']) + b3
        GAMA['r_mean'] = (r1*GAMA['logM*']) + r2
        GAMA['f_pass1'] = models.f_passive(GAMA['logM*'], alpha, beta, zeta)
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
                oldarea = 1317162.8627450706
                newarea = 828981.03
                n2 = n2/((sfr[1] - sfr[0])*newarea)
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
            best_fits[i,idx] = dblquad(integrands.integrand_MHI_blue, -8.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, *params))[0]

            best_fits_1[1,idx] = dblquad(integrands.integrand_MHI_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1+0.1, h2+0.1, np.log(0.1)))[0]
            best_fits_2[1,idx] = dblquad(integrands.integrand_MHI_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1+0.1, h2+0.1, np.log(0.2)))[0]
            best_fits_3[1,idx] = dblquad(integrands.integrand_MHI_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1+0.1, h2+0.1, np.log(0.3)))[0]
            best_fits_4[1,idx] = dblquad(integrands.integrand_MHI_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1+0.1, h2+0.1, np.log(0.4)))[0]
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
    # y3 = log_schechter_true(MHI, 4.8E-3, 9.96, -1.33)
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

    xxGASS, det, nondet = read_files.read_GASS(True)
    ax[2,0].scatter(xxGASS['lgMstar'], xxGASS['lgMHI'], s = 3)
    # for params in samples4[np.random.randint(len(samples4), size=10)]:
    #     b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    #     xxGASS['b_mean'] = (b1*xxGASS['lgMstar']*xxGASS['lgMstar']) + (b2*xxGASS['lgMstar']) + b3
    #     xxGASS['r_mean'] = (r1*xxGASS['lgMstar']) + r2
    #     xxGASS['f_pass1'] = models.f_passive(xxGASS['lgMstar'], alpha, beta, zeta)
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
    HI_labels = ['h1', 'h2', 'lnh1']
    pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    sampler5 = emcee.EnsembleSampler(nwalkers, ndim, likelihoods.MHI_Mstar_fit, pool = pool)
    sampler5.run_mcmc(pos, 1500, progress=True)
    plots.plot_samples_full(sampler5, ndim, 'MHI_scaling', HI_labels)
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
            phi.append(quad(integrands.integrand_MHI_direct, 0, 12, args=(element, h1, h2, -10))[0])
        ax[0,1].plot(MHI, np.log10(phi), color = 'g', alpha = 0.1)
    ax[0,0].set_xlabel(r"$\mathrm{log \, M_{*}}$")
    ax[0,1].set_xlabel(r"$\mathrm{log \, M_{HI}}$")
    ax[0,0].set_ylabel(r"$\mathrm{log \, M_{HI}}$")
    ax[0,1].set_ylabel(r"$\mathrm{log \, \phi}$")
    ax[0,1].set_ylim(-10,0)
    plt.savefig('img/atomic_gas_fraction.pdf')

# nose.run(argv=[__file__, 'nose_tests.py'])

popts = pd.read_csv('bestfits.csv')
mstars = np.linspace(7.6,12.0,45)

bins = np.linspace(-3.5,1.5,51)
sfr_bins = np.linspace(-3.0, 0.6, 19)
GAMA, GAMAb, GAMAr = read_files.read_GAMA()
# GAMA = GAMA.drop(GAMA[(GAMA['logSFR'] <  -0.06*GAMA['logM*']*GAMA['logM*'] + 1.95*GAMA['logM*'] -14.5) & (GAMA ['logM*']<9.0)].index)
xxGASS, det, nondet = read_files.read_GASS(True)
# calculate error matrices etc
S1 = S_error(det['SFRerr_best'].values, [0.2])
S2 = S_error(nondet['SFRerr_best'].values, [0.2])
x1, y1 = det['SFR_best'].values, det['lgMHI'].values
x2, y2 = nondet['SFR_best'].values, nondet['lgMHI'].values

GAMA_pass = GAMA[GAMA['logSFR'] < models.second_order(GAMA['logM*'], -0.06, +1.95, -14.5)]
GAMA_sf = GAMA[GAMA['logSFR'] >= models.second_order(GAMA['logM*'], -0.06, +1.95, -14.5)]
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
# GAMA_data = GAMA['logM*'].values, GAMA['logSFR'].values, GAMA['logM*err'].values, GAMA['logSFRerr'].values
GAMA_data = GAMA['logM*'].values, GAMA['logSFR'].values, np.zeros(len(GAMA)), np.zeros(len(GAMA))

GAMA_passive = GAMA_pass['logM*'], GAMA_pass['logSFR'], GAMA_pass['logM*err'], GAMA_pass['logSFRerr']
GAMA_sforming = GAMA_sf['logM*'], GAMA_sf['logSFR'], GAMA_sf['logM*err'], GAMA_sf['logSFRerr']
passive_data = xnew, ratio, std
sfr_hist_data = sfr, n

xxGASS, det, nondet = read_files.read_GASS(True)
xxGASS_final = fits.open('data/xGASS_RS_final_Serr_180903.fits')
xxGASS_final = Table(xxGASS_final[1].data).to_pandas()
xxGASS['SNR'] = xxGASS_final['SNR']
xxGASS['MHI_err'] = np.power(10, xxGASS['lgMHI'])/xxGASS['SNR']
xxGASS['lgMHI_err'] = xxGASS['MHI_err']/(np.power(10,xxGASS['lgMHI'])*np.log(10))
xxGASS['lgMstar_err'] = 0.0
det = xxGASS[xxGASS['HIconf_flag']==0]
nondet = xxGASS[xxGASS['HIconf_flag']==-99]
# det = xxGASS[xxGASS['SNR']>0]
global GASS_data2
global GASS_data3
S1 = S_error(det['lgMHI_err'].values, [0.0])
S2 = S_error(nondet['lgMHI_err'].values, [0.0])

GASS_data2 = det['lgMstar'], nondet['lgMstar'], det['lgMHI'], nondet['lgMHI'], S1, S2

# minmass = min(GAMA['logM*'])
# maxmass = max(GAMA['logM*'])
# v = max(GAMA['logM*'])
# masses = np.linspace(minmass, maxmass, 30)
# means = []
# values = []
# for idx in range(0,len(masses)-1):
#     GAMA_bins = GAMA[GAMA['logM*'] > masses[idx]]
#     GAMA_bins = GAMA_bins[GAMA_bins['logM*'] <= masses[idx+1]]
#     values.append(len(GAMA_bins))
# masses_means = (masses[1:] + masses[:-1])/2
# popt, pcov = curve_fit(models.second_order, masses_means, np.log10(values))
# popt2, pcov2 = curve_fit(models.third_order, masses_means, np.log10(values))
# popt3, pcov3 = curve_fit(models.fourth_order, masses_means, np.log10(values))
# print (popt3)
# # plt.plot(sfrs, models.second_order(sfrs, *popt), color = 'g')
# plt.plot(masses, models.third_order(masses, *popt2), color = 'r')
# plt.plot(masses, models.fourth_order(masses, *popt3), color = 'b')
# plt.plot(masses_means, np.log10(values))
# plt.show()
# jump = 4
# data = np.zeros((len(mstars) - jump, 7))
# data2 = np.zeros((len(mstars) - jump, 7))
# fig, ax = plt.subplots(nrows = len(mstars) - jump, ncols = 2, squeeze=False, figsize=(12,35), sharex=True)
# offset = np.linspace(0.5,2.0,len(mstars) - jump)
# ms = ((mstars[4:] + mstars[:-4])/2)
# print (ms)
# passive_ratio = (0.02030656*ms*ms*ms) - (0.3111481*ms*ms) + (0.30672944*ms) + 7.95966901
# for idx, row in popts.iterrows():
#     print (idx, mstars[idx], mstars[idx + jump])
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
#         popt, pcov = curve_fit(models.gauss, bins2, n, p0 = [row['B1'], row['Bmean'], row['Bsigma']], maxfev=5000)
#         # ax[idx,1].plot(x, models.double_gauss(x, *popt), color = 'k')
#         ax[idx,1].plot(x, models.gauss(x, *popt[:3]), color = 'b')
#         ax[idx,1].plot(x, models.gauss(x, popt[0], (poptb[0]*m*m) + (poptb[1]*m) + poptb[2], .4), color = 'b', alpha = 0.2, linewidth = 3)
#         data[idx,0], data[idx,1], data[idx,2] = popt[0], popt[1], popt[2]
#         # ax[idx,1].plot(x, models.gauss(x, *popt[3:]), color = 'r')
#     elif m > 11.2:
#         popt, pcov = curve_fit(models.gauss, bins2, n, p0 = [row['B1'], row['Bmean'], row['Bsigma']], maxfev=5000)
#         # ax[idx,1].plot(x, models.double_gauss(x, *popt), color = 'k')
#         ax[idx,1].plot(x, models.gauss(x, *popt[:3]), color = 'r', alpha = 0.2 , linewidth = 3)
#         # ax[idx,1].plot(x, models.gauss(x, popt[0], (poptb[0]*m*m) + (poptb[1]*m) + poptb[2], .4), color = 'r', alpha = 0.2, linewidth = 3)
#         data[idx,3], data[idx,4], data[idx,5] = popt[0], popt[1], popt[2]
#         data[idx,0], data[idx,1], data[idx,2] = 0, 0, 0
#         # ax[idx,1].plot(x, models.gauss(x, *popt[3:]), color = 'r')
#     else:
#         popt, pcov = curve_fit(models.double_gauss, bins2, n, p0 = [row['B1'], row['Bmean'], row['Bsigma'], row['R1'], row['Rmean'], row['Rsigma']], maxfev=5000)
#         # ax[idx,1].plot(x, models.double_gauss(x, *popt), color = 'k')
#         # ax[idx,1].plot(x, models.gauss(x, *popt[:3]), color = 'b')
#         # ax[idx,1].plot(x, models.gauss(x, *popt[3:]), color = 'r')
#         ax[idx,1].plot(x, models.gauss(x, popt[0], (poptb[0]*m*m) + (poptb[1]*m )+ poptb[2], .4), color = 'b', alpha = 0.2, linewidth = 3)
#         ax[idx,1].plot(x, models.gauss(x, popt[3], (poptr[0]*m*m) + (poptr[1]*m )+ poptr[2], .5), color = 'r', alpha = 0.2, linewidth = 3)
#         ax[idx,1].plot(x, models.double_gauss(x, popt[0], (poptb[0]*m*m) + (poptb[1]*m )+ poptb[2], .4, popt[3], (poptr[0]*m*m) + (poptr[1]*m )+ poptr[2], .5), color = 'k', alpha = 0.2, linewidth = 3)
#         data[idx,3], data[idx,4], data[idx,5] = popt[3], popt[4], popt[5]
#         data[idx,0], data[idx,1], data[idx,2] = popt[0], popt[1], popt[2]
#     # data2[idx,1:] = popt
#     # data2[idx,0] = ms[idx]
#
#     data[idx,6] = ms[idx]
#     # print (popt)
#     # popt2, pcov2 = curve_fit(models.triple_gauss, bins2, n, p0 = [200.0, -0.34, 0.2, 42.0, -2.1, 0.3, 30.0, -1.21, 0.3], maxfev=2000)
#     ax[idx,0].plot(x, models.double_gauss(x, *row[['B1', 'Bmean', 'Bsigma', 'R1', 'Rmean', 'Rsigma']].values), color = 'k')
#     ax[idx,0].plot(x, models.gauss(x, *row[['B1', 'Bmean', 'Bsigma']].values), color = 'b')
#     ax[idx,0].plot(x, models.gauss(x, *row[['R1', 'Rmean', 'Rsigma']].values), color = 'r')
#
#
#     ax[idx,0].axvline(sfr)
#     ax[idx,0].axvline(sfr-offset[idx])
#     # print (popt)
#     # popt[1] = popt[1] + 0.1
#     # popt[4] = popt[4] + 0.1
#     # print (popt)
#     # plt.plot(x, models.triple_gauss(x, *popt2))
#     # plt.plot(x, models.double_gauss(x, 200.0, -0.34, 0.2, 42.0, -2.1, 0.3))
#     # plt.plot(x, models.triple_gauss(x, 200.0, -0.34, 0.2, 42.0, -2.1, 0.3, 30.0, -1.21, 0.3))
# # np.savetxt('bestfits.csv', data2, delimiter = ',')
# plt.savefig('img/double_gaussian.pdf')

SDSS, SDSS2, SDSS_blue, SDSS_red = read_files.read_SDSS()
plots.asymmetry(SDSS_blue, SDSS_red)
Chiang = read_files.read_Chiang()

# pool = Pool(6)
# ndim, nwalkers = 10, 100
# param_labels = ['b1', 'b2', 'b3', 'lnb', 'r1', 'r2', 'lnr', 'alpha', 'beta', 'zeta']
# g = [-0.06, +1.8, -12.0, -0.9, 0.5, -7.0, -1.1, 10.6, -0.96, -4.2]
# pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
# print (len(SDSS))
# sampler5 = emcee.EnsembleSampler(   nwalkers, ndim,
#                                     likelihoods.log_marg_mainsequence_full_SDSS,
#                                     pool = pool,
#                                     args = (SDSS['mstellar_median'],
#                                     SDSS['sfr_tot_p50'],
#                                     SDSS['mstellar_err'],
#                                     SDSS['errs']))
# # sampler5 = emcee.EnsembleSampler(   nwalkers, ndim,
# #                                     likelihoods.log_marg_mainsequence_full_SDSS,
# #                                     pool = pool,
# #                                     args = (Chiang['lmass50_all'][:1000],
# #                                     Chiang['lsfr50_all'][:1000],
# #                                     Chiang['m_err'][:1000],
# #                                     Chiang['sfr_err'][:1000]))
# sampler5.run_mcmc(pos, 1000, progress=True)
# af = sampler5.acceptance_fraction
# print("Mean acceptance fraction:", np.mean(af))
# plots.plot_samples_full(sampler5, ndim, 'mainsequence_full_SDSS_full_sample', param_labels)
# samples5 = sampler5.chain[:, 800:, :].reshape((-1, ndim))
# samples5_copy = np.copy(samples5)
# np.savetxt('data/dbl_gauss_SDSS_full_sample.txt', samples5)
samples_SDSS = np.loadtxt('data/dbl_gauss_SDSS_full_sample.txt')
# plots.plot_corner_full(samples5_copy, 'dbl_gauss_SDSS_full_sample')


# g = [-0.06, +1.8, -12.0, -0.9, .64, -8.23, -1.1, 10.6, -0.96, -2.2, 0.8, 10.0, -1.1]
# pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
# sampler5 = emcee.EnsembleSampler(nwalkers, ndim, likelihoods.log_marg_mainsequence_full1, pool = pool)
# sampler5.run_mcmc(pos, 1000, progress=True)
# af = sampler5.acceptance_fraction
# print("Mean acceptance fraction:", np.mean(af))
# plots.plot_samples_full(sampler5, ndim, 'mainsequence_full1', param_labels)
# samples5 = sampler5.chain[:, 800:, :].reshape((-1, ndim))
# np.savetxt('data/dbl_gauss_straight_line.txt', samples5)
samples5 = np.loadtxt('data/dbl_gauss_straight_line.txt')
samples5_copy = np.copy(samples5)
plots.plot_corner_full(samples5_copy, 'dbl_gauss1')

# ndim, nwalkers = 9, 100
# param_labels = ['b1', 'b2', 'b3', 'lnb', 'r2', 'lnr', 'alpha', 'beta', 'zeta']
# gb = [-0.06, +1.8, -12.0, -0.9, -1.5, -1.1, 10.6, -0.96, -2.2]
# posb = [gb + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
# sampler6 = emcee.EnsembleSampler(nwalkers, ndim, likelihoods.log_marg_mainsequence_full1b, pool = pool)
# sampler6.run_mcmc(posb, 1000, progress=True)
# af = sampler6.acceptance_fraction
# print("Mean acceptance fraction:", np.mean(af))
# plots.plot_samples_full(sampler6, ndim, 'mainsequence_full2', param_labels)
# samples6 = sampler6.chain[:, 800:, :].reshape((-1, ndim))
# np.savetxt('data/dbl_gauss_offset.txt', samples6)
samples6 = np.loadtxt('data/dbl_gauss_straight_line.txt')
samples6_copy = np.copy(samples6)
plots.plot_corner_full(samples6_copy, 'dbl_gauss2')

# with the H1 estimation
samples4 = np.loadtxt('data/samples4.txt')
samples5 = np.hstack((samples5, samples4[:len(samples5), -3:]))
samples6 = np.hstack((samples6, samples4[:len(samples6), -3:]))
samples6_copy = np.hstack((samples6_copy, samples4[:len(samples6_copy), -3:]))
samples6_copy[:, -1] = np.exp(samples6_copy[:, -1])
# print (np.shape(samples_SDSS))
samples_SDSS = np.hstack((samples_SDSS, samples4[:len(samples_SDSS), -3:]))
ndim = 13
param_labels = ['b1', 'b2', 'b3', 'lnb', 'r1', 'r2', 'lnr', 'alpha', 'beta', 'zeta', 'h1', 'h2', 'log(h1)']
for i in range(ndim):
    mcmc = np.percentile(samples6_copy[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], param_labels[i])
    # display(Math(txt))
    print (txt)


# params for Baldry+11 GSMF we are using
gsmf_params = 10.66, (0.7**3)*3.96E-3, (0.7**3)*0.79E-3, - 0.35, - 1.47
M = np.linspace(7,12,100)
# phi_Mstar_double = double_schechter_peak(M, 9.0, gsmf_params, -4)
# phi_Mstar_double = double_schechter_peak(M, 9.5, gsmf_params, -4)
# phi_Mstar_double = double_schechter_peak(M, 10.0, gsmf_params, -4)

# print ('Integral p(SFR_T|M*)dSFR = ', quad(integrands.integrand_SFR1c, -np.inf, np.inf, args=(10, samples6[10,:]))[0])
# # print ('testing p(SFR|M*)*phi(M*)', dblquad(integrands.integrand_MHI_total, -np.inf, np.inf,, lambda SFR: min_mass, lambda SFR: max_mass, args=(element, params, gsmf_params))[0])
# print ('Integral p(SFR_x|M*)dSFR = ', quad(models.Gaussian_Conditional_Probability, -200, 200, args = (0,-1.1))[0])
#
# print ('Analytical Answer Single Schechter = ', schechter.single_schechter_analytic(np.power(10,8), gsmf_params))
# print ('Analytical Answer Double Schechter = ', schechter.double_schechter_analytic(np.power(10,8), gsmf_params))
# # print ('(Linear) Integral phi(M*)dM* = ', quad(double_schechter_linear, 0, 10E13, args=(np.array(gsmf_params)))[0])
# # print ('(Linear) Integral phi(M*)dM* = ', quad(single_schechter_linear, np.power(10,8), np.inf, args=(np.array(gsmf_params)))[0])
# print ('(Log) Integral phi(M*)dM* = ', quad(schechter.double_schechter, 8, 1000, args=(np.array(gsmf_params)))[0])
# print ('(Log) Integral phi(M*)dM* = ', quad(schechter.single_schechter, 8, np.inf, args=(np.array(gsmf_params)))[0])

# plots.mass_functions(gsmf_params, samples6)

# plots.sfrmplane(GAMA, SDSS2, samples5, samples6, samples_SDSS)
sfr_hist_only(samples_SDSS, samples6, 0, 12, gsmf_params, 'sfr_hist_double_full_sample')
# MHI_mf_only(samples_SDSS, samples6, -5, 3, 0, 12, gsmf_params)


# np.savetxt('data/samples1.txt', samples4)
#
# ndim, nwalkers = 3, 100
# g = [0.8, 10.0, -1.1]
# pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
#
# sampler5 = emcee.EnsembleSampler(nwalkers, ndim, likelihoods.SFR_HI_fit, pool = pool)
# sampler5.run_mcmc(pos, 1500, progress=True)
# plots.plot_samples_full(sampler5, ndim, 'mainsequence_full')
# samples5 = sampler5.chain[:, 800:, :].reshape((-1, ndim))
# np.savetxt('data/samples5.txt', samples5)

# m_gas_ratio(det)
# do the calculation of the galaxy stellar mass functions
# M, phi_Baldry, phi_GAMA18, xbaldry, ybaldry, baldry_err = plots.mass_functions()
# read in the already run and cut chains
# samples4 = np.loadtxt('data/samples4.txt')
# samples5 = np.loadtxt('data/samples5.txt')
# make 8 panel plots howing all the different trends and parameter estimation fits
# sfr_histogram(GAMA, samples4, samples5, M, phi_Baldry)
# plots.sfrmplane(GAMA_sf, GAMA_pass, samples3)
