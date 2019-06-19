import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import corner

# import COLD GASS functions
import schechter
import models

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
    samples_input[:, 3] = np.exp(samples_input[:, 3])
    samples_input[:, 6] = np.exp(samples_input[:, 6])
    corner.corner(  samples_input,
                    labels=[r"$b1$", r"$b2$", r"$b3$", r"$lnb$",
                    r"$r1$", r"$r2$", r"$lnr$", r"$\alpha$", r"$\beta$", r"$\zeta$"],
                    truths=(np.median(samples_input[:, 0]),
                    np.median(samples_input[:, 1]),
                    np.median(samples_input[:, 2]),
                    np.median(samples_input[:, 3]),
                    np.median(samples_input[:, 4]),
                    np.median(samples_input[:, 5]),
                    np.median(samples_input[:, 6]),
                    np.median(samples_input[:, 7]),
                    np.median(samples_input[:, 8]),
                    np.median(samples_input[:, 9])),
                  truth_color="k",
                  quantiles=[0.16, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('img/corner/' + fname)

def sfrmplane(GAMA, SDSS, samples3, samples6, samples_SDSS):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, squeeze=False, figsize=(12,6))
    x = np.linspace(7,12,300)
    # ax[0,0].scatter(GAMA['logM*'], GAMA['logSFR'], s = 0.1, color = 'k')
    ax[0,0].errorbar(SDSS['mstellar_median'][:300], SDSS['sfr_tot_p50'][:300], xerr = SDSS['mstellar_err'][:300], yerr = SDSS['sfr_tot_p84'][:300] - SDSS['sfr_tot_p50'][:300], label = 'SDSS', fmt='o', markersize = 0.1, linewidth=0.1, capsize=0.1)
    # print (SDSS['sfr_tot_p50'])
    # ax[0,0].scatter(SDSS['mstellar_median'], SDSS['sfr_tot_p50'], s = 0.1, color = 'r')
    ax[0,0].plot(x, models.Saintonge16_MS(x), color = 'orange', label = 'Saintonge+16')
    ax[0,0].plot(x, models.first_order(x, 1.037, - 0.077 - 10), color = 'g', label = 'Bull+16')
    # print (models.second_order(x, np.median(samples3[:,0]), np.median(samples3[:,1]), np.median(samples3[:,2])))
    for b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh in samples3[np.random.randint(len(samples3), size=100)]:
        ax[0,0].plot(x, models.second_order(x, b1, b2, b3), alpha = 0.1, color = 'b')
        ax[0,0].plot(x, models.first_order(x, r1, r2), alpha = 0.1, color = 'r')
        ax[0,1].plot(x, models.f_passive(x, alpha, beta, zeta), color = 'g', alpha = 0.1)
    for b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta in samples_SDSS[np.random.randint(len(samples_SDSS), size=100)]:
        ax[0,0].plot(x, models.second_order(x, b1, b2, b3), alpha = 0.1, color = 'navy')
        ax[0,0].plot(x, models.first_order(x, r1, r2), alpha = 0.1, color = 'crimson')
        ax[0,1].plot(x, models.f_passive(x, alpha, beta, zeta), color = 'darkgreen', alpha = 0.1)
    param1 = (np.median(samples3[:,0]))
    param2 = (np.median(samples3[:,1]))
    param3 = (np.median(samples3[:,2]))
    print (param1, param2, param3)
    ax[0,0].plot(x, models.second_order(x, param1, param2-0.1, param3), color = 'k', label = 'median')

    ax[0,0].set_xlabel(r"$\mathrm{\log_{10} M_{*}\, [M_{\odot}]}$")
    ax[0,0].set_ylabel(r"$\mathrm{\log_{10} SFR \, [M_{\odot}\, yr^{-1}]}$")
    ax[0,1].set_xlabel(r"$\mathrm{\log_{10} M_{*}\, [M_{\odot}]}$")
    ax[0,1].set_ylabel(r"$\mathrm{f_{passive}}$")
    ax[0,0].set_xlim(7, 12)
    ax[0,0].set_ylim(-5, 1.5)
    ax[0,1].set_xlim(7, 12)
    ax[0,1].set_ylim(0, 1)
    # ax[0,0].legend()
    plt.savefig('img/test.pdf')

def mass_functions(gsmf_params, samples1):
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))

    xbaldry = [7.10, 7.30, 7.5, 7.7, 7.9, 8.1, 8.3, 8.5, 8.7, 8.9, 9.1, 9.3, 9.5, 9.7, 9.9, 10.1, 10.3, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5, 11.7, 11.9]
    baldry = [17.9, 43.1, 31.6, 34.8, 27.3, 28.3, 23.5, 19.2, 18.0, 14.3, 10.2, 9.59, 7.42, 6.21, 5.71, 5.51, 5.48, 5.12, 3.55, 2.41, 1.27, 0.338, 0.042, 0.021, 0.042]
    baldry_err = [5.7, 8.7, 9.0, 8.4, 4.2, 2.8, 3.0, 1.2, 2.6, 1.7, 0.6, 0.55, 0.41, 0.37, 0.35, 0.34, 0.34, 0.33, 0.27, 0.23, 0.16, 0.085, 0.030, 0.021, 0.030]
    baldry = np.array(baldry)/1000
    baldry_err = (np.array(baldry_err)/1000)/(baldry*np.log(10))

    # ndim, nwalkers = 5, 100
    # param_labels = ['Mstar', 'phistar1', 'phistar2', 'alpha1', 'alpha2']
    # g = [10.66, 0.00396, -0.35, 0.00079, -1.47]
    # # g = [-0.06, +1.8, -12.0, -0.9, .64, -8.23, -1.1, 10.6, -0.96, -2.2, 0.8, 10.0, -1.1]
    # pos = [g + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    # sampler5 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (xbaldry, baldry, baldry_err))
    # sampler5.run_mcmc(pos, 1000, progress=True)
    # af = sampler5.acceptance_fraction
    # print("Mean acceptance fraction:", np.mean(af))
    # plots.plot_samples_full(sampler5, ndim, 'double_schechter1', param_labels)
    # samples5 = sampler5.chain[:, 500:, :].reshape((-1, ndim))
    # Baldry = {'Mstar': 10.66, 'phistar1': 3.96E-3, 'phistar2': 0.79E-3, 'alpha1': - 0.35, 'alpha2': - 1.47}
    # GAMA18 = {'Mstar': 10.78, 'phistar1': 2.93E-3, 'phistar2': 0.63E-3, 'alpha1': - 0.62, 'alpha2': - 1.50}

    M = np.linspace(6,13,1000)
    M2 = np.logspace(6,13,1000)

    # phi_Mstar_Baldry = np.log(10) * np.exp(-np.power(10,M-Mstar)) * \
    # (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + \
    # phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    phi_Mstar_Baldry = schechter.double_schechter(M, gsmf_params)
    # phi_Mstar_Baldry3 = schechterL(M, 0.71E-3, -1.45, 10.72)
    # print (phi_Mstar_Baldry)
    # print (phi_Mstar_Baldry2)
    # phi_peak1 = double_schechter_peak(M, 9.0, gsmf_params, -4)
    # phi_peak2 = double_schechter_peak(M, 9.5, gsmf_params, -4)
    # phi_peak3 = double_schechter_peak(M, 10.0, gsmf_params, -4)

    # mass_steps = np.linspace(8,12,9)
    # for idx, element in enumerate(mass_steps):
    #     phi_peak = double_schechter_peak(M, element, gsmf_params, -4)
    #     ax[0,0].plot(M, np.log10(phi_peak))
    #     ax[0,1].plot(M, phi_peak*(M[1]-M[0])*10000)


    # phi_Mstar_GAMA18 = np.log(10) * np.exp(-np.power(10,M-GAMA18['Mstar'])) * \
    # (GAMA18['phistar1']*np.power(10,(GAMA18['alpha1']+1)*(M-GAMA18['Mstar'])) + \
    # GAMA18['phistar2']*np.power(10,(GAMA18['alpha2']+1)*(M-GAMA18['Mstar'])))
    # print (np.log10(phi_Mstar_Baldry) - np.log10(phi_Mstar_Baldry2*M2))
    # for params in samples5[np.random.randint(len(samples5), size=10)]:
    #     Mstar, phistar1, phistar2, alpha1, alpha2 = params
    #     params = Mstar, phistar1, phistar2, alpha1, alpha2
    #     print (np.log10(schechter.double_schechter(M, params)))
    #     ax[0,0].plot(M, np.log10(schechter.double_schechter(M, params)), color = 'k')
    for params in samples1[np.random.randint(len(samples1), size=10)]:
        b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
        ax[0,0].plot(M, np.log10(schechter.double_schechter(M, gsmf_params)*models.f_passive(M, alpha, beta, zeta)), color = 'r')
        ax[0,0].plot(M, np.log10(schechter.double_schechter(M, gsmf_params)*(1-models.f_passive(M, alpha, beta, zeta))), color = 'b')

    ax[0,0].plot(M, np.log10(phi_Mstar_Baldry), label = 'Baldry+11')
    Wright17_params = [10.78, 2.93E-3, 0.63E-3, -0.62, -1.50]
    ax[0,0].plot(M, np.log10(schechter.double_schechter(M, Wright17_params)), label = 'Wright+17')
    # ax[0,0].plot(np.log10(M2), np.log10(phi_Mstar_Baldry3), label = 'Baldry', linestyle = ':')
    # ax[0,0].plot(M, np.log10(phi_peak1), label = 'Baldry', color = 'b')
    # ax[0,0].plot(M, np.log10(phi_peak2), label = 'Baldry', color = 'g')
    # ax[0,0].plot(M, np.log10(phi_peak3), label = 'Baldry', color = 'r')

    # ax[0,1].plot(M, phi_peak1*(M[1]-M[0])*10000, label = 'Baldry', color = 'b')
    # ax[0,1].text(0,0, str(round(np.sum(phi_peak1*(M[1]-M[0])*10000), 2)), color = 'b')
    # ax[0,1].plot(M, phi_peak2*(M[1]-M[0])*10000, label = 'Baldry', color = 'g')
    # ax[0,1].text(0,0.2, str(round(np.sum(phi_peak2*(M[1]-M[0])*10000), 2)), color = 'g')
    # ax[0,1].plot(M, phi_peak3*(M[1]-M[0])*10000, label = 'Baldry', color = 'r')
    # ax[0,1].text(0,0.4, str(round(np.sum(phi_peak3*(M[1]-M[0])*10000), 2)), color = 'r')
    # ax[0,0].plot(M, np.log10(phi_Mstar_GAMA18), label = 'GAMA18')
    ax[0,0].errorbar(xbaldry, np.log10(baldry), yerr = baldry_err, label = 'GAMA', fmt = 'o')
    ax[0,0].set_xlabel(r"$\mathrm{\log_{10} M_{*} \, [M_{\odot}]}$")
    ax[0,0].set_ylabel(r"$\mathrm{\log_{10} \phi(M_{*}) \, [Mpc^{-3} \, dex^{-1}]}$")
    ax[0,0].set_xlim(6.8,13.0)
    ax[0,0].set_ylim(-10, 0)
    plt.legend()
    plt.savefig('img/mass_functions.pdf')
    # return M, phi_Mstar_Baldry, phi_Mstar_GAMA18, xbaldry, np.log10(baldry), baldry_err
