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
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

def plot_SFR_M_plane(GAMA, soln, soln2, x1, y1, std):
    print (soln['x'])
    print (soln2['x'])
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    # plt.scatter(GAMA['logM*'], GAMA['logSFR'])
    x=np.linspace(7,12,500)
    y=(-.01828*x*x*x)+(0.4156*x*x) -(2.332*x)
    y2=(soln.x[0]*x*x)+(soln.x[1]*x) +(soln.x[2])
    y3=(soln2.x[0]*x)+(soln2.x[1])
    ax[0,0].errorbar(x1,y1, yerr=std, fmt='h', capsize = 4, markersize = 7, linewidth=2, markeredgewidth=2, capthick=3,  mfc='w', mec='k', ecolor = 'k')
    ax[0,0].plot(x,y, color = 'red', label = 'Saintonge+16')
    ax[0,0].plot(x,y2, color = 'k', label = 'Median Bins')
    ax[0,0].plot(x,y3, color = 'b', label = 'GAMA data')
    ax[0,0].hist2d(GAMA['logM*'], GAMA['logSFR'], bins=100, cmap = 'Blues', vmin=1,vmax =8)
    ax[0,0].set_xlim(7,12)
    # plt.xlim(8,11 .5)
    ax[0,0].set_ylim(-3.5,1.5)
    plt.legend()
    plt.savefig('sfrmplane.pdf')
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

def SFRM_plane():
    # Reading in the GAMA file and converting to pandas df
    GAMA = fits.open('GAMA.fits')
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
    print (GAMA)

def read_GAMA_A():
    GAMA = pd.read_csv('GAMA_sample.dat', comment = '#', header = None, sep=r"\s*")
    GAMA.columns = ['CATAID', 'z', 'logM*', 'logM*err', 'logSFR', 'logSFRerr', 'ColorFlag']
    GAMA = GAMA[np.isfinite(GAMA['logSFR'])]
    GAMA = GAMA[np.isfinite(GAMA['logM*'])]
    GAMA = GAMA[GAMA['logM*']>7.0]
    GAMA = GAMA[GAMA['logM*']<12]
    GAMA = GAMA[GAMA['logSFR']<1.5]
    GAMA = GAMA[GAMA['logSFR']>-3.5]
    GAMA = GAMA[GAMA['ColorFlag']==1]


    bins = np.linspace(8,11,21)
    x1, y1, std = [], [], []
    for i in range (1,len(bins)):
        # print (bins)
        inbin = GAMA[(GAMA['logM*']>=bins[i-1]) & (GAMA['logM*']< bins[i])]
        x1.append((bins[i]+bins[i-1])/2)
        y1.append(np.median(inbin['logSFR']))
        std.append(np.std(inbin['logSFR']))
    x1, y1, std = np.array(x1), np.array(y1), np.array(std)

    nll = lambda *args: -log_likelihood_SFR_M2(*args)
    initial = np.array([ -0.08345945,   2.23921288, -14.08417751, 0.4])
    soln = minimize(nll, initial, args=(x1, y1, std))

    nll = lambda *args: -log_likelihood_SFR_M(*args)
    initial = np.array([2, -20, 0.2])
    soln2 = minimize(nll, initial, args=(GAMA['logM*'], GAMA['logSFR'], GAMA['logSFRerr']))
    plot_SFR_M_plane(GAMA, soln, soln2, x1, y1, std)

    ndim, nwalkers = 3, 100
    pos = [soln2["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(GAMA['logM*'], GAMA['logSFR'], GAMA['logSFRerr']))
    sampler.run_mcmc(pos, 500)
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    plot_emcee_result(GAMA, samples, soln2)

    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
    ax[0,0] = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],truths=[soln2.x[0], soln2.x[1], soln2.x[2]])
    plt.savefig("triangle.png")

    # print (soln['x'])
    # print (soln2['x'])


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

def log_likelihood_SFR_M2(theta, x, y, yerr):
    # print (x)
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
    # print (M)
    # print (M)
    dM = M[1]-M[0]
    Mstar = 10.66
    phi1 = 3.96E-3
    phi2 = 0.79E-3
    alph1 = -0.35
    alph2 = -1.47
    phis = []
    for i in range(0,len(M)):
        frac = 10**(M[i]-Mstar)
        exp = math.exp(-frac)
        phibit1 = (phi1*(frac**(alph1+1)))
        phibit2 = (phi2*(frac**(alph2+1)))
        log = np.log(10)
        phis.append(exp*log*(phibit1+phibit2))
    xmajorLocator   = MultipleLocator(1)
    xminorLocator   = MultipleLocator(0.2)
    ymajorLocator   = MultipleLocator(1)
    yminorLocator   = MultipleLocator(0.2)
    # fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(8,8))
    # ax[0,0].tick_params(axis='both', which='major', labelsize=15)
    # ax[0,0].plot(xbaldry, np.log10(baldry), color = 'b', label = 'baldry')
    # ax[0,0].plot(M, np.log10(phis), color = 'g', label ='fit')
    # plt.savefig('baldry.pdf')
    # plt.legend()
    return phis, M, baldry

def log_schechter(logL, log_phi, log_L0, alpha):
    schechter = log_phi
    schechter += (alpha+1)*(logL-log_L0)*np.log(10)
    schechter -= pow(10,logL-log_L0)
    return schechter

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

# Vm calc ######################################################################
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

# Schechter ####################################################################

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


## MAIN ########################################################################

# SFRM_plane()
read_GAMA_A()
bins = np.linspace(5.5,11,18)
bootstrap = 0.8

################################################################################
# Producing the pandas catalogue

# Reading in the COLD GASS file and converting to pandas df
xCOLDGASS = fits.open('xCOLDGASS_PubCat.fits')
xCOLDGASS_data = Table(xCOLDGASS[1].data).to_pandas()
# Calculate lumdist, Vm, MH2 including limits
V_CG, V_CG2 = calcVm(xCOLDGASS_data, len(xCOLDGASS_data), bootstrap)
xCOLDGASS_data['D_L'] = lumdistance(xCOLDGASS_data)
xCOLDGASS_data['V_m'] = V_CG
xCOLDGASS_data['V_m2'] = V_CG2
xCOLDGASS_data['MH2'] = xCOLDGASS_data['LOGMH2'] + xCOLDGASS_data['LIM_LOGMH2']
xCOLDGASS_data['new_LOGMH2'] = 0
# print (xCOLDGASS_data.columns)
# print (xCOLDGASS_data)
################################################################################




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
