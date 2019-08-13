import numpy as np
import pandas as pd
import math
from scipy import integrate
from astropy.io import fits
from astropy.table import Table

import read_files

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

def Schechter_GASS(data, bins):
    l = data['lgMHI'].values
    w = data['weight'].values
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

def estimate_H2_masses(xCOLDGASS_data, samples1, N):
    xCOLDGASS_data['MH2_estimated'] = 0
    xCOLDGASS_data['MH2_estimated_err'] = 0
    xCOLDGASS_data['LCO_estimated'] = 0
    xCOLDGASS_data['LCO_estimated_err'] = 0
    for idx, row in xCOLDGASS_data.iterrows():
        # only work out the estimated error for the non-detections
        if row['LOGMH2_ERR'] == 0:
            masses = []
            LCOs = []
            for c1, c2, lnc in samples1[np.random.randint(len(samples1), size=N)]:
                # print (row['LOGMSTAR'], c1, c2, c1*row['LOGMSTAR'] + c2 + np.random.normal(0, np.exp(lnc)))
                MH2_mass = c1*row['LOGSFR_BEST'] + c2 + np.random.normal(0, np.exp(lnc))
                masses.append(MH2_mass)
                LCOs.append(MH2_mass- np.log10(row['XCO_A17']))
            xCOLDGASS_data.loc[idx, 'MH2_estimated'] = np.mean(masses)
            xCOLDGASS_data.loc[idx, 'MH2_estimated_err'] = np.std(masses)
            xCOLDGASS_data.loc[idx, 'LCO_estimated'] = np.mean(LCOs)
            xCOLDGASS_data.loc[idx, 'LCO_estimated_err'] = np.std(LCOs)
        else:
            xCOLDGASS_data.loc[idx, 'LCO_estimated'] = np.log10(row['LCO_COR'])
    return xCOLDGASS_data

def match_COLD_GASS_to_GASS():
    xCOLDGASS_data = read_files.read_COLD_GASS()
    xCOLDGASS_data = xCOLDGASS_data[[   'ID', 'LOGMSTAR', 'WEIGHT', 'LOGSFR_BEST',
                                        'LOGSFR_ERR', 'LOGMH2', 'LOGMH2_ERR',
                                        'LIM_LOGMH2']]
    xxGASS = fits.open('data/xxGASS_MASTER_CO_170620_final.fits')
    xxGASS = Table(xxGASS[1].data).to_pandas()
    print (xxGASS.columns)
    xxGASS = xxGASS[['GASS', 'lgMstar', 'HIsrc', 'HIconf_flag', 'lgMHI', 'weight']]
    xxGASS['ID'] = xxGASS['GASS']
    xxGASS = xxGASS[xxGASS['GASS'].isin(xCOLDGASS_data['ID'])]
    newdf = pd.merge(xCOLDGASS_data, xxGASS, on = 'ID')
    samples1 = np.loadtxt('data/SFR_MH2_chain.txt')
    newdf = estimate_H2_masses(newdf, samples1, 200)
    newdf['H2_full'] = newdf['MH2_estimated'] + newdf['LOGMH2']
    newdf['H2_full_err'] = newdf['MH2_estimated_err'] + newdf['LOGMH2_ERR']
    print (newdf)


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

def calc_Omega(bins, yrho):
    rhocrit = 9.2*(10**(-27))
    dMH2 = bins[1] - bins[0]
    rhoH2 = (np.sum((10**yrho)*dMH2)*(2*(10**30)))/((3.086*(10**22))**3)
    OmegaH2 = (rhoH2/rhocrit)
    return OmegaH2

def OmegaH2(bins, yrho):
    rhocrit = 9.2*(10**(-27))
    dMH2 = bins[1] - bins[0]
    rhoH2 = (np.sum((10**yrho)*dMH2)*(2*(10**30)))/((3.086*(10**22))**3)
    OmegaH2 = (rhoH2/rhocrit)
    return OmegaH2

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

    # y=log_schechter(M, b_phi1, b_Mstar, b_alpha1)
    # y2 = log_schechter_true(M, b_phi1, b_Mstar, b_alpha1)
    # y3 = log_double_schechter_true(M, phi1, phi2, Mstar, alph1, alph2)

    # Mstar = 10.66
    # phistar1 = 3.96E-3
    # phistar2 = 0.79E-3
    # alpha1 = - 0.35
    # alpha2 = - 1.47
    # y4 = np.log(10) * np.exp(-np.power(10,M2-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M2-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M2-Mstar)))

    # xmajorLocator   = MultipleLocator(1)
    # xminorLocator   = MultipleLocator(0.2)
    # ymajorLocator   = MultipleLocator(1)
    # yminorLocator   = MultipleLocator(0.2)
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

# def sfr_best(m, logsfr_nuvir, logsfr_nuvir_err):
#
# c = 2.9979E10#in cm/s
# lam1=3.4/1E4
# freq1=c/lam1
# lam3=12.0/1E4 #in cm
# freq3=c/lam3
# lam4=22.0/1E4 #in cm
# freq4=c/lam4
# delM4=6.620  # this is for W4
# delM3=5.174  # for W3
# lamnuv=2271./1E8
# lamfuv=1528./1E8
# freqnuv=c/lamnuv
# freqfuv=c/lamfuv
#
# logsfr_nuvir, logsfr_nuvir_err = [], []
# logsfr_fuvir, logsfr_fuvir_err = [], []
#
# for idx, row in m.iterrows():
#     DL = lumdist(row['z'],/silent)*3.08E24  # in cm
#     sn12=row['W3_FLUX']/row['W3_FLUXERR']  #WISE 12um
#     sn22=row['W4_FLUX']/row['W4_FLUXERR']  #WISE 22um
#     snUV=row['NUV_FLUX']/row['NUV_FLUXERR'] #GALEX NUV
#     snFUV=row['FUV_FLUX']/row['FUV_FLUXERR'] #GALEX FUV
#     f3=m[i].['W1_FLUX']*1D-23
#     f12=m[i].['W3_FLUX']*1D-23
#     f22=m[i].['W4_FLUX']*1D-23
#     L3=freq1*(4*np.pi*DL**2)*f3/3.83D33
#     ef3=m[i].['W1_FLUXERR']*1D-23
#     ef12=m[i].['W3_FLUXERR']*1D-23
#     L12=freq3*(4*np.pi*DL**2)*f12/3.83D33 # in Lsun
#     eL12=freq3*(4*np.pi*DL**2)*ef12/3.83D33
#     eL3=freq1*(4*np.pi*DL**2)*ef3/3.83D33
#     SFR12=4.91E-10*(L12-0.201*L3)
#     eSFR12=4.91E-10*np.sqrt(eL12**2+0.201**2*eL12**2)
#     if (f3 lt 0.0 or sn12 lt 2.0) then begin
#        sfr12=0.0
#        esfr12=0.01
#     endif
#     fnuv=m[i].['NUV_FLUX']*1D-23
#     efnuv=m[i].['NUV_FLUXERR']*1D-23
#     ffuv=m[i].['FUV_FLUX']*1D-23
#     effuv=m[i].['FUV_FLUXERR']*1D-23
#     Lnuv=freqnuv*(4*np.pi*DL**2)*fnuv #in erg/s
#     eLnuv=freqnuv*(4*np.pi*DL**2)*efnuv
#     Lfuv=freqfuv*(4*np.pi*DL**2)*ffuv #in erg/s
#     eLfuv=freqfuv*(4*np.pi*DL**2)*effuv
#     sfrnuv=Lnuv*6.76E-44  #calibration from Kennicutt+Evans12
#     sfrfuv=Lfuv*4.47E-44  #calibration from Kennicutt+Evans12
#     esfrnuv=eLnuv*6.76E-44
#     esfrfuv=eLfuv*4.47E-44
#     if (fnuv lt 0.0 or snuv lt 2.0) then begin
#        sfruv=0.0
#        esfruv=0.01
#     endif
#        mysfr=sfr12+sfrnuv
#        emysfr=sqrt(esfr12^12+esfrnuv**2)
#        if ((f3 lt 0.0 or sn12 lt 2.0) or (fnuv lt 0.0 or snuv lt 2.0)) then begin
#           mysfr=0.0
#           emysfr=0.0
#        endif
#
#
#        logsfr_nuvir[i]=alog10(mysfr)
#        logsfr_nuvir_err[i]=emysfr/(mysfr*alog(10.0))
# endfor
# end

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

def bootstrap_with_replacement_HI(df, n, bins):
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
        Nm = bootstrapped['lgMHI'].values
        weightNEW = bootstrapped['weight'].values
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
