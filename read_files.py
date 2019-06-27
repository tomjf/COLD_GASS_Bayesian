import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table

import models

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
    return GAMA, GAMAb, GAMAr

def read_SDSS():
    SDSS = fits.open('data/Skyserver_query.fits')
    SDSS = Table(SDSS[1].data).to_pandas()
    print (SDSS.columns)
    print(len(SDSS))
    SDSS = SDSS[SDSS['sfr_tot_p50'] > -9999]
    SDSS = SDSS[SDSS['mstellar_median'] > 5.0]
    print (len(SDSS))
    SDSS['errs'] = (SDSS['sfr_tot_p84'] - SDSS['sfr_tot_p16'])/2
    SDSS['ratio'] = (SDSS['sfr_tot_p84'] - SDSS['sfr_tot_p50'])/(SDSS['sfr_tot_p50'] - SDSS['sfr_tot_p16'])
    # SDSS2 = SDSS[SDSS['ratio'] > 0.5]
    # SDSS2 = SDSS2[SDSS2['ratio'] < 1.5]
    # print ('fraction without uneven errors', len(SDSS2)/len(SDSS))
    # print (min(SDSS2['z']))
    # print (max(SDSS2['z']))
    # print (min(SDSS2['mstellar_median']))
    # print (max(SDSS2['mstellar_median']))
    # print (min(SDSS2['sfr_tot_p50']))
    # print (max(SDSS2['sfr_tot_p50']))
    SDSS_red = SDSS[SDSS['sfr_tot_p50'] < models.second_order(SDSS['mstellar_median'], -0.0325, 1.2222, -9.1109 - 1.0)]
    SDSS_blue = SDSS[SDSS['sfr_tot_p50'] >= models.second_order(SDSS['mstellar_median'], -0.0325, 1.2222, -9.1109 - 1.0)]
    return SDSS, SDSS, SDSS_blue, SDSS_red

def read_Chiang():
    input = fits.open('data/sw_input.fits')
    input = Table(input[1].data).to_pandas()
    output = fits.open('data/sw_output.fits')
    output = Table(output[1].data).to_pandas()
    output['z'] = input['redshift']
    output = output[output['z']<0.06]
    output = output[output['z']>0.005]
    output['m_err'] = 0
    output['sfr_err'] = 0
    # print (output[['lmass16_all', 'lmass50_all', 'lmass84_all']])
    # print (output[['lsfr16_all', 'lsfr50_all', 'lsfr84_all']])
    return output
