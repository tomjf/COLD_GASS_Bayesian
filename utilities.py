import numpy as np

def calc_Omega(bins, yrho):
    rhocrit = 9.2*(10**(-27))
    dMH2 = bins[1] - bins[0]
    rhoH2 = (np.sum((10**yrho)*dMH2)*(2*(10**30)))/((3.086*(10**22))**3)
    OmegaH2 = (rhoH2/rhocrit)
    return OmegaH2

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
