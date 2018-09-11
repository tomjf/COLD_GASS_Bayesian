import numpy as np

def integrand_MHI(double M, double SFR, double MHI, *params):
    a1, a2, a3, lnf, b1, b2, lnf1 = params
    cdef double Mstar = 10.72
    cdef double phistar1 = 0.71E-3
    cdef double alpha1 = -1.45
    cdef double phi_Mstar, P_SFR_given_Mstar, P_MHI_given_SFR = 0
    phi_Mstar = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))
    P_SFR_given_Mstar = (1/np.sqrt(2*np.pi*np.power(lnf,2))) * np.exp((-1/(2*np.power(lnf,2))) * np.power((SFR-((a1*M*M) + (a2*M) + a3)),2))
    P_MHI_given_SFR = (1/np.sqrt(2*np.pi*np.power(lnf1,2))) * np.exp((-1/(2*np.power(lnf1,2))) * np.power((MHI-((b1*SFR) + b2)),2))
    return phi_Mstar * P_SFR_given_Mstar * P_MHI_given_SFR

def integrand_MHI2(double M, double SFR, double MHI, *params):
    a1, a2, a3, lnf, b1, b2, lnf1 = params
    cdef double P_SFR_given_Mstar, P_MHI_given_SFR = 0
    P_SFR_given_Mstar = (1/np.sqrt(2*np.pi*np.power(lnf,2))) * np.exp((-1/(2*np.power(lnf,2))) * np.power((SFR-((a1*M*M) + (a2*M) + a3)),2))
    P_MHI_given_SFR = (1/np.sqrt(2*np.pi*np.power(lnf1,2))) * np.exp((-1/(2*np.power(lnf1,2))) * np.power((MHI-((b1*SFR) + b2)),2))
    return 105.20474037036288*P_SFR_given_Mstar * P_MHI_given_SFR

def integrand_phi(double M):
    cdef double Mstar = 10.72
    cdef double phistar1 = 0.71E-3
    cdef double alpha1 = -1.45
    return np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))

# dblquad(integrand_MHI2, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (9.0,-0.07,1.91,-12.39,0.31,0.80,9.49,0.44))[0]
