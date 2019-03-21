import numpy as np

def integrand_MHI(double M, double SFR, *params):
    MHI, a1, a2, a3, lnf, b1, b2, lnf1 = params
    cdef double Mstar = 10.72
    cdef double phistar1 = 0.71E-3
    cdef double alpha1 = -1.45
    cdef double phi_Mstar, P_SFR_given_Mstar, P_MHI_given_SFR = 0
    phi_Mstar = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))
    P_SFR_given_Mstar = (1/np.sqrt(2*np.pi*np.power(lnf,2))) * np.exp((-1/(2*np.power(lnf,2))) * np.power((SFR-((a1*M*M) + (a2*M) + a3)),2))
    P_MHI_given_SFR = (1/np.sqrt(2*np.pi*np.power(lnf1,2))) * np.exp((-1/(2*np.power(lnf1,2))) * np.power((MHI-((b1*SFR) + b2)),2))
    return phi_Mstar * P_SFR_given_Mstar * P_MHI_given_SFR

def integrand_MHI_logistic(double M, double SFR, *params):
    MHI, a1, a2, a3, lnf, A1, b1, b2, lnf1 = params
    cdef double Mstar = 10.72
    cdef double phistar1 = 0.71E-3
    cdef double alpha1 = -1.45
    cdef double phi_Mstar, P_SFR_given_Mstar, P_MHI_given_SFR = 0
    phi_Mstar = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))
    P_SFR_given_Mstar = (lnf*np.exp(-(SFR-((a1*M*M) + (a2*M) + a3))/A1))/(A1 * np.power(1 + np.exp(-(SFR-((a1*M*M) + (a2*M) + a3)/A1)), 1 + lnf))
    P_MHI_given_SFR = (1/np.sqrt(2*np.pi*np.power(lnf1,2))) * np.exp((-1/(2*np.power(lnf1,2))) * np.power((MHI-((b1*SFR) + b2)),2))
    return phi_Mstar * P_SFR_given_Mstar * P_MHI_given_SFR

def integrand_MHI_double(double M, double SFR, double MHI, *params):
    # MHI, a1, a2, a3, lnf, b1, b2, lnf1, d1, d2, lnf2 = params
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    cdef double Mstar = 10.66
    cdef double phistar1 = 3.96E-3
    cdef double phistar2 = 0.79E-3
    cdef double alpha1 = -0.35
    cdef double alpha2 = -1.47
    cdef double phi_Mstar, P_SFR_given_Mstar, P_MHI_given_SFR = 0
    phi_Mstar = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    P_SFR_given_Mstar = ((1/np.sqrt(2*np.pi*np.power(lnb,2))) * np.exp((-1/(2*np.power(lnb,2))) * np.power((SFR-((b1*M*M) + (b2*M) + b3)),2))) + \
    ((1/np.sqrt(2*np.pi*np.power(lnr,2))) * np.exp((-1/(2*np.power(lnr,2))) * np.power((SFR-((r1*M) + r2)),2)))
    P_MHI_given_SFR = (1/np.sqrt(2*np.pi*np.power(lnh,2))) * np.exp((-1/(2*np.power(lnh,2))) * np.power((MHI-((h1*SFR) + h2)),2))
    return phi_Mstar * P_SFR_given_Mstar * P_MHI_given_SFR

def integrand_MH2(double M, double SFR, *params):
    MH2, a1, a2, a3, lnf, c1, c2, lnf2 = params
    cdef double Mstar = 10.72
    cdef double phistar1 = 0.71E-3
    cdef double alpha1 = -1.45
    cdef double phi_Mstar, P_SFR_given_Mstar, P_MHI_given_SFR = 0
    phi_Mstar = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))
    P_SFR_given_Mstar = (1/np.sqrt(2*np.pi*np.power(lnf,2))) * np.exp((-1/(2*np.power(lnf,2))) * np.power((SFR-((a1*M*M) + (a2*M) + a3)),2))
    P_MH2_given_SFR = (1/np.sqrt(2*np.pi*np.power(lnf2,2))) * np.exp((-1/(2*np.power(lnf2,2))) * np.power((MH2-((c1*SFR) + c2)),2))
    return phi_Mstar * P_SFR_given_Mstar * P_MH2_given_SFR

def integrand_MHI_var_sigma(double M, double SFR, *params):
    MHI, a1, a2, a3, a4, a5, b1, b2, lnf1 = params
    cdef double Mstar = 10.72
    cdef double phistar1 = 0.71E-3
    cdef double alpha1 = -1.45
    cdef double phi_Mstar, P_SFR_given_Mstar, P_MHI_given_SFR = 0
    phi_Mstar = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))
    lnf = (a4*M) + a5
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
