import numpy as np
import mpmath
import models

def log_schechter_true(logL, log_phi, log_L0, alpha):
    # print (log_phi, log_L0, alpha)
    log = np.log(10)
    frac = np.power(10,(alpha+1)*(logL-log_L0))
    exp = np.exp(-np.power(10,logL-log_L0))
    return log*log_phi*frac*exp

def double_schechter(M, gsmf_params):
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    return phi_Mstar_double

def single_schechter_analytic(M, gsmf_params):
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    Mstar2 = np.power(10,Mstar)
    phi_Mstar_single = phistar2*mpmath.gammainc(alpha2+1, M/Mstar2)
    return float(str(phi_Mstar_single))

def double_schechter_analytic(M, gsmf_params):
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    Mstar2 = np.power(10,Mstar)
    phi_Mstar_double = phistar1*mpmath.gammainc(alpha1+1, M/Mstar2) + phistar2*mpmath.gammainc(alpha2+1, M/Mstar2)
    return float(str(phi_Mstar_double))

def single_schechter(M, gsmf_params):
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    return phi_Mstar_double

def single_schechter_linear(M, gsmf_params):
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    Mstar2 = np.power(10,Mstar)
    ratio = M/Mstar2
    phi_Mstar_single = np.exp(-ratio)*((phistar2*np.power(ratio, alpha2)))*(1/Mstar2)
    return phi_Mstar_single

def double_schechter_linear(M, gsmf_params):
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    Mstar2 = np.power(10,Mstar)
    ratio = M/Mstar2
    # part1 = np.exp(-M/Mstar)
    # part2 = (phistar1*np.power(M/Mstar, alpha1))
    # part3 = (phistar2*np.power(M/Mstar, alpha2))
    # print ('exp', part1)
    # print ('pow1', part2)
    # print ('pow2', part3)
    phi_Mstar_double = np.exp(-ratio)*((phistar1*np.power(ratio, alpha1)) + (phistar2*np.power(ratio, alpha2)))*(1/Mstar2)
    return phi_Mstar_double

def schechterL(luminosity, phiStar, alpha, LStar):
    """Schechter luminosity function."""
    LOverLStar = (luminosity/LStar)
    return (phiStar/LStar) * LOverLStar**alpha * np.exp(- LOverLStar)


def double_schechter_peak(M, M_peaked, gsmf_params, sigma):
    # phi_Mstar_double = double_schechter(M_peaked, gsmf_params)
    phi_Mstar_double = 0.1
    phi = phi_Mstar_double*np.sqrt(2*np.pi*np.exp(sigma)*np.exp(sigma))*models.Gaussian_Conditional_Probability(M, M_peaked, sigma)
    return phi
