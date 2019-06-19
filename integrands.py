import numpy as np

# import COLD GASS functions
import schechter
import models

def integrand_SFR1(M, SFR, m_step, log_sigma, params, gsmf_params):
    # parameters inferred from emcee
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    # probabilities
    phi_Mstar_double = schechter.double_schechter_peak(M, m_step, gsmf_params, log_sigma)
    fpass = models.f_passive(M, alpha, beta, zeta)
    # P_SFR_given_passive
    y_passive = r1*M + r2
    P_SFR_given_Mstar_red = models.Gaussian_Conditional_Probability(SFR, y_passive, lnr)
    # P_SFR_given_sforming
    y_sforming = (b1*M*M) + (b2*M) + b3
    P_SFR_given_Mstar_blue = models.Gaussian_Conditional_Probability(SFR, y_sforming, lnb)
    # P_SFR_total
    P_SFR_given_Mstar_total = fpass*P_SFR_given_Mstar_red + (1-fpass)*P_SFR_given_Mstar_blue
    # return phi_SFR
    return phi_Mstar_double*P_SFR_given_Mstar_total

def integrand_SFR_Saintonge16(M, SFR, params, gsmf_params):
    # parameters inferred from emcee
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    # probabilities
    phi_Mstar_double = schechter.double_schechter(M, gsmf_params)
    fpass = models.f_passive(M, alpha, beta, zeta)
    # P_SFR_given_passive
    # y_passive = r1*M + r2
    # P_SFR_given_Mstar_red = models.Gaussian_Conditional_Probability(SFR, y_passive, lnr)
    # P_SFR_given_sforming
    y_sforming = models.Saintonge16_MS(M)
    P_SFR_given_Mstar_blue = models.Gaussian_Conditional_Probability(SFR, y_sforming, lnb)
    # P_SFR_total
    P_SFR_given_Mstar_total = (1-fpass)*P_SFR_given_Mstar_blue
    # return phi_SFR
    return phi_Mstar_double*P_SFR_given_Mstar_total

def integrand_SFR_blue1(M, SFR, params, gsmf_params):
    # parameters inferred from emcee
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    # probabilities
    phi_Mstar_double = schechter.double_schechter(M, gsmf_params)
    fpass = models.f_passive(M, alpha, beta, zeta)
    # P_SFR_given_blue
    y_sforming = (b1*M*M) + (b2*M) + b3
    # y_sforming = models.first_order(M, 1.037, - 0.077 - 10)
    # y_sforming = models.Saintonge16_MS(M)
    P_SFR_given_Mstar_blue = models.Gaussian_Conditional_Probability(SFR, y_sforming, -0.94)
    P_SFR_given_Mstar_total = (1-fpass)*P_SFR_given_Mstar_blue
    # return phi_SFR
    return phi_Mstar_double*P_SFR_given_Mstar_total

def integrand_SFR_red1(M, SFR, params, gsmf_params):
    # parameters inferred from emcee
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    # probabilities
    phi_Mstar_double = schechter.double_schechter(M, gsmf_params)
    fpass = models.f_passive(M, alpha, beta, zeta)
    # P_SFR_given_passive
    y_passive = r1*M + r2
    P_SFR_given_Mstar_red = models.Gaussian_Conditional_Probability(SFR, y_passive, lnr)
    P_SFR_given_Mstar_total = fpass*P_SFR_given_Mstar_red
    # return phi_SFR
    return phi_Mstar_double*P_SFR_given_Mstar_total

def integrand_SFR1c(SFR, M, params):
    # parameters inferred from emcee
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    # probabilities
    fpass = models.f_passive(M, alpha, beta, zeta)
    # P_SFR_given_passive
    y_passive = r1*M + r2
    P_SFR_given_Mstar_red = models.Gaussian_Conditional_Probability(SFR, y_passive, lnr)
    # P_SFR_given_sforming
    y_sforming = (b1*M*M) + (b2*M) + b3
    P_SFR_given_Mstar_blue = models.Gaussian_Conditional_Probability(SFR, y_sforming, lnb)
    # P_SFR_total
    P_SFR_given_Mstar_total = fpass*P_SFR_given_Mstar_red + (1-fpass)*P_SFR_given_Mstar_blue
    # return phi_SFR
    return P_SFR_given_Mstar_total

def integrand_SFR1b(M, SFR, params, gsmf_params):
    # parameters inferred from emcee
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    # probabilities
    phi_Mstar_double = schechter.double_schechter(M, gsmf_params)
    fpass = models.f_passive(M, alpha, beta, zeta)
    # P_SFR_given_passive
    y_passive = r1*M + r2
    P_SFR_given_Mstar_red = models.Gaussian_Conditional_Probability(SFR, y_passive, lnr)
    # P_SFR_given_sforming
    y_sforming = (b1*M*M) + (b2*M) + b3
    # y_sforming = models.first_order(M, 1.037, - 0.077 - 10)
    # y_sforming = models.Saintonge16_MS(M)
    P_SFR_given_Mstar_blue = models.Gaussian_Conditional_Probability(SFR, y_sforming, -0.94)
    # P_SFR_total
    P_SFR_given_Mstar_total = fpass*P_SFR_given_Mstar_red + (1-fpass)*P_SFR_given_Mstar_blue
    # return phi_SFR
    return phi_Mstar_double*P_SFR_given_Mstar_total

def integrand_SFR2(M, SFR, params, gsmf_params):
    # parameters inferred from emcee
    b1, b2, b3, lnb, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    # probabilities
    phi_Mstar_double = schechter.double_schechter(M, gsmf_params)
    fpass = models.f_passive(M, alpha, beta, zeta)
    # P_SFR_given_passive
    y_passive = (b1*M*M) + (b2*M) + b3 + r2
    P_SFR_given_Mstar_red = (1/np.sqrt(2*np.pi*np.power(np.exp(lnr),2)))*np.exp((-1/(2*np.power(np.exp(lnr),2)))*np.power((SFR-y_passive),2))
    # P_SFR_given_sforming
    y_sforming = (b1*M*M) + (b2*M) + b3
    P_SFR_given_Mstar_blue = (1/np.sqrt(2*np.pi*np.power(np.exp(lnb),2)))*np.exp((-1/(2*np.power(np.exp(lnb),2)))*np.power((SFR-y_sforming),2))
    # P_SFR_total
    P_SFR_given_Mstar_total = fpass*P_SFR_given_Mstar_red + (1-fpass)*P_SFR_given_Mstar_blue
    # return phi_SFR
    return phi_Mstar_double*P_SFR_given_Mstar_total

def integrand_MHI_direct(M, MHI, *params):
    # parameters inferred from emcee
    a1, a2, lna = params

    Mstar = 10.78
    phistar1 = 2.93E-3
    phistar2 = 0.63E-3
    alpha1 = - 0.62
    alpha2 = - 1.50
    # probabilities
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    P_MHI_given_Mstar = (1/np.sqrt(2*np.pi*np.power(np.exp(lna),2)))*np.exp((-1/(2*np.power(np.exp(lna),2)))*np.power((MHI - ((a1*M) + a2)),2))
    # P_SFR_total
    return phi_Mstar_double*P_MHI_given_Mstar

def integrand_SFR_blue2(M, SFR, params, gsmf_params):
    # parameters inferred from emcee
    b1, b2, b3, lnb, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    # probabilities
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    fpass = models.f_passive(M, alpha, beta, zeta)
    y_sforming = (b1*M*M) + (b2*M) + b3
    P_SFR_given_Mstar_blue = (1/np.sqrt(2*np.pi*np.power(np.exp(lnb),2)))*np.exp((-1/(2*np.power(np.exp(lnb),2)))*np.power((SFR-y_sforming),2))
    # P_SFR_total
    P_SFR_given_Mstar_total = (1-fpass)*P_SFR_given_Mstar_blue
    # return phi_SFR
    return phi_Mstar_double*P_SFR_given_Mstar_total

def integrand_SFR_red2(M, SFR, params, gsmf_params):
    # parameters inferred from emcee
    b1, b2, b3, lnb, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    # probabilities
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    fpass = models.f_passive(M, alpha, beta, zeta)
    # P_SFR_given_passive
    y_passive = (b1*M*M) + (b2*M) + b3 + r2
    P_SFR_given_Mstar_red = (1/np.sqrt(2*np.pi*np.power(np.exp(lnr),2)))*np.exp((-1/(2*np.power(np.exp(lnr),2)))*np.power((SFR-y_passive),2))
    P_SFR_given_Mstar_total = fpass*P_SFR_given_Mstar_red
    # return phi_SFR
    return phi_Mstar_double*P_SFR_given_Mstar_total

def integrand_MHI_blue(M, SFR, MHI, *params):
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    lnh
    Mstar, phistar1, alpha1 = 10.72, 0.71E-3, -1.45
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))
    f = (b1*M*M) + (b2*M) + b3
    P_SFR_given_Mstar = (1/np.sqrt(2*np.pi*np.power(np.exp(lnb),2)))*np.exp((-1/(2*np.power(np.exp(lnb),2)))*np.power((SFR-f),2))
    f2 = (h1*SFR) + h2
    P_MHI_given_SFR = (1/np.sqrt(2*np.pi*np.power(np.exp(lnh),2)))*np.exp((-1/(2*np.power(np.exp(lnh),2)))*np.power((MHI-f2),2))
    return phi_Mstar_double*P_SFR_given_Mstar*P_MHI_given_SFR

def integrand_MHI_total(M, SFR, MHI, params, gsmf_params):
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    # probabilities
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    fpass = models.f_passive(M, alpha, beta, zeta)
    # P_SFR_given_passive
    y_passive = r1*M + r2
    P_SFR_given_Mstar_red = (1/np.sqrt(2*np.pi*np.power(np.exp(lnr),2)))*np.exp((-1/(2*np.power(np.exp(lnr),2)))*np.power((SFR-y_passive),2))
    # P_SFR_given_sforming
    y_sforming = (b1*M*M) + (b2*M) + b3
    P_SFR_given_Mstar_blue = (1/np.sqrt(2*np.pi*np.power(np.exp(lnb),2)))*np.exp((-1/(2*np.power(np.exp(lnb),2)))*np.power((SFR-y_sforming),2))
    # P_SFR_total
    P_SFR_given_Mstar_total = fpass*P_SFR_given_Mstar_red + (1-fpass)*P_SFR_given_Mstar_blue
    # P_MHI_given_SFR
    f2 = (h1*SFR) + h2
    P_MHI_given_SFR = (1/np.sqrt(2*np.pi*np.power(np.exp(lnh),2)))*np.exp((-1/(2*np.power(np.exp(lnh),2)))*np.power((MHI-f2),2))
    return phi_Mstar_double*P_SFR_given_Mstar_total*P_MHI_given_SFR

def integrand_MHI_blue(M, SFR, MHI, params, gsmf_params):
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    # probabilities
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    fpass = models.f_passive(M, alpha, beta, zeta)
    # P_SFR_given_sforming
    y_sforming = (b1*M*M) + (b2*M) + b3
    P_SFR_given_Mstar_blue = (1/np.sqrt(2*np.pi*np.power(np.exp(lnb),2)))*np.exp((-1/(2*np.power(np.exp(lnb),2)))*np.power((SFR-y_sforming),2))
    # P_SFR_total
    P_SFR_given_Mstar_total = (1-fpass)*P_SFR_given_Mstar_blue
    # P_MHI_given_SFR
    f2 = (h1*SFR) + h2
    P_MHI_given_SFR = (1/np.sqrt(2*np.pi*np.power(np.exp(lnh),2)))*np.exp((-1/(2*np.power(np.exp(lnh),2)))*np.power((MHI-f2),2))
    return phi_Mstar_double*P_SFR_given_Mstar_total*P_MHI_given_SFR

def integrand_MHI_red(M, SFR, MHI, params, gsmf_params):
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    Mstar, phistar1, phistar2, alpha1, alpha2 = gsmf_params
    # probabilities
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)) + phistar2*np.power(10,(alpha2+1)*(M-Mstar)))
    fpass = models.f_passive(M, alpha, beta, zeta)
    # P_SFR_given_passive
    y_passive = r1*M + r2
    P_SFR_given_Mstar_red = (1/np.sqrt(2*np.pi*np.power(np.exp(lnr),2)))*np.exp((-1/(2*np.power(np.exp(lnr),2)))*np.power((SFR-y_passive),2))
    # P_SFR_total
    P_SFR_given_Mstar_total = fpass*P_SFR_given_Mstar_red
    # P_MHI_given_SFR
    f2 = (h1*SFR) + h2
    P_MHI_given_SFR = (1/np.sqrt(2*np.pi*np.power(np.exp(lnh),2)))*np.exp((-1/(2*np.power(np.exp(lnh),2)))*np.power((MHI-f2),2))
    return phi_Mstar_double*P_SFR_given_Mstar_total*P_MHI_given_SFR
