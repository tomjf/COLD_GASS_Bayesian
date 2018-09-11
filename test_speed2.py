import numpy as np
from scipy.integrate import quad, dblquad
# from integrand import integrand_MHI
from sympy import *
import math
mpmath.mp.dps = 100

f = lambda x, y: x*y*y

def integrand_MHI(M, SFR):
    # print (M)
    MHI = 9.0
    a1, a2, a3, lnf, b1, b2, lnf1 = -0.07, 1.91, -12.39, 0.31, 0.80, 9.49, 0.44
    Mstar = 10.72
    phistar1 = 0.71E-3
    alpha1 = -1.45
    # phi_Mstar, P_SFR_given_Mstar, P_MHI_given_SFR = 0.0, 0.0, 0.0
    phi_Mstar = np.log(10) * np.exp(float(-np.power(10,M-Mstar))) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))
    P_SFR_given_Mstar = (1/np.sqrt(2*np.pi*np.power(lnf,2)))* np.exp((-1/(2*np.power(lnf,2)))*np.power((float(SFR)-float((a1*M*M) + (a2*M) + a3)),2))
    P_MHI_given_SFR = (1/np.sqrt(2*np.pi*np.power(lnf1,2))) * np.exp((-1/(2*np.power(lnf1,2))) * np.power((float(MHI)-float((b1*SFR) + b2)),2))
    return phi_Mstar * P_SFR_given_Mstar * P_MHI_given_SFR

# MHI = np.linspace(6,11,10)
# for idx, element in enumerate(MHI):
integrand2 = mpmath.quad(integrand_MHI, [-5.0,2.0], [0.0,12.0], error = True)
print (integrand2)
integrand = dblquad(integrand_MHI, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0)[0]
print (integrand)
    # print (integrand)
    # print (integrand2)

    # integrand = dblquad(integrand_MHI, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, -0.07, 1.91, -12.39, 0.31, 0.80, 9.49, 0.44))[0]
    # integrand2 = mpmath.quad(integrand_MHI, [0.0, 12.0], [-5.0, 2.0], args = (element, -0.07, 1.91, -12.39, 0.31, 0.80, 9.49, 0.44))
    # print (integrand)
    # print (integrand)
    # print (integrand2)
    # print (pi)
