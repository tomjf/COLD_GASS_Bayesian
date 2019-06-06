import numpy as np
from scipy.integrate import quad, dblquad, nquad

import schechter
import integrands
import models

def pointless_test():
    a = 2.0
    assert a == 2.0

def test_integral_pSFR_total_M_dSFR():
    samples4 = np.loadtxt('data/samples4.txt')
    samples6 = np.loadtxt('data/dbl_gauss_straight_line.txt')
    samples6 = np.hstack((samples6, samples4[:len(samples6), -3:]))
    p_SFR_M_integrated = quad(integrands.integrand_SFR1c, -np.inf, np.inf, args=(10, samples6[10,:]))[0]
    assert round(p_SFR_M_integrated,10) == 1.0

def test_integral_pSFR_total_X_dSFR():
    p_SFR_M_integrated = quad(models.Gaussian_Conditional_Probability, -np.inf, np.inf, args=(0, 1.1))[0]
    assert round(p_SFR_M_integrated,10) == 1.0

def test_integral_phi_single_numerical_vs_analytic():
    gsmf_params = 10.66, 3.96E-3, 0.79E-3, - 0.35, - 1.47
    analytical = schechter.single_schechter_analytic(np.power(10,8), gsmf_params)
    numerical = quad(schechter.single_schechter, 8, 20, args=(np.array(gsmf_params)))[0]
    assert round(analytical,10) == round(numerical,10)

def test_integral_phi_double_numerical_vs_analytic():
    gsmf_params = 10.66, 3.96E-3, 0.79E-3, - 0.35, - 1.47
    analytical = schechter.double_schechter_analytic(np.power(10,8), gsmf_params)
    numerical = quad(schechter.double_schechter, 8, 15, args=(np.array(gsmf_params)))[0]
    assert round(analytical,10) == round(numerical,10)
