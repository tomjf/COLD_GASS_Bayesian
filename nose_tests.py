import numpy as np
from scipy.integrate import quad, dblquad, nquad

# import COLD GASS functions
import schechter
import integrands
import models

# check the integral of the total PDF for the SFR-M plane is equal to 1
def test_integral_pSFR_total_M_dSFR():
    samples4 = np.loadtxt('data/samples4.txt')
    samples6 = np.loadtxt('data/dbl_gauss_straight_line.txt')
    samples6 = np.hstack((samples6, samples4[:len(samples6), -3:]))
    p_SFR_M_integrated = quad(integrands.integrand_SFR1c, -np.inf, np.inf, args=(10, samples6[10,:]))[0]
    print (p_SFR_M_integrated)
    assert round(p_SFR_M_integrated,10) == 1.0

# check the integral of the one of the PDFs for the SFR-M plane is equal to 1
def test_integral_pSFR_total_X_dSFR():
    p_SFR_M_integrated = quad(models.Gaussian_Conditional_Probability, -np.inf, np.inf, args=(0, 1.1))[0]
    assert round(p_SFR_M_integrated,10) == 1.0

# check the integral of the single schechter mass function written in log_10 M*
# equals the analytical solution
def test_integral_phi_single_numerical_vs_analytic():
    gsmf_params = 10.66, 3.96E-3, 0.79E-3, - 0.35, - 1.47
    analytical = schechter.single_schechter_analytic(np.power(10,8), gsmf_params)
    numerical = quad(schechter.single_schechter, 8, 20, args=(np.array(gsmf_params)))[0]
    assert round(analytical,10) == round(numerical,10)

# check the integral of the double schechter mass function written in log_10 M*
# equals the analytical solution
def test_integral_phi_double_numerical_vs_analytic():
    gsmf_params = 10.66, 3.96E-3, 0.79E-3, - 0.35, - 1.47
    analytical = schechter.double_schechter_analytic(np.power(10,8), gsmf_params)
    numerical = quad(schechter.double_schechter, 8, 15, args=(np.array(gsmf_params)))[0]
    assert round(analytical,10) == round(numerical,10)


# check the integral of the model sfr histogram has the same number density as
# the input double schechter mass function
def test_number_conserved_full_schechter():
    samples4 = np.loadtxt('data/samples4.txt')
    samples6 = np.loadtxt('data/dbl_gauss_straight_line.txt')
    samples6 = np.hstack((samples6, samples4[:len(samples6), -3:]))
    gsmf_params = 10.66, 3.96E-3, 0.79E-3, - 0.35, - 1.47
    mmin, mmax = 0, 16
    sfrmin, sfrmax = -20, 20
    analytical = schechter.double_schechter_analytic(np.power(10,mmin), gsmf_params)
    Num_Dens_MF, err_m = quad(schechter.double_schechter, mmin, mmax,
                                args=(np.array(gsmf_params)),
                                epsrel = 1e-8, epsabs = 0, limit = 100)
    Num_Dens_SFR, err_sfr = dblquad(integrands.integrand_SFR1b, sfrmin, sfrmax,
                                    lambda SFR: mmin, lambda SFR: mmax,
                                    args=(samples6[10,:], gsmf_params),
                                    epsrel = 1e-8, epsabs = 0)
    print ('mass function', Num_Dens_MF, err_m)
    print ('mass function analytical', analytical)
    print ('sfr function', Num_Dens_SFR, err_sfr)
    assert round(Num_Dens_MF,8) == round(Num_Dens_SFR,8)

# check the integral of the model sfr histogram has the same number density as
# the input peaked mass function
def test_number_conserved_full_peak():
    samples4 = np.loadtxt('data/samples4.txt')
    samples6 = np.loadtxt('data/dbl_gauss_straight_line.txt')
    samples6 = np.hstack((samples6, samples4[:len(samples6), -3:]))
    gsmf_params = 10.66, 3.96E-3, 0.79E-3, - 0.35, - 1.47
    mmin, mmax = 0, 15
    sfrmin, sfrmax = -50, 20
    log_sigma = -4
    M = np.linspace(8,12,3)
    for idx, element in enumerate(M):
        Num_Dens_MF, err_m = quad(schechter.double_schechter_peak, mmin, mmax,
                                    args=(element, np.array(gsmf_params), log_sigma),
                                    epsrel = 1e-8, epsabs = 0)
        Num_Dens_SFR, err_sfr = dblquad(integrands.integrand_SFR1, sfrmin, sfrmax,
                                    lambda SFR: mmin, lambda SFR: mmax,
                                    args=(element, log_sigma, samples6[10,:], gsmf_params),
                                    epsrel = 1e-8, epsabs = 0)
        print ('mass function', element, Num_Dens_MF, err_m)
        print ('sfr function', element, Num_Dens_SFR, err_sfr)
        assert round(Num_Dens_MF,8) == round(Num_Dens_SFR,8)
