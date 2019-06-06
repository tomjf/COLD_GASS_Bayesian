import double_gaussian

def test_phi_num_vs_analytic():
    gsmf_params = 10.66, 3.96E-3, 0.79E-3, - 0.35, - 1.47
    analytical = double_gaussian.single_schechter_analytic(np.power(10,8), gsmf_params)
    numerical = quad(single_schechter, 8, np.inf, args=(np.array(gsmf_params)))[0]
    assert analytical == numerical
