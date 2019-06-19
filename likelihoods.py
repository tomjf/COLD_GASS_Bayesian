import numpy as np

def log_likelihood(theta, x, y, yerr):
    Mstar, phistar1, phistar2, alpha1, alpha2 = theta
    # model = np.log(10.0) * \
    # ((phistar1 * np.power((np.power(10, x) / np.power(10, Mstar)), (alpha1 + 1))) + \
    # (phistar2 * np.power((np.power(10, x) / np.power(10, Mstar)), (alpha2 + 1)))) * \
    # np.exp(-1 * np.power(10, x) / np.power(10, Mstar))
    model = models.double_schechter(x, theta)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))

def log_prior(theta):
    Mstar, phistar1, phistar2, alpha1, alpha2 = theta
    if 10.0 < Mstar < 11.0 and \
    0.000001 < phistar1 < 0.005 and \
    0.000001 < phistar2 < 0.005 and \
    -1.0 < alpha1 < 0.0 and \
    -2.0 < alpha2 < -1.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

def SFR_HI_fit_params(params):
    h1, h2, lnh = params
    if  0.6 < h1 < 1.0 and \
        8.0 < h2 < 11.0 and \
        -2.0 < lnh < 2.0:
        return 0
    return -np.inf

def SFR_HI_fit(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    h1, h2, lnh = params
    x1, x2, y1, y2, S1, S2 = GASS_data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood for the linear SFR-MHI plane
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = np.array([-h1, 1.0])
    deltaN = y1 - (h1 * x1) - h2
    model = (h1 * x2) + h2
    sigma = np.dot(np.dot(S1, v), v) + np.exp(2 * lnh)
    sigma2 = np.dot(np.dot(S2, v), v) + np.exp(2 * lnh)
    sigma2 = sigma2 ** 0.5
    ll1 = -0.5 * np.sum(np.square(deltaN) / sigma + np.log(sigma))
    I = np.zeros(len(x2))
    for i in range(0,len(x2)):
        I[i] = np.log(((2 * np.pi) ** 0.5) * 0.5 * (special.erf((y2[i]-model[i]) / ((2 ** 0.5) * sigma2[i])) + 1))
    ll2 = np.sum(I)
    LL_SFR_MHI = ll1  + ll2
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return LL_SFR_MHI + SFR_HI_fit_params(params)

def MHI_Mstar_fit_params(params):
    a1, a2, lna = params
    if  0.0 < a1 < 2.0 and \
        -5.0 < a2 < 8.0 and \
        -2.0 < lna < 2.0:
        return 0
    return -np.inf

def MHI_Mstar_fit(params):
    a1, a2, lna = params
    x, x2, y, y2, S1, S2 = GASS_data2
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood for the linear SFR-MHI plane
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = np.array([-a1, 1.0])
    deltaN = y - (a1 * x) - a2
    model = (a1 * x2) + a2
    sigma = np.dot(np.dot(S1, v), v) + np.exp(2 * lna)
    sigma2 = np.dot(np.dot(S2, v), v) + np.exp(2 * lna)
    sigma2 = sigma2 ** 0.5
    ll1 = -0.5 * np.sum(np.square(deltaN) / sigma + np.log(sigma))
    I = np.zeros(len(x2))
    for i in range(0,len(x2)):
        I[i] = np.log(((2 * np.pi) ** 0.5) * 0.5 * (special.erf((y2[i]-model[i]) / ((2 ** 0.5) * sigma2[i])) + 1))
    ll2 = np.sum(I)
    LL_SFR_MHI = ll1  + ll2

    return LL_SFR_MHI + MHI_Mstar_fit_params(params)


def log_mainsequence_priors_full(params):
    b1, b2, b3, lnb, r1, r2, r3, lnr, alpha, beta, zeta = params
    if  -0.3 < b1 < 0.0 and \
        2.0 < b2 < 4.0 and \
        -22.0 < b3 < -16.0 and \
        -5.0 < lnb < 5.0 and \
        0.0 < r1 < 1.0 and \
        -14.0 < r2 < -10.0 and \
        50.0 < r3 < 70.0 and \
        -5.0 < lnr < 5.0 and \
        9.0 < alpha < 12.0 and \
        -2.0 < beta < 0.0 and \
        -7.0 < zeta < -2.0:
        return 0
    return -np.inf

def log_marg_mainsequence_full(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    b1, b2, b3, lnb, r1, r2, r3, lnr, alpha, beta, zeta = params
    x, y, xerr, yerr =  GAMA_data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # passive fraction
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    c = 1 + np.tanh(zeta)
    f_pass = c + ((1-c)/(1+np.power(np.power(10,x-alpha), beta)))
    # assume R = 1
    B = (1-f_pass)/f_pass
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # star forming likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_blue = np.square(xerr)*np.square((2*b1*x) + b2) + np.square(yerr) + np.exp(2*lnb)
    DeltaN_blue = y - (b1*x*x) - (b2*x) - b3
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # passive likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_red = np.square(xerr)*np.square((2*r1*x) + r2) + np.square(yerr) + np.exp(2*lnr)
    DeltaN_red = y - (r1*x*x) - (r2*x) - r3
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ll_all = np.sum(np.log((B/Sigma2_blue)*np.exp(-DeltaN_blue**2/(2*Sigma2_blue)) + (1/Sigma2_red)*np.exp(-DeltaN_red**2/(2*Sigma2_red))))
    return ll_all + log_mainsequence_priors_full(params)


def log_mainsequence_priors_full2(params):
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    if  -0.3 < b1 < 0.0 and \
        0.0 < b2 < 4.0 and \
        -16.0 < b3 < -10.0 and \
        -5.0 < lnb < 5.0 and \
        0.3 < r1 < 1.5 and \
        -9.0 < r2 < -5.5 and \
        -5.0 < lnr < 5.0 and \
        9.0 < alpha < 12.0 and \
        -4.0 < beta < 0.0 and \
        -7.0 < zeta < 0.0 and \
        0.6 < h1 < 1.0 and \
        8.0 < h2 < 11.0 and \
        -2.0 < lnh < 2.0:
        return 0
    return -np.inf

def log_mainsequence_priors_full1(params):
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta = params
    # if models.f_passive(6.0, alpha, beta, zeta) < 0.4:
    #     return -np.inf
    if  -1.0 < b1 < 1.0 and \
        0.0 < b2 < 3.0 and \
        -20.0 < b3 < -5.0 and \
        -5.0 < lnb < 5.0 and \
        0.0 < r1 < 1.5 and \
        -15.0 < r2 < -1.0 and \
        -5.0 < lnr < 5.0 and \
        9.0 < alpha < 12.0 and \
        -2.0 < beta < 0.0 and \
        -4.5 < zeta < 0.0:
        return 0
    return -np.inf


# def log_mainsequence_priors_full1(params):
#     b1, b2, lnb, r2, lnr, alpha, beta, zeta = params
#     mu5, s5 = -0.9, 0.01
#     mu6, s6 = -1.8, 0.1
#     mu7, s7 = -1.1, 0.1
#     mu0, s0 = 10.6, 0.1
#     mu1, s1 = -0.96, 0.1
#     mu2, s2 = -2.2, 0.1
#     mu3, s3 = 0.75, 0.04
#     mu4, s4 = -7.5, 0.5
#     return np.log(1.0/(np.sqrt(2*np.pi)*s0))-0.5*(alpha-mu0)**2/s0**2 + \
#     np.log(1.0/(np.sqrt(2*np.pi)*s1))-0.5*(beta-mu1)**2/s1**2 + \
#     np.log(1.0/(np.sqrt(2*np.pi)*s2))-0.5*(zeta-mu2)**2/s2**2 + \
#     np.log(1.0/(np.sqrt(2*np.pi)*s3))-0.5*(b1-mu3)**2/s3**2 + \
#     np.log(1.0/(np.sqrt(2*np.pi)*s4))-0.5*(b2-mu4)**2/s4**2 + \
#     np.log(1.0/(np.sqrt(2*np.pi)*s5))-0.5*(lnb-mu5)**2/s5**2 + \
#     np.log(1.0/(np.sqrt(2*np.pi)*s6))-0.5*(r2-mu6)**2/s6**2 + \
#     np.log(1.0/(np.sqrt(2*np.pi)*s7))-0.5*(lnr-mu7)**2/s7**2


def log_marg_mainsequence_full1(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta = params
    x, y, xerr, yerr = GAMA_data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # total likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DeltaN_b = y - (b1*x*x) - (b2*x) - b3
    DeltaN_r = y - (r1*x) - r2
    Sigma2_b = np.square(xerr)*np.square((2*b1*x) + b2) + np.square(yerr) + np.exp(2*lnb)
    Sigma2_r = np.square(xerr)*np.square(r1) + np.square(yerr) + np.exp(2*lnr)
    c = .5*(1+np.tanh(zeta))
    f_pass = (c + ((1-c)/(1+np.power(np.power(10,x-alpha), beta))))
    # print ('N', N)
    blue_part = ((1-f_pass)/np.sqrt(2*np.pi*Sigma2_b))*np.exp(-np.square(DeltaN_b)/(2*Sigma2_b))
    red_part = (f_pass/np.sqrt(2*np.pi*Sigma2_r))*np.exp(-np.square(DeltaN_r)/(2*Sigma2_r))
    ll_sfrm = np.sum(np.log(blue_part + red_part))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return ll_sfrm + log_mainsequence_priors_full1(params)

def log_marg_mainsequence_full_SDSS(params, x, y, xerr, yerr):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta = params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # total likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DeltaN_b = y - (b1*x*x) - (b2*x) - b3
    DeltaN_r = y - (r1*x) - r2
    Sigma2_b = np.square(xerr)*np.square((2*b1*x) + b2) + np.square(yerr) + np.exp(2*lnb)
    Sigma2_r = np.square(xerr)*np.square(r1) + np.square(yerr) + np.exp(2*lnr)
    c = .5*(1+np.tanh(zeta))
    f_pass = (c + ((1-c)/(1+np.power(np.power(10,x-alpha), beta))))
    # print ('N', N)
    blue_part = ((1-f_pass)/np.sqrt(2*np.pi*Sigma2_b))*np.exp(-np.square(DeltaN_b)/(2*Sigma2_b))
    red_part = (f_pass/np.sqrt(2*np.pi*Sigma2_r))*np.exp(-np.square(DeltaN_r)/(2*Sigma2_r))
    ll_sfrm = np.sum(np.log(blue_part + red_part))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return ll_sfrm + log_mainsequence_priors_full1(params)

def log_mainsequence_priors_full1b(params):
    b1, b2, b3, lnb, r2, lnr, alpha, beta, zeta = params
    # if models.f_passive(6.0, alpha, beta, zeta) < 0.4:
    #     return -np.inf
    if  -1.0 < b1 < 1.0 and \
        0.0 < b2 < 3.0 and \
        -20.0 < b3 < -5.0 and \
        -5.0 < lnb < 5.0 and \
        -4.0 < r2 < -1.5 and \
        -5.0 < lnr < 5.0 and \
        9.0 < alpha < 12.0 and \
        -2.0 < beta < 0.0 and \
        -4.5 < zeta < 0.0:
        return 0
    return -np.inf

def log_marg_mainsequence_full1b(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    b1, b2, b3, lnb, r2, lnr, alpha, beta, zeta = params
    x, y, xerr, yerr = GAMA_data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # total likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DeltaN_b = y - (b1*x*x) - (b2*x) - b3
    DeltaN_r = y - (b1*x*x) - (b2*x) - b3 - r2
    Sigma2_b = np.square(xerr)*np.square((2*b1*x) + b2) + np.square(yerr) + np.exp(2*lnb)
    Sigma2_r = np.square(xerr)*np.square((2*b1*x) + b2) + np.square(yerr) + np.exp(2*lnr)
    c = .5*(1+np.tanh(zeta))
    f_pass = (c + ((1-c)/(1+np.power(np.power(10,x-alpha), beta))))
    # print ('N', N)
    blue_part = ((1-f_pass)/np.sqrt(2*np.pi*Sigma2_b))*np.exp(-np.square(DeltaN_b)/(2*Sigma2_b))
    red_part = (f_pass/np.sqrt(2*np.pi*Sigma2_r))*np.exp(-np.square(DeltaN_r)/(2*Sigma2_r))
    ll_sfrm = np.sum(np.log(blue_part + red_part))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return ll_sfrm + log_mainsequence_priors_full1b(params)

def log_marg_mainsequence_full2(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    x, y, yerr = passive_data
    sfr, n = sfr_hist_data
    xb, yb, xerrb, yerrb = GAMA_sforming
    xr, yr, xerrr, yerrr = GAMA_passive
    xt, yt, xerrt, yerrt = GAMA_data
    x1, x2, y1, y2, S1, S2 = GASS_data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # passive fraction likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    c = .5*(1+np.tanh(zeta))
    Sigma2 = np.square(yerr)
    Delta = y - (c + ((1-c)/(1+np.power(np.power(10,x-alpha), beta))))
    ll_pass_frac = -0.5 * np.sum(Delta**2 / Sigma2 + np.log(Sigma2))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # star forming likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_sf = np.square(xerrb)*np.square((2*b1*xb) + b2) + np.square(yerrb) + np.exp(2*lnb)
    DeltaN = yb - (b1*xb*xb) - (b2*xb) - b3
    ll_sf = -0.5 * np.sum(DeltaN**2/Sigma2_sf + np.log(Sigma2_sf))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # passive likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_pass = np.square(xerrr)*np.square(r1) + np.square(yerrr) + np.exp(2*lnr)
    DeltaN = yr - (r1*xr) - r2
    ll_pass = -0.5 * np.sum(DeltaN**2/Sigma2_pass + np.log(Sigma2_pass))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood for the linear SFR-MHI plane
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = np.array([-h1, 1.0])
    deltaN = y1 - (h1 * x1) - h2
    model = (h1 * x2) + h2
    sigma = np.dot(np.dot(S1, v), v) + np.exp(2 * lnh)
    sigma2 = np.dot(np.dot(S2, v), v) + np.exp(2 * lnh)
    sigma2 = sigma2 ** 0.5
    ll1 = -0.5 * np.sum(np.square(deltaN) / sigma + np.log(sigma))
    I = np.zeros(len(x2))
    for i in range(0,len(x2)):
        I[i] = np.log(((2 * np.pi) ** 0.5) * 0.5 * (special.erf((y2[i]-model[i]) / ((2 ** 0.5) * sigma2[i])) + 1))
    ll2 = np.sum(I)
    LL_SFR_MHI = ll1  + ll2
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return ll_sf + ll_pass + ll_pass_frac + LL_SFR_MHI + log_mainsequence_priors_full2(params)

def log_marg_mainsequence_full3(params):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta = params
    x, y, yerr = passive_data
    sfr, n = sfr_hist_data
    xb, yb, xerrb, yerrb = GAMA_sforming
    xr, yr, xerrr, yerrr = GAMA_passive
    xt, yt, xerrt, yerrt = GAMA_data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # passive fraction likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    c = .5*(1+np.tanh(zeta))
    Sigma2 = np.square(yerr)
    Delta = y - (c + ((1-c)/(1+np.power(np.power(10,x-alpha), beta))))
    ll_pass_frac = -0.5 * np.sum(Delta**2 / Sigma2 + np.log(Sigma2))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # star forming likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_sf = np.square(xerrb)*np.square((2*b1*xb) + b2) + np.square(yerrb) + np.exp(2*lnb)
    DeltaN = yb - (b1*xb*xb) - (b2*xb) - b3
    ll_sf = -0.5 * np.sum(DeltaN**2/Sigma2_sf + np.log(Sigma2_sf))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # passive likelihood
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sigma2_pass = np.square(xerrr)*np.square(r1) + np.square(yerrr) + np.exp(2*lnr)
    DeltaN = yr - (r1*xr) - r2
    ll_pass = -0.5 * np.sum(DeltaN**2/Sigma2_pass + np.log(Sigma2_pass))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # log likelihood delta MHI
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # make the means, passive fractions for this set of params
    b_mean = (b1*xt*xt) + (b2*xt) + b3
    r_mean = (r1*xt) + r2
    f_pass = models.f_passive(xt, alpha, beta, zeta)
    rand = np.random.uniform(0, 1, len(xt))
    sfrs = []
    for idx, element in enumerate(b_mean):
        if rand[idx] <= f_pass[idx]:
            sfrs.append(r_mean[idx])
        else:
            sfrs.append(b_mean[idx])
    n2, sfr5 = np.histogram(sfrs, sfr_bins)
    delta = n - n2
    ll_delta_sfr = -0.5 * np.sum(delta**2/np.sqrt(n) + np.log(np.sqrt(n)))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return ll_sf + ll_pass + ll_pass_frac + ll_delta_sfr + log_mainsequence_priors_full2(params)
