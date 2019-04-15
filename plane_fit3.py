import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import emcee
import corner
from itertools import product
from astropy.io import fits
from astropy.table import Table
from scipy.integrate import quad, dblquad, nquad
import pandas as pd

def second_order(x, a1, a2, a3):
    return (a1*x*x) + (a2*x) + a3

def integrand_MHI_blue1(M, SFR, MHI, *params):
    b1, b2, b3, lnb, r1, r2, lnr, alpha, beta, zeta, h1, h2, lnh = params
    lnh
    Mstar, phistar1, alpha1 = 10.72, 0.71E-3, -1.45
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))
    f = (b1*M*M) + (b2*M) + b3
    P_SFR_given_Mstar = (1/np.sqrt(2*np.pi*np.power(np.exp(lnb),2)))*np.exp((-1/(2*np.power(np.exp(lnb),2)))*np.power((SFR-f),2))
    f2 = (h1*SFR) + h2
    P_MHI_given_SFR = (1/np.sqrt(2*np.pi*np.power(np.exp(lnh),2)))*np.exp((-1/(2*np.power(np.exp(lnh),2)))*np.power((MHI-f2),2))
    return phi_Mstar_double*P_SFR_given_Mstar*P_MHI_given_SFR

def integrand_MHI_blue(M, SFR, MHI, params):
    m1, m2, b, lnl, b1, b2, b3, lnb = params
    Mstar, phistar1, alpha1 = 10.72, 0.71E-3, -1.45
    phi_Mstar_double = np.log(10) * np.exp(-np.power(10,M-Mstar)) * (phistar1*np.power(10,(alpha1+1)*(M-Mstar)))
    f = (b1*M*M) + (b2*M) + b3
    P_SFR_given_Mstar = (1/np.sqrt(2*np.pi*np.power(np.exp(lnb),2)))*np.exp((-1/(2*np.power(np.exp(lnb),2)))*np.power((SFR-f),2))
    MHI_mean = (m1*M) + (m2*SFR) + b
    P_MHI_given_SFR_and_Mstar = (1/np.sqrt(2*np.pi*np.power(np.exp(lnl),2)))*np.exp((-1/(2*np.power(np.exp(lnl),2)))*np.power((MHI-MHI_mean),2))
    return phi_Mstar_double*P_SFR_given_Mstar*P_MHI_given_SFR_and_Mstar

def read_GASS():
    xxGASS = fits.open('data/xxGASS_MASTER_CO_170620_final.fits')
    xxGASS = Table(xxGASS[1].data).to_pandas()
    xxGASS = xxGASS[xxGASS['SFR_best'] > -80]

    xxGASS_final = fits.open('data/xGASS_RS_final_Serr_180903.fits')
    xxGASS_final = Table(xxGASS_final[1].data).to_pandas()

    xxGASS['SNR'] = xxGASS_final['SNR']
    xxGASS['MHI_err'] = np.power(10, xxGASS['lgMHI'])/xxGASS['SNR']
    xxGASS['lgMHI_err'] = xxGASS['MHI_err']/(np.power(10,xxGASS['lgMHI'])*np.log(10))
    xxGASS['lgMstar_err'] = 0.0

    # print (xxGASS.columns)
    # print (xxGASS[['HIar_flag', 'Gdcode', 'GASSDR', 'zHI', 'W50cor', 'lgMHI_old', 'lgMHI', 'lgGF', 'HIconf_flag']])
    data = xxGASS[['SFR_best', 'lgMHI', 'lgMHI_err', 'lgMstar', 'SFRerr_best', 'HIsrc', 'HIconf_flag']]
    data = data[np.isfinite(data['lgMstar'])]
    data = data[np.isfinite(data['SFR_best'])]
    data = data[np.isfinite(data['lgMHI'])]
    data = data[np.isfinite(data['SFRerr_best'])]
    data = data[np.isfinite(data['lgMHI_err'])]
    # det = data[data['HIsrc']!=4]
    # nondet = data[data['HIsrc']==4]
    data['SFRerr_best'] = data['SFRerr_best']/(data['SFR_best']*np.log(10))
    data['SFR_best'] = np.log10(data['SFR_best'])
    # data = data[data['SFR_best'] >= second_order(data['lgMstar'], -0.06, +1.95, -14.5)]

    det = data[data['HIconf_flag']==0]
    nondet = data[data['HIconf_flag']==-99]
    return data, det, nondet

# A helper function to make the plots with error ellipses
def plot_error_ellipses(ax, X, S, color="k"):
    for n in range(len(X)):
        vals, vecs = np.linalg.eig(S[n])
        theta = np.degrees(np.arctan2(*vecs[::-1, 0]))
        w, h = 2 * np.sqrt(vals)
        ell = Ellipse(xy=X[n], width=w, height=h,
                      angle=theta, color=color, lw=0.5)
        ell.set_facecolor("none")
        ax.add_artist(ell)
    ax.plot(X[:, 0], X[:, 1], ".", color=color, ms=4)

def plot_samples_4(sampler, ndim, fname):
    fig, axes = plt.subplots(ndim, figsize=(10, 20), sharex=True)
    samples = sampler.get_chain()
    labels = ['m1', 'm2', 'm3', 'm4', 'b', 'lnl']
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    plt.savefig('img/sampler' + fname + '.pdf')

def plot_dataset(X_true, X, samples_D, fname):
    # Plot the simulated dataset.
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    for xi, yi in product(range(3), range(3)):
        if yi <= xi:
            continue
        ax = axes[yi-1, xi]
        plot_error_ellipses(ax, X_true[:, [xi, yi]], X[:,[[xi, xi], [yi, yi]],[[xi, yi], [xi, yi]]])

        # ax.set_xlim(-7, 17)
        # ax.set_ylim(-7, 17)
    Mstar = np.linspace(9,12,100)
    SFR = np.linspace(-2,1,100)
    # for params in samples_D[np.random.randint(len(samples_D), size=10)]:
    #     m1, m2, b, lnl = params
    #     axes[0,0].plot(Mstar, ((-m1/m2)*Mstar) -( b/m2), color = 'g', alpha = 0.1)
    #     axes[1,0].plot(Mstar, (m1*Mstar) + b, color = 'g', alpha = 0.1)
    #     axes[1,1].plot(SFR, (m2*SFR) + b, color = 'g', alpha = 0.1)
    # Make the plots look nicer...
    ax = axes[0, 1]
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 0]
    ax.set_ylabel("$SFR$")
    ax.set_xticklabels([])
    ax = axes[1, 0]
    ax.set_xlabel("$M_{*}$")
    ax.set_ylabel("$M_{HI}$")
    ax = axes[1, 1]
    ax.set_xlabel("$SFR$")
    ax.set_yticklabels([])
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(fname)
    # plt.show()

np.random.seed(42)

# # Generate the true coordinates of the data points.
# N = 50
# m_true = np.array([1.2, -0.3])
# b_true = -0.1
# X_true = np.empty((N, 3))
# X_true[:, 0] = np.random.uniform(0, 10, N)
# X_true[:, 1] = np.random.uniform(0, 10, N)
# X_true[:, 2] = np.dot(X_true[:, :-1], m_true) + b_true
# X = np.empty((N, 3))

# Generate error ellipses and add uncertainties to each point.
# S = np.zeros((N, 3, 3))
# for n in range(N):
#     L = np.zeros((3, 3))
#     L[np.diag_indices_from(L)] = np.exp(np.random.uniform(-1, 1))
#     L[np.tril_indices_from(L, -1)] = 0.5 * np.random.randn()
#     S[n] = np.dot(L, L.T)
#     X[n] = np.random.multivariate_normal(X_true[n], S[n])

# Finally add some scatter.
# lambda_true = 2.0
# X[:, -1] += lambda_true * np.random.randn(N)

xxGASS, det, nondet = read_GASS()

# det = xxGASS[xxGASS['HIconf_flag']==0]
# nondet = xxGASS[xxGASS['HIconf_flag']==-99]

# x is Mstar, y is SFR, z is MHI
N = len(xxGASS)
X_true = np.empty((N, 3))
X_true[:,0] = xxGASS['lgMstar'].values
X_true[:,1] = xxGASS['SFR_best'].values
X_true[:,2] = xxGASS['lgMHI'].values
# Generate error ellipses and add uncertainties to each point.
S = np.zeros((N, 3, 3))
yerr = xxGASS['SFRerr_best'].values
zerr = xxGASS['lgMHI_err'].values
for n in range(N):
    L = np.zeros((3, 3))
    L[0,0] = np.square(0.1)
    L[1,1] = np.square(yerr[n])
    L[2,2] = np.square(zerr[n])
    S[n] = L
    # X[n] = np.random.multivariate_normal(X_true[n], S[n])

# global Xdata
# global Sdata
# Xdata = X
# Sdata = S

X = X_true

def log_prob_priors1(params):
    m = params[:2]
    b, log_lambda = params[2:]
    if  -2.0 < m[0] < 2.0 and \
        -2.0 < m[1] < 2.0 and \
        -5.0 < b < 20.0 and \
        -2.0 < log_lambda < 2.0:
        return 0
    return -np.inf

def log_prob_D1(params):
    # print (params)
    m = params[:2]
    b, log_lambda = params[2:]
    v = np.append(-m, 1.0)

    # Compute \Sigma^2 and \Delta.
    Sigma2 = np.dot(np.dot(S, v), v) + np.exp(2*log_lambda)
    Delta = np.dot(X, v) - b

    # Compute the log likelihood up to a constant.
    ll = -0.5 * np.sum(Delta**2 / Sigma2 + np.log(Sigma2)) + log_prob_priors1(params)
    # print (ll)
    return ll

def log_prob_priors(params):
    m = params[:3]
    b, log_lambda = params[3:]
    # if  0 < m[0] < 1.0 and \
    #     0.0 < m[1] < 1.0 and \
    #     -0.5 < m[2] < 0.5 and \
    #     -20.0 < b < 8.0 and \
    if -2.0 < log_lambda < 2.0:
        return 0
    return -np.inf

def log_prob_D(params):
    # print (params)
    m = params[:3]
    b, log_lambda = params[3:]
    # v1 = np.array([2*m[0]*X[:,0] + m[1], 2*m[2]*X[:,1] + m[3]])
    # print
    # v = np.append(-v1, 1.0)
    # print (np.shape(v))

    # Compute \Sigma^2 and \Delta.
    Sigma2 = (np.square(m[0] + m[2]*X[:,1])*np.square(0.1)) + (np.square(m[1] + m[2]*X[:,0])*np.square(yerr)) + np.square(zerr) + np.exp(2*log_lambda)
    # print (Sigma2)
    Delta = X[:,2] - (m[0]*X[:,0] + m[1]*X[:,1] + m[2]*X[:,0]*X[:,1] + b)
    # print (Delta)
    # Compute the log likelihood up to a constant.
    ll = -0.5 * np.sum(Delta**2 / Sigma2 + np.log(Sigma2)) + log_prob_priors(params)
    # print (ll)
    return ll



m_true = np.array([1.2, -0.3])
b_true = -10.0
lambda_true = 0.5
# Run the MCMC.
# nwalkers, ndim = 100, 5
nwalkers, ndim = 100, 4
sampler_D = emcee.EnsembleSampler(nwalkers, ndim, log_prob_D1)
# p0 = [0.3, 0.3, 0.1, 6.5, -1]
p0 = [0.3, 0.9, 6.5, -1]
p0 = p0 + 1e-4 * np.random.randn(nwalkers, len(p0))
pos, _, _ = sampler_D.run_mcmc(p0, 2000, progress = True)
# sampler_D.reset()
# sampler_D.run_mcmc(pos, 5000)
plot_samples_4(sampler_D, ndim, 'plane')
samples_D = sampler_D.chain[:, 1500:, :].reshape((-1, ndim))
plot_dataset(X_true, S, samples_D, 'img/dataset.pdf')
# samples_D = samples_D.flatchain

# tau = sampler_D.get_autocorr_time(c=4)
# nsamples = len(samples_D)
# print("{0:.0f} independent samples of m1".format(nsamples / tau[0]))
# print("{0:.0f} independent samples of m2".format(nsamples / tau[1]))
# print("{0:.0f} independent samples of b".format(nsamples / tau[2]))
# print("{0:.0f} independent samples of ln(lambda)".format(nsamples / tau[3]))
#
# corner.corner(samples_D, labels=["$m_1$", "$m_2$", "b", "$\ln\lambda$"],
#               truths=np.append(m_true, [b_true, np.log(lambda_true)]));
# plt.show()

m1 = np.mean(samples_D[:, 0])
m2 = np.mean(samples_D[:, 1])
b = np.mean(samples_D[:, 2])
lnl = np.mean(samples_D[:, 3])
print (m1, m2, b, lnl)
#
# m1, m2, b, lnl = 0.27, 0.29, 6.5, -1.01

# plt.figure()
# ax = plt.subplot(111, projection='3d')
# ax.scatter(X_true[:, 0], X_true[:, 1], X_true[:, 2], color='b')
#
# # plot plane
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# X,Y = np.meshgrid(np.arange(xlim[0], xlim[1], 0.1),
#                   np.arange(ylim[0], ylim[1], 0.1))
# Z = np.zeros(X.shape)
# Z2 = np.zeros(X.shape)
# for r in range(X.shape[0]):
#     print (r)
#     for c in range(X.shape[1]):
#         Z[r,c] = m1 * X[r,c] + m2 * Y[r,c] + b
#         Z2[r,c] = (m1 * X[r,c]) + (m2 * Y[r,c]) + (0.01*X[r,c]*Y[r,c]) + b
#
#
# ax.plot_wireframe(X,Y,Z, color='k')
# # ax.plot_wireframe(X,Y,Z+np.exp(lnl), color='b')
# # ax.plot_wireframe(X,Y,Z-np.exp(lnl), color='b')
# # ax.plot_wireframe(X,Y,Z2, color='r')
# ax.set_xlabel('log M*')
# ax.set_ylabel('log SFR')
# ax.set_zlabel('log MHI')
# plt.show()


N, n = 20, 20
best_fits = np.zeros((N,n))
best_fits1 = np.zeros((N,n))
# load the converged main sequence fits from previous work
samples4 = np.loadtxt('data/samples4.txt')
samples5 = samples4[:50000,:4]

print (np.shape(samples5))
print (np.shape(samples_D))
samples_D = np.hstack((samples_D, samples5))
print (np.shape(samples_D))
MHI = np.linspace(6,12,n)
i = 0
for params in samples_D[np.random.randint(len(samples_D), size = N)]:
    for idx, element in enumerate(MHI):
        # # m1, m2, b, lnl = params
        # print (i, idx)
        best_fits[i,idx] = dblquad(integrand_MHI_blue, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, params, ))[0]
    i+=1
i = 0
for params in samples4[np.random.randint(len(samples4), size = N)]:
    for idx, element in enumerate(MHI):
        # # m1, m2, b, lnl = params
        # print (i, idx)
        best_fits1[i,idx] = dblquad(integrand_MHI_blue1, -5.0, 2.0, lambda SFR: 0.0, lambda SFR: 12.0, args = (element, *params))[0]
    i+=1
fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze=False, figsize=(6,6))
for i in range(0,N):
    ax[0,0].plot(MHI, np.log10(best_fits[i,:]), color = 'g', alpha = 0.1)
    ax[0,0].plot(MHI, np.log10(best_fits1[i,:]), color = 'r', alpha = 0.1)

# read in the ALFA ALFA datasets from the 40% paper
ALFAALFA = pd.read_csv('ALFAALFA.csv', comment = '#', header = None, sep=",")
ALFAALFA.columns = ['x', 'y', 'dy', 'MHI', 'phi', 'err', 'phi_err']
ALFAALFA = ALFAALFA[np.isfinite(ALFAALFA['phi_err'])]
MHI_alfa, phi_alfa, phi_err_alfa = np.round(ALFAALFA['MHI'].values,2), ALFAALFA['phi'].values, ALFAALFA['phi_err'].values
ax[0,0].errorbar(MHI_alfa, phi_alfa, yerr = phi_err_alfa, fmt='o', capsize = 2, markersize = 3, linewidth=2, markeredgewidth=2, capthick=2, mfc='gray', mec='gray', ecolor = 'gray')
ax[0,0].set_ylim(-6,0)
ax[0,0].set_xlim(6,11)
ax[0,0].set_xlabel('log MHI')
ax[0,0].set_ylabel('phi(MHI)')
plt.savefig('img/MHI_with_plane.pdf')
