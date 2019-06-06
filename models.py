import numpy as np

def Gaussian_Conditional_Probability(x, mean, sigma):
    sigma = np.exp(sigma)
    return (1/np.sqrt(2*np.pi*np.power(sigma,2)))*np.exp((-1/(2*np.power(sigma,2)))*np.power((x-mean),2))

def f_passive(x, a, b, zeta):
    c = 1 + np.tanh(zeta)
    return c + ((1-c)/(1+np.power(np.power(10,x-a), b)))
