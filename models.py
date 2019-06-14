import numpy as np

def Gaussian_Conditional_Probability(x, mean, sigma):
    sigma = np.exp(sigma)
    return (1/np.sqrt(2*np.pi*np.power(sigma,2)))*np.exp((-1/(2*np.power(sigma,2)))*np.power((x-mean),2))

def f_passive(x, a, b, zeta):
    c = 1 + np.tanh(zeta)
    return c + ((1-c)/(1+np.power(np.power(10,x-a), b)))

def Saintonge16_MS(x):
    return (-2.332*x) + (0.4156*x*x) - (0.01828*x*x*x)

def first_order(x, a1, a2):
    return (a1*x) + a2

def second_order(x, a1, a2, a3):
    return (a1*x*x) + (a2*x) + a3

def third_order(x, a1, a2, a3, a4):
    return (a1*x*x*x) + (a2*x*x) + (a3*x) + a4

def fourth_order(x, a1, a2, a3, a4, a5):
    return (a1*x*x*x*x) + (a2*x*x*x) + (a3*x*x) + (a4*x) + a5

def gauss(x,a1,mu1,sigma1):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2))

def double_gauss(x,a1,mu1,sigma1,a2,mu2,sigma2):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + a2*np.exp(-(x-(mu2))**2/(2*sigma2**2))

def triple_gauss(x,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3):
    return a1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + \
            a2*np.exp(-(x-(mu2))**2/(2*sigma2**2)) + \
            a3*np.exp(-(x-(mu3))**2/(2*sigma3**2))

def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z
