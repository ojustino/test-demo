import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import stats as s

N = 4
mu = np.linspace(0,10,200)

P = mu**N * np.exp(-mu)/misc.factorial(N)



plt.plot(mu,P)
#plt.show()

x = np.linspace(1,6,6)
D = np.array([74,15,11,4,3,1])
degree = 3

yo = s.polynices(x,D,degree,sigma=0)
print yo

# probability of these coefficients being correct is product of Poisson likelihoods)

def poiL(N,D,mu):
    loglike = np.log(mu**N * np.exp(-mu)/misc.factorial(N))
    return loglike

# for lev-marq

# data is already made (D)

def residuals():
    model = D
    
