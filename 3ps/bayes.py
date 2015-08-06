import numpy as np
import matplotlib.pyplot as plt

'''data should be whether or not a star had a companion.
so d is an a array with three 1s and 18 0s.'''

'''f = .8
d_i = np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # for 21 stars

N = 0; k = 0
j = 0
while(j < len(d_i)):
    N += 1
    if d_i[j] == 1:
        k += 1
    j += 1

print k, N

prob_d = f**k * (1-f)**(N-k)
#probability of this data is f**3 * (1-f)**18

x = np.linspace(0,1,500) # the fraction of stars with a companion
sigma = 1./15
prior = np.exp(-1./2*((.8-x)/sigma)**2)
#prior = np.ones(len(x))

bayes = prob_d * prior

#plt.plot(x,bayes)
#plt.plot(x,prior)
#plt.show()'''


 # f is your fraction, k is # of relevant events, N is total # of events
def thehood(f,k,N):
    likely = f**k * (1-f)**(N-k)
    return likely

f1 = np.genfromtxt('ps3_q4data.txt')
mets = f1[:,1]
planet = f1[:,2]

metsP  = [] # has companion
metsNP = [] # does not have companion

d = 0.; M = 0.
j = 0
while(j < len(planet)):
    M += 1
    if planet[j] == 1:
        d += 1
        metsP.append(mets[j])
    else:
        metsNP.append(mets[j])
    j += 1

metsP  = np.array(metsP)
metsNP = np.array(metsNP)

alpha = np.linspace(0,100,501); lena = len(alpha)
beta  = np.linspace(0,100,501); lenb = len(beta)

likely = np.zeros([lena,lenb])

j = 0; k = 0
while(j < lena):
    while(k < lenb):
        f     = alpha[j] * 10**(beta[k]*metsP)
        f_not = alpha[j] * 10**(beta[k]*metsNP)
        # THEY'RE SEPARATE THINGS
        likely[j][k] = thehood()
        k += 1
    j += 1

prior_a = 1
prior_b = 1 # i guess we're ignoring these for now -- uninformative priors
# uninformative, drunk priors

#alpha = ; beta = 
#params = [alpha,beta]

'''flaco = np.array(sorted(f1[:,1]))
plt.plot(10**flaco,)
plt.plot(np.linspace(0,len(f1[:,1]),len(f1[:,1])),)
plt.show()'''




