import numpy as np
import matplotlib.pyplot as plt

'''data should be whether or not a star had a companion.
so d is an a array with three 1s and 18 0s.'''

f = .8
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
'''probability of this data is f**3 * (1-f)**18'''

x = np.linspace(0,1,500) # the fraction of stars with a companion
sigma = 1./15
prior = np.exp(-1./2*((.8-x)/sigma)**2)
#prior = np.ones(len(x))

bayes = prob_d * prior

#plt.plot(x,bayes)
#plt.plot(x,prior)
#plt.show()







f1 = np.genfromtxt('ps3_q4data.txt')
mets = f1[:,1]
planet = f1[:,2]

d = 0.; M = 0.
j = 0
while(j < len(planet)):
    M += 1
    if planet[j] == 1:
        d += 1
    j += 1

f = d/M

L = f**d * (1-f)**(M-d)
prior_a = np.linspace(0,1,500)
prior_b = np.linspace(0,1,500)

prob = L*prior_a*prior_b
plt.plot(prob)
plt.show()


'''flaco = np.array(sorted(f1[:,1]))
plt.plot(10**flaco,)
plt.plot(np.linspace(0,len(f1[:,1]),len(f1[:,1])),)
plt.show()'''




