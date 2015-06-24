import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

def normpdf(A,sigma,mu):
    prob = 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-1./2 * ((A-mu)**2))
    return prob

def model(x,a_0,a_1,sigma=0): # x is a numpy array, a_0 and a_1 are preferrably floats
    line = a_0 + a_1*x
    points = len(line)
    x = np.arange(points)
    
    z = 0 
    while(z < points):
        line[z] += sigma*rand.randn() 
        z += 1

    plt.plot(np.arange(points),line,color='#00245D')

    errs = sigma * rand.randn(points)
    #print(np.shape(x), np.shape(line), np.shape(errs))
    plt.errorbar(np.arange(points),line,yerr=errs,fmt='.',ecolor='#F1AA00')
    plt.xlim(-.5,2.5)
    plt.show()
    return line

#model(np.array([1,2,3]),-1.,2.,6.5)

def poly(x,a,plot=False,sigma=0): # a is a matrix of coefficients for x, which is also an array. you will sum over a_i * x^ifrom 0 to N-1
    xsize = float(len(x))
    asize = float(len(a))
    xarray = np.resize(x, (xsize,asize))
    aarray = np.resize(a, (asize,xsize))
    
    powers = np.zeros(xsize)
    b = 0
    while(b < 0):
        powers[a] = float(b)
        b += 1

    y = xarray**powers * aarray.T
    # plotting sum of each column against x
    ysums = np.sum(y,axis=0)

    if plot:
        plt.plot(x,ysums)
        plt.show()

poly([1,2],[7,5,1],plot=True)

# datum 1
sigma = 2.
mu = np.arange(20,60.5,.5)
A1 = 41.4
prob1 = normpdf(A1,sigma,mu) #/max(normpdf(A1,sigma,mu))
one = plt.plot(prob1,linewidth=6,alpha=.45,color='#BB9753')

# datum 2
sigma = 3.
mu = np.arange(20,60.5,.5)
A2 = 46.9
prob2 = normpdf(A2,sigma,mu) #/max(normpdf(A2,sigma,mu))
two = plt.plot(prob2,linewidth=6,alpha=.45,color='#008348')

# datum 3
sigma = 6.1
mu = np.arange(20,60.5,.5)
A3 = 44.1
prob3 = normpdf(A3,sigma,mu) #/max(normpdf(A3,sigma,mu))
three = plt.plot(prob3,linewidth=6,alpha=.45,color='#C60C30')

plt.title('Gaussians')
plt.xlabel('mu')
plt.ylabel('probability')
plt.legend((one,two,three),('datum 1','datum 2','datum 3'),loc='upper right',shadow=True)
plt.show()

plt.plot(prob1*prob2*prob3,linewidth=6,alpha=.45,color='#FAB383')

plt.title('Gaussians')
plt.xlabel('mu')
plt.ylabel('probability')
plt.show()


