import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

#plt.ion()

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

    plt.plot(x,line,color='#00245D')

    errs = sigma * rand.randn(points)
    #print(np.shape(x), np.shape(line), np.shape(errs))
    plt.clf() # this, in tandem with plt.ion() at top, allows instant plotting
    plt.errorbar(x,line,yerr=errs,fmt='.',ecolor='#F1AA00')
    plt.xlim(min(x)-.5,max(x)+.5)
    plt.show()
    return line

#model(np.array([1.,2.,3.,4.,5.]),1.,2.,sigma=0.)

# poly() returns len(x) points on a line whose polynomial equation has coefficients 'a' and has len(a)-1 as its highest degree. can add noise with a sigma term.
def poly(x,a,sigma=0.,plot=False): 
    xsize = float(len(x))
    asize = float(len(a))
    xarray = np.resize(x, (asize,xsize))
    aarray = np.resize(a, (xsize,asize))
    
    powers = np.arange(asize)
    parray = np.resize(powers, (xsize,asize))

    y = aarray.T * (xarray**parray.T)
    # plotting sum of each column against x
    ysums = np.sum(y,axis=0) + sigma*rand.randn(xsize)
    print xarray
    print aarray
    print y

    if plot:
        #fig = plt.figure(1)
        #plt.clf() 
        plt.scatter(x,ysums)
        #plt.(-10000,350000)
        plt.show()

    return ysums

#poly([1.,2.,3.,4.,5.],[1.,2.],plot=True)
#poly([1.,2.,3.,4.,5.],[2.,3.,-1.],plot=True)

def logL(N,D,mu,sigma=0): # N data points, D is measurements, mu is truth/model.
    constant = -1./2*N*np.log(2*np.pi*sigma**2)
    chi2s    = ((D-mu)/(sigma))**2
    chi2tot  = -1./2*np.sum(chi2s, axis=0)
    loglike  = constant + chi2tot  
    return loglike # should just be one number

def fit_model(sig):
    x = np.array([0.,1,2,3,4]) #0.,1,2,3,4
    a = np.array([1.,2])
    truth = poly(x,a,sigma=sig) # we're given that sigma is .5. DON'T TOUCH ME
    
    a0 = np.linspace(.85,1.15,101); len0 = len(a0) #31
    a1 = np.linspace(1.85,2.15,101); len1 = len(a1) #31

    j = 0; k = 0

    likely = np.zeros([len0,len1])

    while(j < len0):
        while(k < len1):
            D = poly(x,[a0[j],a1[k]]) # sigma should be 0.
            #print D
            likely[j][k] = logL(len(x),D,truth,sigma=sig) # N could also be len
            k += 1
        j += 1
        k = 0

    #like2 = np.exp(likely)
    #print likely
    likelynew = np.exp(likely-likely.max())

    #plt.contourf(a0,a1,likely)
    plt.contourf(a0,a1,likelynew)
    plt.show()

    # we don't really care about a1 (y intercept, so we marginalize it out. to marginalize: sum one parameter over one value of the other (like take all values of a0 with the same a1 -- just one COLUMN of the likelihood array), then plot the resulting curve.
    marglike = np.sum(likelynew,axis=1)
    plt.plot(a0,marglike,linewidth=3,linestyle='--',color='#85B7EA')
    plt.show()
    #lines = [model(x,1.,2.,sigma=.2) for i in range(100)]

#fit_model(.1)

# a more general chi squared minimization method. sigma constant for now
def polynices(x,D,degree,sigma=1.): # should take data. degree is a float now; it's order + 1 (i.e. 0th order means degree = 1.)
    points = len(x)
    xarray = np.resize(x, (degree, points))
    powers = np.arange(degree)
    parray = np.resize(powers, (points,degree))

    B = xarray**(parray.T)
    B = np.matrix(B.T) # now has dimensions len(x) by len(a).

    # just a test; degrees was an array here
    '''if D == 0:
        D = poly(x,degrees,sigma=.02) 
        D = np.matrix(D).T # now (supposedly) has dimensions a by 1.
    else:'''
    D += rand.randn(points)*sigma # this is 'noise'
    D = np.matrix(D).T # now (supposedly) has dimensions a by 1.
    #print D

    A = np.linalg.inv(B.T*B)*B.T*D
    # multiiplication is equivalent to dot product for 2D numpy matrices

    return A[0], A[1], A[2]

    # loop 'points' times. in each iteration, newdata = data, then newdata += noise, and newpars[i] = polynices(x,newdata,degree,sigma=sig).

def hists(X):
    x,D,degree,sigma = X
    
    i = 0
    points = len(x)
    trials = 1000
    newpars = np.zeros([trials,degree])

    while(i < trials):
        newdata = D + rand.randn(points)*sigma
        #print np.shape(newdata), np.shape(noise)
        newpars[i] = polynices(x,newdata,degree,sigma=sigma)
        i += 1

    print newpars

    j = 0
    while(j < degree):
        plt.hist(newpars[:,j],bins=50,color='#F0EBD2')
        plt.show()
        j += 1

# test all this
x   = np.array([-10.,-5,0,5,10])
points = len(x)
a   = np.array([-3.,11,4]) # TOGGLE. problem when first index is 1???
sig = np.array([.2,.5,3,0,.97]) 
data = poly(x,a,sig)
data += rand.randn(points) * sig # adding noise
degree = 3 # order of desired polynomial, PLUS THREE
X = [x,data,degree,sig]
hists(X)

'''# datum 1
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
plt.show()'''
