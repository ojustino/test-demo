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
def poly(x,a,plot=False,sigma=0): 
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

def logL(N,D,mu,sigma=0): # N is ___, D is the guesses array, mu is the 'truth.'
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
    print likely
    likelynew = np.exp(likely-likely.max())

    #plt.contourf(a0,a1,likely)
    plt.contourf(a0,a1,likelynew)
    plt.show()

    #lines = [model(x,1.,2.,sigma=.2) for i in range(100)]

    # to marginalize: sum one parameter over one value of the other (like take all values of a0 with the same a1 -- just one COLUMN of the likelihood array), then plot the resulting curve. see Maurice for more...?

fit_model(.1)


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
