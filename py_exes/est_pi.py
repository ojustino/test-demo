import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import matplotlib.axes as ax

# ARRAYS ARE (supposedly) FASTER THAN LOOPS, MOST OF THE TIME

def est_pi(N,R): # takes in number of points and radius of circle
    a = 0
    inside = 0.; outside = 0.  # counters for point locations
    while(a < N):
        x = R*np.random.rand()
        y = R*np.random.rand()

        # if condition to determine whether or not point is in circle
        if x**2 + y**2 <= R**2:
            inside += 1
        else:
            outside += 1

        a += 1

    return 4.*(inside/(inside+outside))

def est_pi2(N,R):
    #np.random.seed(3) # makes random generator give same number each time
    x = R*rand.random(N); y = R*rand.random(N)
    M = x**2 + y**2 <= R**2 # boolean array with same length as x and y based on conditional
    M_in = float(len(x[M]))
    pi_est = 4.*M_in/len(x) # could've also been len(y) at bottom... or N

    plt.scatter(x,y,marker='x',alpha=.25,color='#1C105E')
    plt.plot([0,0],[0,R],linewidth=3,linestyle='--',color='#000000')
    plt.plot([0,R],[R,R],linewidth=3,linestyle='--',color='#000000')
    plt.plot([R,R],[R,0],linewidth=3,linestyle='--',color='#000000')
    plt.plot([R,0],[0,0],linewidth=3,linestyle='--',color='#000000')
    circle = plt.Circle((0,0),R,alpha=.5,linewidth=10,color='#E65F20')
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    plt.title('Dart Locations')
    plt.xlim(-.99,.99)
    plt.xlim(0-.1,R+.1)
    plt.ylim(0-.1,R+.1)
    plt.show()

    return pi_est

a1 = np.zeros(1000); b1 = np.zeros(1000); c1 = np.zeros(1000)
d1 = np.zeros(1000); e1 = np.zeros(1000)

a1 = [est_pi2(1e3,10) for i in range(100)]
plt.hist(a1,bins=10,histtype='barstacked',alpha=.2,linewidth=7,color='#B71234')
plt.title('Pi Estimate Distribution, N = 1e3')
plt.xlabel('Estimate')
plt.ylabel('Frequency')
plt.show()

b1 = [est_pi2(1e4,10) for i in range(100)]
plt.hist(a1,bins=10,histtype='barstacked',alpha=.2,linewidth=7,color='#B71234')
plt.title('Pi Estimate Distribution, N = 1e4')
plt.xlabel('Estimate')
plt.ylabel('Frequency')
plt.show()

c1 = [est_pi2(1e5,10) for i in range(100)]
plt.hist(a1,bins=10,histtype='barstacked',alpha=.2,linewidth=7,color='#B71234')
plt.title('Pi Estimate Distribution, N = 1e5')
plt.xlabel('Estimate')
plt.ylabel('Frequency')
plt.show()

d1 = [est_pi2(1e6,10) for i in range(100)]
plt.hist(a1,bins=10,histtype='barstacked',alpha=.2,linewidth=7,color='#B71234')
plt.title('Pi Estimate Distribution, N = 1e6')
plt.xlabel('Estimate')
plt.ylabel('Frequency')
plt.show()

e1 = [est_pi2(1e7,10) for i in range(100)]
plt.hist(a1,bins=10,histtype='barstacked',alpha=.2,linewidth=7,color='#B71234')
plt.title('Pi Estimate Distribution, N = 1e7')
plt.xlabel('Estimate')
plt.ylabel('Frequency')
plt.show()


'''# if you want a plot of a single result from a trial at each N, remove the loops and call each function twice, then uncomment this
e3 = est_pi2(1e3,10)
e4 = est_pi2(1e4,10)
e5 = est_pi2(1e5,10)
e6 = est_pi2(1e6,10)
e7 = est_pi2(1e7,10)
e8 = est_pi2(1e8,10)


estimates = np.array([e3,e4,e5,e6,e7,e8])
plt.plot(estimates,linewidth=3,color='#000000')
plt.xticks(np.arange(7),['1e3','1e4','1e5','1e6','1e7','1e8'])
plt.plot([-100,100],[3.14159,3.14159],linewidth=4,linestyle='--',color='#DAAA00')
plt.title('Pi Estimates by Dart Number')
plt.xlim(0,5)
plt.xlabel('Darts Thrown')
plt.ylabel('Pi Estimate')
plt.show()
'''
