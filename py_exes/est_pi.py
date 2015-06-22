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
    pi_est = 4.*M_in/len(x) # could've also been len(y) at bottom
    return pi_est

e3 = est_pi2(1e3,10)
e4 = est_pi2(1e4,10)
e5 = est_pi2(1e5,10)
e6 = est_pi2(1e6,10)
e7 = est_pi2(1e7,10)
e8 = est_pi2(1e8,10)

# timing?

estimates = [e3,e4,e5,e6,e7,e8]
plt.plot(estimates,linewidth=3,color='#000000')
ax.set_xticks(ticks=['1e3','1e4','1e5','1e6','1e7','1e8'])
plt.plot([-100,100],[3.14159,3.14159],linewidth=8,linestyle='--',color='#DAAA00')
plt.xlim(0,5)
plt.show()
