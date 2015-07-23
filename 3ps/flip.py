import numpy as np
import matplotlib.pyplot as plt
#import stats as s

def like(H,k,N): # k is number of heads, N is number of trials
    return H**k * (1-H)**(N-k)

H = np.linspace(0,1,100)

#params = np.array([1,.25,1/6,0])
#prior = s.makedata(params,H)
sigma = 1./15
prior = np.exp(-1./2*((.25-H)/sigma)**2)
#plt.plot(H,prior)

#plt.plot(H,prior)
#plt.show()

new1 = prior*like(H,2,5)
new2 = new1*like(H,5,10)
new3 = new2*like(H,8,15)
new4 = new3*like(H,13,20)

one, = plt.plot(H,new1/max(new1),linewidth=2,color='#E56020')
#plt.legend((one),('prior 1'))
plt.title('prior')
plt.show()

one, = plt.plot(H,new1/max(new1),linewidth=2,color='#E56020')
two = plt.scatter(H,new2/max(new2),linewidth=1,marker='x',color='#1D1160')
plt.legend((one,two),('prior 1', 'prior 2'))
plt.xlim(0,1.001)
plt.ylim(0,1.001)
plt.title('after 5 flips')
plt.show()

one, = plt.plot(H,new1/max(new1),linewidth=2,color='#E56020')
two = plt.scatter(H,new2/max(new2),linewidth=1,marker='x',color='#1D1160')
thr, = plt.plot(H,new3/max(new3),linewidth=4,linestyle=':',color='#F9A01B')
four, = plt.plot(H,new4/max(new4),linewidth=2,linestyle='--',color='#B95915')
plt.legend((one,two,thr,four),('prior 1', 'prior 2','prior 3','prior 4'))
plt.xlim(0,1.001)
plt.ylim(0,1.001)
plt.title('after 20 flips')
plt.show()


'''plt.plot(H,like(H,k,N))
plt.plot(H,like(H,1,1))
plt.plot(H,like(H,1,1)*100*like(H,3,5))
plt.plot(H,like(H,1,1)*1000*like(H,6,10))
plt.show()

prior = like(H,0,1)'''
