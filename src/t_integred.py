import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import leastsq
import sys

#star = int(sys.argv[1])
star = 0
len  = 32
time = len*1000
raw  = np.load('/home/nch/raw_data/raw.npy')
freq = np.load('/home/nch/pulsar_data/freq.npy')
raw  = np.nan_to_num(raw)
data = raw[:,0,star:time+star]
data[2078:2150,:] = data[1654:1725,:] = 0
d_on  = []
d_off = []
for i in range(data.shape[1]/len/2):
      for j in range(len):
          d_on.append(  data[:,j + 2*i*len ])
          d_off.append( data[:,j + (2*i+1)*len])
d_on    = np.array(d_on).T
d_off   = np.array(d_off).T
s_on    = d_on.sum(axis=0)
s_off   = d_off.sum(axis=0)
fit_on  = signal.medfilt(s_on,15)
fit_off = signal.medfilt(s_off,15)




plt.figure(figsize = (12,12))
plt.subplot(2,2,1)
plt.title('Noise on')
plt.plot( s_on,'r',label='sum')
plt.plot( fit_on,'black',label = 'filtered by medium')
plt.ylabel('Intensity')
plt.xlabel('time')
plt.legend( loc = 'upper left')
plt.grid()

plt.subplot(2,2,2)
plt.title('Noise off')
plt.plot( s_off,'r',label='sum')
plt.plot( fit_off,'black',label = 'filtered by medium')
plt.ylabel('Intensity')
plt.xlabel('time')
plt.legend(loc = 'upper left')
plt.grid()

plt.subplot(2,2,3)
plt.title('Substraction')
plt.plot( s_on,'r',label='sum_on')
plt.plot( s_off,'black',label = 'sum_off')
sub = s_on - s_off
plt.plot( sub,'green',label = 'substraction')
plt.ylabel('Intensity')
plt.xlabel('time')
plt.plot(freq[np.argmax(sub)],sub.max(),'ro',label = str(sub.max()))
plt.plot(freq[np.argmin(sub)],sub.min(),'bo',label = str(sub.min()))
plt.legend(loc = 'upper left')
plt.grid()

plt.subplot(2,2,4)
plt.title('Substraction')
#plt.plot( fit_on,'r',label='fit_on')
plt.plot( s_off,'black',label = 's_off')
sub = s_on - s_off
s_off_c = s_off / sub
s_off_c = np.nan_to_num(s_off_c)
plt.plot( sub,'green',label = 'substraction')
plt.plot( s_off_c,'blue',label = 'Noise cal')
plt.ylabel('Intensity')
plt.xlabel('time')
plt.plot(freq[np.argmax(s_off)],s_off.max(),'ro',label = str(s_off.max()))
plt.plot(freq[np.argmax(s_off_c)],s_off_c.max(),'bo',label = str(s_off_c.min()))
plt.legend(loc = 'upper left')
plt.grid()

plt.savefig('/home/nch/FFT_search/graph/t_integred_cali')
plt.show()
print sub.shape
#np.save('/home/nch/FFT_search/src/chan_equiliz',sub)

