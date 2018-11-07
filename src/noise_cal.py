import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import leastsq

time  = 1024*5
freq  = np.load('/home/ycli/data/burst_data/freq.npy')
raw   = np.load('/home/ycli/data/burst_data/raw.npy')
tx    = np.arange(time/2)
data  = raw[:,0,:time]
data[2078:2150,:] = data[1654:1725,:] = data[800:921,:] =data[542:644,:] = data[410:424,:] =  0
d_on  = []
d_off = []
for i in range(data.shape[1]/32/2):
      for j in range(32):
          d_on.append(  data[:,j + 2*i*32 ])
          d_off.append( data[:,j + (2*i+1)*32])
d_on    = np.array(d_on)
d_off   = np.array(d_off)
s_on    = d_on.sum(axis=0)/time*2
s_off   = d_off.sum(axis=0)/time*2
fit_on  = signal.medfilt(s_on,15)
fit_off = signal.medfilt(s_off,15)


Nois_g  = fit_on - fit_off
Nois_g  = s_on   - s_off
#Nois_g  = Nois.reshape(-1,1)
#no_cal    = data[:,:data.shape[1]/2].T/Nois_g
no_cal    = d_off / Nois_g
no_cal    = np.nan_to_num(no_cal)

plt.figure(figsize = (15,15))
plt.subplot(2,2,1)
plt.title('Noise cal')
plt.xlabel('frequency')
plt.ylabel('time')
plt.ylim(tx.min(),tx.max())
plt.pcolormesh(freq,tx,no_cal)
plt.colorbar()

plt.subplot(2,2,2)
plt.title('Noise on')
plt.xlabel('frequency')
plt.ylabel('time')
plt.ylim(tx.min(),tx.max())
plt.pcolormesh(freq,tx,d_on)
plt.colorbar()

plt.subplot(2,2,3)
plt.title('Noise off')
plt.xlabel('frequency')
plt.ylabel('time')
plt.ylim(tx.min(),tx.max())
plt.pcolormesh(freq,tx,d_off)
plt.colorbar()


plt.subplot(2,2,4)
plt.title('original')
plt.ylim(tx.min(),tx.max())
plt.xlabel('frequency')
plt.ylabel('time')
plt.pcolormesh(freq, tx, data[:,:data.shape[1]/2].T)
plt.colorbar()

plt.savefig('/home/nch/FFT_search/graph/2D_noise_cal')
plt.show()
