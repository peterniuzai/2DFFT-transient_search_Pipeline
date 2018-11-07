import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import leastsq
import sys

def calibration( chan_equaliz , raw_sets) :
  length   = 32
  d_sets   = raw_sets
  dat_set  = []
  chan	   = chan_equaliz
  for d_i in range(len(d_sets)):
	chan_equaliz = chan
	data  = d_sets[d_i]
	data[2078:2150,:] = data[1654:1725,:] = 0
	d_on  = []
	d_off = []
	for i in range(data.shape[1]/length/2):
	      for j in range(length):
        	  d_on.append(  data[:,j + 2*i*length ])
	          d_off.append( data[:,j + (2*i+1)*length])
	d_on    = np.array(d_on).T
	d_off   = np.array(d_off).T
	temp	= np.ones(d_on.shape[1],dtype = data.dtype)
	chan_equaliz = chan_equaliz[:,None] * temp[None , :]
	
	cal_on  = d_on  / chan_equaliz - 1
        cal_off = d_off / chan_equaliz
	cal_off = np.where(np.isinf(cal_off),0,cal_off)
	cal_on	= np.where(np.isinf(cal_on),0,cal_on)
	cal_on  = np.nan_to_num(cal_on)
	cal_off = np.nan_to_num(cal_off)

	data 	= np.empty(data.shape,dtype = data.dtype)
        for i in range(data.shape[1]/length/2):
		for j in range(length):
        		data[:, j + 2*i*length     ]  = cal_on[: , j + i*length]
			data[:, j + (2*i+1)*length ]  = cal_off[:, j + i*length]
	dat_set.append(data)
	
  return dat_set

if __name__ == '__main__' :
	raw  = np.load('/home/nch/raw_data/raw.npy')
	raw  = raw[:,0,:1024*4]
	tx   = np.arange(raw.shape[1])
	chan_equaliz = np.load('/home/nch/FFT_search/src/chan_equaliz.npy')
	freq = np.load('/home/nch/pulsar_data/freq.npy')
	raw  = np.nan_to_num(raw)
	raws = [raw]
	data = calibration(chan_equaliz, raws)
	data = data[0]



	plt.figure(figsize = (12,12))
	plt.subplot(2,2,1)
	plt.pcolormesh(tx,freq,data)
	plt.xlim(tx.min(),tx.max())
	plt.ylim(freq.min(),freq.max())
        plt.xlabel('time')
        plt.ylabel('Frequency (Mhz)')
	plt.title('Calibrated data map')
	plt.colorbar()

	plt.subplot(2,2,3)
        plt.xlabel('time')
        plt.ylabel('Frequency (MHz)')
	plt.title('Raw data map')
        plt.xlim(tx.min(),tx.max())
        plt.ylim(freq.min(),freq.max())
	plt.pcolormesh(tx,freq,raw)
        plt.colorbar()

	raw_s = raw.sum(axis = 1)/tx.size
	data_s = data.sum(axis = 1)/tx.size
	plt.subplot(2,2,2)
	plt.title('Spectrum')
        plt.ylabel('Intensity')
        plt.xlabel('Frequency Channel (Mhz)')
	plt.plot(freq,raw_s,label = 'raw')
	plt.plot(freq,data_s,label = 'calibrated')       
	sub_s = raw_s -data_s
	plt.plot(freq,sub_s,label = 'residual ')
	plt.legend(loc = 'upper left')
	plt.grid()

	sub = raw - data
        plt.subplot(2,2,4)
	plt.title('Residual')
        plt.pcolormesh(tx,freq,sub)
	plt.xlabel('time')
	plt.ylabel('Frequency (MHz)')
        plt.xlim(tx.min(),tx.max())
        plt.ylim(freq.min(),freq.max())
	plt.colorbar()
	plt.show()
