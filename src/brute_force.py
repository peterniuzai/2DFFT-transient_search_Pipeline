import numpy as np
from sigpyproc.Readers import FilReader
import sys
from DM_calculate import time_delay
import matplotlib.pyplot as plt

if __name__ == '__main__':
    DM_range=[50,100]
    f_dir   = '../data/'
    f_name  = 'noise_level_test.fil'
    f = FilReader(f_dir + f_name)
    data = f.dedisperse(100)
    std  = data[:2000].std()
    mean = data[:2000].mean()
#    snr = (data.max()-mean)/std
    snr  = (data.max()-data.mean())/data.std()
    lo	 = np.where(data==data.max())
    flag = 0
    for i in range(len(lo[0])):
	 if  2490 <lo[i][0] < 2505:
	      flag = 1
 	      print 'True'    
	 else:
	      print 'False'
    print snr
    plt.plot(data)
    plt.show()
    exit(1)
    hdr    = f.header
    ftop   = hdr['ftop']
    nsample= hdr['nsamples']
    fbot   = hdr['fbottom']
    nch    = hdr['nchans']
    t_rsl  = hdr['tsamp']*1000. # unit (ms)
    freq   = np.linspace(ftop,fbot,nch)
    fy     = freq**-2
    Nsamp  = int(time_delay(DM_range,fbot,ftop)/t_rsl)
    print 'Nasmples:', nsample
    print 'Load file from:',f_name
    print 'Delay within samples :',Nsamp
