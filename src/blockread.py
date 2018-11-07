from sigpyproc.Readers import FilReader
import numpy as np
import matplotlib.pyplot as plt
f	= FilReader('../data/2011-02-20-01:52:19.fil')
data	= f.readBlock(2*1024*20,1024*20)
d_l	= []
fft_l	= []
for i in range(10):
    d_l.append(data[:,i*2048:(i+1)*2048])
d_l=np.array(d_l)
for i in range(10):
    fft = np.fft.fft2(d_l[i])
    fft = np.fft.fftshift(fft)/fft.size
    fft	= fft[1:fft.shape[0]/2+1,fft.shape[1]/2:fft.shape[1]/2+fft.shape[0]/2]
    fft[-1, :] = 0
    fft[ :, 0] = 0
    fft_l.append(abs(fft))
    
for i in range(10):
    plt.pcolormesh(fft_l[i])
    plt.colorbar()
    plt.show()

