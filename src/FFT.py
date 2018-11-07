import numpy as np
import matplotlib.pyplot as plt
import time
def FFT(data, Dim  ,L_fft = 0, msk_cycle = 0,t_gulp =0,W_f=1):
    """
    FFT on data with numpy.fft
    :param 
    data:  Input data: 
    Dim:   Values of (1,2), 1 means 1DFFT,2 means 2DFFT
    L_fft: int,Length of data from center matric after 2DFFT,\nonly valid when Dim = 2.
    mask_cycle: int, If data are realy bad, mask extream high data.
    W_f:   add window as filter after 2DFFT 

    Return: 
	  data after 2DFFT or 1DFFT
    """

    if Dim == 2:   #1st 2-D FFT

#	   pad_l1 = int((t_gulp - data.shape[1])/2)
#	   pad_l2 = t_gulp - data.shape[1] - pad_l1
#	   data	  = np.pad(data,[(0,0),(pad_l1,pad_l2)],mode='constant')
#          data   = np.pad(data,[(0,0),(0,t_gulp - data.shape[1])],mode='constant')
#	   print data.shape,"After pad"

	   data   = np.fft.rfft2(data,norm = 'ortho')
	   data	  = data[int(-L_fft):, 1:int(L_fft)+1]
	   
           blackman_x    = np.blackman(data.shape[1]*2)
           blackman_y    = np.blackman(data.shape[0]*2)
	   #Window function
           window  = blackman_y[:,None] * blackman_x[None,:]
	   window  = window[:data.shape[0],data.shape[1]:]
	   data = data * window

	   data[-3:,:] = 0
	   data[:,:3] = 0

	   if msk_cycle >0:
           	for i in np.arange(msk_cycle):
              	 	x_sum   =  np.abs(data).sum(axis = 0)
               		y_sum   =  np.abs(data).sum(axis = 1)
              	 	x_max   =  np.argmax(x_sum)
              		y_max   =  np.argmax(y_sum)
               		data[:,	  x_max-2:x_max+2] = 0
               		data[y_max-2:y_max+2,  :  ] = 0
    elif Dim == 1: #2nd 1D-FFT along radius
	   
	   pad_l  = 2**(int(np.log2(data.shape[0]))+1)-data.shape[0]
	   pad_l1 = int(pad_l/2)
	   pad_l2 = pad_l - pad_l1
#	   data	  = np.pad(data,[(pad_l1,pad_l2),(0,0)],mode='constant')
#	   data   = np.pad(data,[(pad_l,0),(0,0)],mode='constant')
           data   = np.fft.fft(data,axis=0,norm = 'ortho')
           data   = np.fft.fftshift(data,axes=0)
    return data

if __name__ == '__main__':
      
	print 'FFT process'
