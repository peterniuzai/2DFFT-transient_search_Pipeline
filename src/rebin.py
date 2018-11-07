import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#import gc

def rebin(data,fy, nbin,tx=0):
    """
    .. Note::
	     Use histogram method
    Change the freqency axis to wave squre axis
    The signal line will become a straight line after this process.

    :param:
    data:  Input data require I(f,t) formate
    fy	   frequency ^ -2 
    nbin   How many channel of bins left, def =Nch
    tx	   Time axis
    :Output 
    data:  data after rebin with shape of (nbin,t)
    faxis: f axis after rebin
    """
    f_axis       = np.linspace(fy.min(),fy.max(),nbin)
    b_aray,b_edg = np.histogram(fy,bins=nbin)
    data	 = np.nan_to_num(data)
    data1	 = np.zeros((nbin,data.shape[1]),data.dtype)
    for i in np.arange(nbin):
            length  =  b_aray[i]
            if   i == 0:
                 index = 0
            elif i == 1:
                 index = b_aray[0]
            else:
                 index = b_aray[0:i].sum()#when i = 1 ,the bin_array[0:1] doesn't include the bin_array[1]
            for ii in np.arange(length):
                 data1[i,:]+=data[ii+index,:]
    tem = np.ones(data.shape[1])
    bin_weight = b_aray[:,None] * tem[None,:]
    data1 = data1 / 1.0 / bin_weight
    data1 = np.nan_to_num(data1)
    #for i in np.arange(5):
    #           y_s     =  data1.sum( axis = 1 )
    #           y_max   =  np.argmax(y_s)
    #           data1[y_max-10:y_max+11,:]=0
#    del tem,data
#    gc.collect()

    return data1, f_axis

def rebin_inter(data, fy, nbin, tx):
    """
    .. Note::
             Use histogram method
    Change the freqency axis to wave squre axis
    The signal line will become a straight line after this process.

    :param:
    data:  Input data require I(f,t) formate
    fy     frequency ^ -2 
    nbin   How many channel of bins left, def =Nch
    tx 	   time axis

    :Output 
    data:  data after rebin with shape of (nbin,t)
    faxis: f axis after rebin 
    """
    f		= np.linspace(fy.min(),fy.max(),nbin)
    b_aray 	= fy
    data	= interp1d(b_aray,data,axis=0)
    data	= data(f)
    data	= np.nan_to_num(data)
#    for i in np.arange(5):
#               y_s     =  data.sum( axis = 1 )
#               y_max   =  np.argmax(y_s)
#               data[y_max-2:y_max+2,:]=0
	
#	       x_s     =  data.sum( axis = 0 )
#               x_max   =  np.argmax(x_s)
#               data[:,x_max-2:x_max+2]=0

    return data, f


if __name__ == '__main__':
	exit()
