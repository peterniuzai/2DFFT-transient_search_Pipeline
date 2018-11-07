import numpy as np
import matplotlib.pyplot as plt
from DM_calculate import DM_calculate
import os

def dedispersion(DM, data , freq, t_rsl):
    for i in np.arange(len(freq)):
	     delt =  4.148808e6 * DM * (freq[0]**-2  -  freq[i]**-2)
	     step =  int(round(delt/1000/t_rsl))# divide 1000  1000ms= 1s
	     data[i,:] = np.roll(data[i,:],step,-1)
    return data






if __name__ == '__main__':
     #DM     = 623.30    
     d_dir  ='/home/nch/FFT_search/data/wigglez_found/data_pick/'
     time   = np.load('/home/ycli/data/burst_data/time.npy') 
     freq   = np.load('/home/ycli/data/burst_data/freq.npy')
     t_rsl  = time[2]-time[1]
     f_axis = freq ** -2
     nbin   = 2048
     data_id  = [
            [0],                                             #0
            [6,10,11,12,13,14,20,22,27,28,36,37,53,55],      #1
            [4,7,10,15,21,22,25,31,32,38,40],                #2
            [1,2,5,7,9,11,17,20,27,28,29,37,40,47],          #3
            [0,5,11,13,19,21,36,38,43,44,45,48,52,53,54,55], #4
            [4,12,19,20,26,38,39],                           #5
            [8,9,16,17,23],                                  #6
            [2,3,9,10,17,25,26],                             #7
            [0],                                             #8
            [55],                                            #9
            [23],                                            #10
            [6],                                             #11
            [33,44,46],                                      #12
            [41],                                            #13  strong pulsar
            [52],                                             #14
            [5,11,13,22,54,59]                               #15
            ]

     location_id =[
            [2.23],                                             #0
            [1.97,14.743,2.07,2.07,2.07,57,1.68,64,57,2,2.17,2.17,57,13],      #1
            [1.38,1.68,39,2.37,1.979,2.57,6.332,2.37,37,2.27,2],                #2
            [60,70,60,35.227,1.97,10,72.136,2,2.07,1.88,2.07,2.07,2.275,61.54],          #3
            [22.56,33.94,6.6298,2.374,1.28,59.8,2.8,24.045,2.2,2.2,60,68.969,2.37,32.15,14.44,5.1455], #4
            [2,2,2.572,2,2.4,11.97,2],                           #5
            [2.37,2.37,2.57,2.37,2.37],                                  #6
            [2.27,1.88,2.47,2.07,2.27,2.67,2.5],                             #7
            [2.2],                                             #8
            [2.27],                                            #9
            [2.27],                                            #10
            [70],                                             #11
            [2.27,1.088,0.98952],                                      #12
            [2.6717],                                            #13  strong pulsar
            [59.173],                                             #14
            [3.364,33.45,0.3958,2.077,0.3958,19.79]                               #15
            ]


     for i in range(16):
		for j in range(len(data_id[i])):
                	data = np.load(d_dir + str(i) + '_' + str(data_id[i][j]) +'.npy')
#                        DM   = DM_calculate(f_axis,t_rsl,location_id[i][j],nbin)
                        DM   = 13.9
                        print str(i) + '_' + str(data_id[i][j]),', DM:',DM
                        data = dedispersion(DM, data , freq, t_rsl)
                        
                        data   =  np.sum(data,axis=0)
                        m      =  np.mean(data)
                        SNR    =  (data.max() - m )/m
                        plt.plot(data)
		        plt.title('Dedispersion SNR:'+str(SNR) + 'DM:'+str(DM))
			plt.ylabel('value')
			plot_dir = '/home/nch/FFT_search/graph/DM_13_9/'
			plt.savefig(plot_dir+str(i) + '_' + str(data_id[i][j]))
                        plt.close()
