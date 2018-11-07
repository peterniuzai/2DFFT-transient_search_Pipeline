import numpy as np
import matplotlib.pyplot as plt
import sys,os,time
import scipy.signal as signal
from DM_calculate import DM_calculate
def Signal_finding(DM_axis,threshold,data,pixel,DM_range=[50,2000],comm_rank=0,p_n=0,i_ch=0,std_r=0,In_snr=0):
	 '''
	 Find signal above Threshold
	 '''
	 In_snr = In_snr
	 comm_rank = comm_rank
	 p_n	= p_n
	 i_ch   = i_ch
	 sig	= data.std()
	 seq	= comm_rank*p_n + i_ch
         data	= abs(data)
	 count  = 1
	 Gtransient=[]
	
	 n_count = np.zeros(data.shape[1])
	 DM_list    = np.zeros(data.shape[1]) #The max value of each DM

	 lo_threshold = np.where(data > threshold * data.std())
	 lo_t = np.array(lo_threshold)
	
	 for ii in range(len(lo_t[0])):

	 	n_count[lo_t[1][ii]] += count
	 	DM_list[lo_t[1][ii]] += data[lo_t[0][ii],lo_t[1][ii]]

	     	for jj in range(len(lo_t[0])):
	                for jj in range(len(lo_t[0])):

        	                if abs(lo_t[1][ii]-lo_t[1][jj]) < pixel and ii!= jj: #and abs(lo_t[0][ii]-lo_t[0][jj]) < pixel:

                	              n_count[lo_t[1][ii]] +=count
                        	      DM_list[lo_t[1][ii]] += data[lo_t[0][jj],lo_t[1][jj]]

	
	 lo_DM_max  = np.where(DM_list == DM_list.max())

	 lo_DM	= np.where(DM_list != 0)

         lo_count  = np.where(n_count !=0)#>=2)

	 DM_l =  DM_list[lo_DM] # Screen vaules of each DM that's above threshold second time.
	 SNR = (DM_l - data.mean())/sig
#	 SNR = DM_l
#	 SNR    = SNR / n_count[lo_count]
	 if len(SNR) > 0:
		 SNR = np.clip(SNR,0,SNR.max())

		 if n_count.max() > 1:
			 lo_count_max= np.where(n_count == n_count.max())

			 lo_intersection =list(set(lo_DM_max[0]).intersection(lo_count_max[0]))

			 for l in range(len(lo_intersection)):
				 dm =  DM_axis[lo_intersection[l]]
				 points = n_count[lo_intersection]
				 significance= (DM_l.max() - data.mean())/sig#/points[l]
				 print "\n\n########################"
				 print "Found bright Transient!!"
				 print "DM:",dm
				 print "Significance",significance
				 print "Number of pixels:",points[l]
				 print 'Seq of Total Chuncks:',seq
				 print 'rank of processor:',comm_rank
				 print 'Seq of chuncks of each processor:',i_ch
				 np.save('../data/2nd_FFT_'+str(seq),data)
				 print "########################\n\n"
				 Gtransient.extend([dm,seq,significance])
	 file_seq = np.ones(len(lo_DM[0]))*seq
	 DM_f_SNR_c = [DM_axis[lo_DM],file_seq,SNR,n_count[lo_count]]
	 DM_f_SNR_c = np.array(DM_f_SNR_c).T
	 
	 return DM_f_SNR_c, Gtransient

#################################################

if __name__ == '__main__':
        pixel = 2
	threshold = 7
	DM_range = [50,2000]
	data  = np.load('2ndFFT.npy')
#	data  = np.load('2ndFFT_1-4.npy')
	data  = abs(data)
	DM_axis = np.linspace(50,2000,data.shape[1])
#	y_axis = np.linspace(-data.shape[0]/2,data.shape[0]/2,data.shape[0])
#	plt.pcolormesh(DM_axis,y_axis,data,vmin=data.std()*threshold)
#	plt.xlim(195,210)
#	plt.colorbar()
#	plt.show()
#	exit()
	seq =0
	print "load over!"
        l,d,SNR,G=Signal_finding(DM_axis,threshold,data, pixel,DM_range,5)
	
		
        print 'l_count:',l
	print 'DM_l:',d
	print 'SNR:',SNR
	print 'Gtransient:',G[1]
	print len(G[0])
#	plt.plot(l[0],l[1],'ro')
#	plt.show()
