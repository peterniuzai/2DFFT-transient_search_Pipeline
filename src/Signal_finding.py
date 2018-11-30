import numpy as np
import matplotlib.pyplot as plt
import sys,os,time
import scipy.signal as signal
from DM_calculate import DM_calculate
from scipy.optimize import curve_fit
from scipy.stats import norm,chi2


def gauss(x, *p):
    '''
    Define model function to be used to fit to the data:
    '''
#    A, mu, sigma = p
    mu,sigma = p
#    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    return 1./((2*np.pi)**0.5*sigma)*np.exp(-(x-mu)**2/(2.*sigma**2))

def Signal_finding(DM_axis,threshold,data,pixel,DM_range=[50,2000],seq=0,In_snr=0):
	 '''
	 Find signal above Threshold
	 '''
	 
	 In_snr = In_snr
	 sig	= data.std()
	 data	= data
#	 seq	= comm_rank*p_n + i_ch
	 seq 	= seq
	 dump_flag = 0
	 Gtransient=[]
	
	 n_count = np.zeros(data.shape[1])
	 DM_list    = np.zeros(data.shape[1]) #The max value of each DM
	 
         n_count1 = np.zeros(data.shape[1])
         DM_list1 = np.zeros(data.shape[1]) #The max value of each DM
	 
	 p0 = [ 0., 1]
	 d_r = data.real.reshape(-1)
         d_i = data.imag.reshape(-1)
	 nbin=int(data.size**0.5)
 	 a_r, b_r =np.histogram( d_r,bins=nbin,density=True)
	 a_i, b_i =np.histogram( d_r,bins=nbin,density=True)
	 b_r_centers = (b_r[:-1] + b_r[1:])/2
	 b_i_centers = (b_i[:-1] + b_i[1:])/2
	 #print a_r.min(),a_r.max(),a_r.shape
	 #print a_i.min(),a_i.max(),a_i.shape
	 a_r_coeff,a_r_var_matrix = curve_fit(gauss,b_r_centers,a_r,p0=p0)#, maxfev=500000)
	 a_i_coeff,a_i_var_matrix = curve_fit(gauss,b_i_centers,a_i,p0=p0)
	 a_r_std  = a_r_coeff[1]
	 a_r_mean = a_r_coeff[0]
         a_i_std  = a_i_coeff[1]
         a_i_mean = a_i_coeff[0]
	 threshold = threshold*2 # 2 is std for chi2 distribution with df = 2 .
	 alfa = chi2.cdf(threshold,2)	 
	 #alfa = 0.99999
	 cut_off = chi2.ppf(alfa,2)
	 #print cut_off
         #******************************
         #Normalization of distribution*
         #******************************
	 mu_c = a_r_mean + 1j * a_i_mean
	 std_c= abs(a_r_std  + a_i_std*1j)
	 std_c= a_r_std
	 data = (data-mu_c) / (std_c)
	 data = (abs(data))**2
#######
#Test##
######
#	 plt.hist(data.reshape(-1),bins=nbin,histtype='step', density=False,label='data')
#	 rvs = chi2.rvs(2,size=data.size)
#	 plt.hist(rvs,bins=nbin,histtype='step',density=False,label='standard')
#	 plt.axvline(x=cut_off,label=str(alfa*100)+'%')
#	 plt.legend()
#	 plt.show()
#	 exit()
#	 data = data - cut_off
#	 data = np.clip(data,0,data.max()) 	

#	 DM_list = data.sum(axis=0)
	 
         lo_threshold = np.where(data > cut_off)
	
	 	 
	 n_count ,b = np.histogram(DM_axis[lo_threshold[1]],bins=DM_axis.size,range=(DM_axis.min(),DM_axis.max()))
	 
         data = data - cut_off
         data = np.clip(data,0,data.max())
         DM_list = data.sum(axis=0)	
#	 test_SHOW =1
#	 if seq == 100 and test_SHOW ==1:	
#		 plt.figure(figsize=(14,12))
#                 plt.subplot(2,2,1)
#                 plt.plot(DM_axis,n_count)
#                 plt.plot(n_count,'.-')
#                 plt.axvline(x=100,color='red',linestyle='--')
#
#                 plt.subplot(2,2,2)
#                plt.plot(DM_axis,DM_list,'.-',color='darkgreen')
#                 plt.plot(DM_list,'.')
#                plt.axvline(x=100,color='red',linestyle='--')


	 lo_threshold = np.where(DM_list > 0)
	 for ii in range(len(lo_threshold[0])):
		for jj in range(len(lo_threshold[0])):
			if abs(lo_threshold[0][ii]-lo_threshold[0][jj]) < pixel:# and ii!=jj:
				n_count1[lo_threshold[0][ii]] += n_count[lo_threshold[0][jj]]
				DM_list1[lo_threshold[0][ii]] += DM_list[lo_threshold[0][jj]]
#  	 DM_list1 += DM_list
#	 n_count1 += n_count
#         if seq == 100 and test_SHOW==1:
#                 plt.subplot(2,2,3)
#                 plt.plot(DM_axis,n_count1)
#		 plt.plot(n_count1,'.-')
#                 plt.axvline(x=100,color='red',linestyle='--')

#		 plt.subplot(2,2,4)
#	 	 plt.plot(DM_axis,DM_list1,'.-',color='darkgreen')
#		 plt.plot(DM_list1,'.')
#		 plt.axvline(x=100,color='red',linestyle='--')
#		 plt.show()
	
	 lo_DM_max  = np.where(DM_list1 == DM_list1.max())


         lo_count  = np.where(n_count1 >=2)
	 DM_l =  DM_list1[lo_count] # Screen vaules of each DM that's above threshold second time.
	 SNR = DM_l
	 if len(SNR) > 0:
		 SNR = np.clip(SNR,0,SNR.max())	
		 if n_count1.max() >=2:
			 lo_count_max= np.where(n_count1 == n_count1.max())

			 lo_intersection =list(set(lo_DM_max[0]).intersection(lo_count_max[0]))
			 if lo_intersection == [] and (lo_DM_max[0][0] - lo_count_max[0][0] > pixel):
			 	lo_intersection = []
			 elif lo_intersection != []:
				lo_intersection = lo_intersection
			 elif lo_intersection == [] and (abs(lo_DM_max[0][0] - lo_count_max[0][0])<pixel*2):
			     	lo_intersection = lo_DM_max[0]
		
			 if len(lo_intersection) != 0:
				 l = len(lo_intersection)/2-1				
				 dm =  DM_axis[lo_intersection[l]]
				 points = n_count1[lo_intersection]
				 significance= (DM_l.max() - data.mean())/sig/points[l]
				 print "-------------------------------"
				 print "Found bright Transient!!"
				 print "DM:%.2f"%dm
				 print "Significance:%.2f"%significance
				 print "Number of pixels:",points[l]
				 print 'Seq of Total Chuncks:',seq
				 print "-------------------------------"
				 dump_flag = 1
				 Gtransient.extend([dm,seq,significance])
#				 Gtransient.extend([dm,seq,alfa])
	 file_seq = np.ones(len(lo_count[0]))*seq
	 DM_f_SNR_c = [DM_axis[lo_count],file_seq,SNR,n_count1[lo_count]]
	 DM_f_SNR_c = np.array(DM_f_SNR_c).T
	 return DM_f_SNR_c, Gtransient, dump_flag

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
