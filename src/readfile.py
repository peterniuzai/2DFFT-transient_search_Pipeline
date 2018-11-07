import numpy as np
from sigpyproc.Readers import FilReader
from DM_calculate import angle_range
from DM_calculate import time_delay
from DM_calculate import length_calculate
#import readGBT
#import pyfits
#import h5py
def read_data(f_name, t_len, nbin, comm_rank=0,comm_size=4,DM_range=[100,1500],Wp=10,ang=[0,0]):
	 '''
	 load data,chunked them to data sets with shape :nbin * t_length (Default got from DM range). Wp means pulse width in (ms).
	 param:
		f_name:	    name of input file
		t_len:	    time length want to process	
		nbin:       number of channel after rebin step.
		comm_rank:  MPI,rank of process, def=0
		comm_size:  MPI, Number of total process, def =4
		DM_range:   list, [DM_min,DM_max]
		Wp:	    int, pulse width,(ms)
		angle:	    list, A list for [Angle_max, Angle_min]	

	 '''
	 fil	= FilReader(f_name)
	 hdr	= fil.header
	 ftop	= hdr['ftop']
	 fbot	= hdr['fbottom']
	 nch	= hdr['nchans']
	 t_rsl  = hdr['tsamp'] * 1000 # unit (ms)
	 nsamp	= hdr['nsamples']
	 C	= 4.148908e6 # (ms)
	 t_de	= time_delay(DM_range,fbot,ftop) #(ms)
	 t_lenr	= int(t_de / t_rsl) 
	 # t_lenr: real process time length
	 #t_gulp = 2**(int(np.log2(t_lenr))+1)
	 # Easy for FFT, pad time length equal 2^n with 0
	 t_gulp = 2**(round(np.log2(t_lenr)))


	 if nbin != 0:
		 nbin = nbin
	 else:
		 nbin = nch
		 #2**(round(np.log2(nch)))

         if t_len != 0:
		if t_len >= nch:
                    t_len  = t_len
         	else:
		    t_len = nch
	 else:
#                 t_len  = t_lenr
		  t_len  = t_gulp

         if nsamp < t_len:
                t_len = nsamp
                t_len = int(2**(round(np.log2(t_len))))


	 freq	= np.linspace(ftop,fbot,nch)
	 fy	= freq**-2

#Multi processors parameter
         num    = int(nsamp/t_len)	#Total number of chunks to process 
         p_n    = int(num/comm_size)	#Number of chunks for each thread to process
	
	 if p_n ==0:
		if comm_rank/num == 0:
			p_n = 1
		else:
			p_n = 0
#		print 'p_n:',p_n,'comm_rank:',comm_rank,'num:',num
#		exit(0)
		if comm_rank==0:
		  print "###!!!### \
			\nTime scale is too short ,will use (",num,") threads to process..\
			\n If you want full threads you input pleas:\
			\n 1)Input larger time scale data \
			\nor \
			\n 2)Use less threads \n"
	 
	 T	= t_len * t_rsl		# Each Chunk unit time scale(ms) 
#Image parameter
	 FFT_rs = ((1./nbin)**2 + (1./t_len)**2)**0.5 #For pixel
         angle	= angle_range(fy,DM_range,nbin,T)
	 if (ang[0] != 0) or (ang[1] !=0):
		angle = ang

	 if int(Wp/t_rsl) == 0:
		Wp = 1
	 else:
		Wp = int(Wp/t_rsl) 

	 ang_rs, L_fft	= length_calculate(fy,t_rsl ,DM_range,nbin,Wp, FFT_rs)
	 #L_fft means length of signal after FFT, it's deciede by Width of pulse in search
	 Ang_rs = (2**0.5/L_fft)*180/np.pi
#         print 'Angle:', angle
#         print 'L_fft:', L_fft
#         print 'Rad_rs', Rad_rs
#         print 'Width resolution:Ang_rs', Ang_rs
#         print 'Matrix Ang resoulution',ang_rs

	 if ang_rs < Ang_rs :
		Ang_rs = Ang_rs
#		print 'Using matrix angle resolution..'
	 else:
	        Ang_rs  = ang_rs
#		print 'Using matrix angle resolution..'
	 n_deg = int((angle[1]-angle[0])/Ang_rs)
	 
	# Radius resolutiongrid size for interpolate in polar-coordin transform.
     	# Angle resolution for interpolate in polar-coordin transform.
	
         return fil, num, p_n, freq, t_rsl, t_len, t_gulp, nbin, nch , T , fy,angle, n_deg, L_fft

if   __name__ == '__main__':
     f_dir  = '../data/'
     f_name = 'data_2017-08-30_17-35-36.fil'
     f, num,p_n,freq, t_rsl, t_len = read_data(f_dir ,f_name ,0,0,10)
     print freq.shape
     print p_n
     b = f.readBlock(0,100)
     print b.shape
     print 'load over!'
     exit()
