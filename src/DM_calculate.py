import numpy as np
from sigpyproc.Readers import FilReader
import sys
def DM_calculate(fy,loc,nbin,T):
    """
    Calculate DM from angle of candidate
    param:	
	fy:	array, frequency axis ^2 ,\nsame as wave square axis
	loc:	int deg, location of candidate between[0,90]
	nbin:	number of channel after rebin step.
	T:	Total observed time
    Return:
	DM value of candidate
    """
    C     = 4.148908e6 # (ms)
    k2    = np.tan(location/180.*np.pi)
    f_rsl = (fy[-1] - fy[0])
    unit  = f_rsl / T
    DM    = k2 / C / unit
    return DM

def length_calculate(fy,t_rsl ,DM_range,nbin,Wp,FFT_rs):
    '''
    Calculate signal Length in and 2DFFT map
    param:
	 fy:       array, frequency axis ^2 ,\nsame as wave square axis
	 t_rsl:	   time resolution of observation
	 DM_range: list, [DM_min,DM_max]
	 nbin:     number of channel after rebin step.
	 Wp:	   Width pulse, def = 10ms
	 FFT_rs:   FFT resolution got from matrix.
    return:
	 ang_rs, angle resolution got from FFT theory
	 L_fft,	 fft length got from FFT theory,
		 decided by width of pulse
	 

    '''

    C		= 4.148908e6 # (ms)
    f_rsl	= (fy[-1] - fy[0])/nbin
    unit	= f_rsl / t_rsl
    k1_l	= 1. / DM_range[1] / C / unit 
    #K1 with max DM in rebin map
    deg_min	= np.degrees( np.arctan(k1_l) ) 
    L_rebin	= nbin *(1+k1_l**2)**0.5/k1_l 
    # 1+cot^2 = 1/sin^2 => sin=k*{1/(1+k^2)}^0.5
    ang_rs	= 2.0/L_rebin
    ang_rs	= ang_rs / FFT_rs
    
    L_fft	= Wp*np.sin(deg_min*np.pi/180.)
    #Wp means pulse width in unit of 1 pixel.
    L_fft	= 2.0/L_fft
    L_fft	= L_fft /FFT_rs
    ang_rs	= ang_rs/(L_fft*np.pi/2)*90
    return ang_rs, L_fft

def angle_range(fy,DM_range,nbin,T):
    '''
    Calculate angle range according to given DM range
    param: 
	fy:       Array, frequency axis ^2 ,\nsame as wave square axis
	DM_range: list, [DM_min,DM_max]
	nbin:     number of channel after rebin step.
	T:        Total observed time.
    Return:
	Angle,	  list, [Angle_min,Anlge_max]
    '''

    C    	= 4.148908e6 # (ms)
    f_rsl	= (fy[-1] - fy[0])
    unit	= f_rsl / T
    k2_b    	= DM_range[0] * C * unit #K2 with min DM, close to horizontal
    k2_t    	= DM_range[1] * C * unit #K2 with max DM  far from horizontal
    theta2_b	= np.arctan(k2_b) *180 / np.pi
    theta2_t	= np.arctan(k2_t) *180 / np.pi
    angle = [theta2_b,theta2_t]
    return angle

def time_delay(DM_range,fbot,ftop):
    '''
    Calculate time delay according to top frequency and bottom frequency with max DM
    param:
	DM_range: list, [DM_min,DM_max]
    	fbot:	  float, bottom frequency
	ftop:	  float, top frequency

    '''
    C     = 4.148908e6 # (ms)
    t_delay   = C * DM_range[1] * (fbot**-2  -  ftop**-2) 
    #Time delay betweent top and bottom frequency with max DM value. (ms)
    return t_delay


if __name__ == '__main__':
    ang     =  0.3357#np.float(sys.argv[1])
    DM_range=  [50,500]#np.float(sys.argv[1])
    f_dir   = '/data0/FRB_data/parameter_test/'
    f_name  = 'Benchmark_2048_1ms_5s_snr_5_dm_500.fil'
    f = FilReader(f_dir + f_name)
    hdr    = f.header
    ftop   = hdr['ftop']
    fbot   = hdr['fbottom']
    nch    = hdr['nchans']
    t_rsl  = hdr['tsamp']*1000. # unit (ms)
    N_s_chunck = nch #* 4
#    t_len  = 250/1000.
    freq   = np.linspace(ftop,fbot,nch)
    #np.save(f_dir+'freq_s.npy',freq)
    fy     = freq**-2
    nbin   = nch
    T	   = N_s_chunck*t_rsl
    Nsamp  = time_delay(DM_range,fbot,ftop)/t_rsl
    Nchunck  = time_delay(DM_range,fbot,ftop)/t_rsl/N_s_chunck
    dm = DM_calculate(fy,ang,nbin,T)
    degree = angle_range(fy,DM_range,nbin,t_rsl * N_s_chunck)
    print 'DM is : ',dm ,'pc*cm^-3 at',ang,'deg'
    print 'Chunks samples:',N_s_chunck
    print 'The degree is :',degree, ' degree'
    print 'Load file from:',f_name
    print 'Delay within samples :',Nsamp
    print "Delay within chuncks:",Nchunck
