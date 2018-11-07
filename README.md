# 2DFFT-search
--This is a pipeline use 2DFFT method to search Fast Radio burst  

## 1. Introduction to 2DFFT pipeline:
 Here we proposed a different FRB searching algorithm which basically trace a curve in frequency-time image. This algorithm is mainly realized by 2 dimensional Fast Fourier Transform(2DFFT). We take a 2DFFT on I(f^-2^,t)  data map, Then trace the signal along the angle of straight line. In this searching method, it's easier to remove RFI in large scale and will bring a speed up benefit in well-developed 2D FFT library both in CPU and GPU code.

## 1. How to use
Add* */2DFFT_transient_search/src*   to your PYTHONPATH and LD_LIBRARY_PATH varible in your .bashrc file.  

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:path_to_download/2DFFT_transient_search/src"
	
	export PYTHONPATH="$PYTHONPATH:path_to_download/2DFFT_transient_search/src"
	cd src/  
Use

	 python __main__.py -h:  

	Usage: mpirun -n <number of processor> python __main__.py  [options]   
	Options:  
	  -h, --help show this help message and exit  
	  -f FILE, --file=FILE Put filterbank file want to search  
	  -t THRESHOLD, --threshold=THRESHOLD  
	                        Threshold(sigma) for candidates pick  
	  --dm=DM               Set  DM range, Suggest use default [50,2000]  
	  --pixel=PIXEL         Pixels number at 2nd 1DFFT map  
	  --mask_cycle=MSK_CYCLE   
	                        Mask abnormal lines at 1st 2DFFt map, only when RFI is    
	                        terrible   
	  --nsamps_gulp=T_LEN   Samples number for onece process, suggest use self-  
	                        calculate(def)  
	  --nbin=NBIN           number of channels after re-bin step, suggest use  
	                        self-calculate(def)  
	  --wp=WP               Set width of pulse to search, Suggest use default  
	                        10(ms).  
	  --angle=ANGLE         Angle range within [0,90], For de-buging.  
	  -v, --verbose         Show details of process  
	  -p, --plot            Make overview plot for final result  
	  --Plot_proc=PLOT_PROC  
	                        Input process step key words want to make signle plot.  
	                        Key words Including:  {raw, rebin, 1stFFT,  
	                        polar_sets_3D, polar_sets_2D, 2ndFFT_3D, 2ndFFT_2D}  
	                        (This function Remain updates)  
 
==Input file is required for filterbank file (*.fil) . SIGPYPROC is required at this stage==

## 3. Result overview:
If we use '-p' in parameter options, we will get a final plot like this:
![ ](https://raw.githubusercontent.com/peterniuzai/2DFFT-search/master/data/FRB090625.png  "FRB090625 with 2DFFT")

Plot for FRB090625 search result. Data has 4620288 time samples and 1024 frequency channels.  Top left give the candidate plot at 2nd 1DFFT map. Right top plot the candidates pixels number at each DM value and significance. Bottom plot is like traditional plot for DM and time, but time here are instead by time samples interval.
