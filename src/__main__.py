import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import sys,os,time
from optparse import OptionParser
import mpi4py.MPI as MPI
from readfile import read_data
from dir_create import dir_create
#from calibrated import calibration
from rebin import rebin, rebin_inter
from FFT import FFT
from polar_transform import polar_coordinates_convert, polar_coordinates_convert_inter
from Signal_finding import Signal_finding
from DM_calculate import DM_calculate
from plot_all import plot

import warnings
warnings.filterwarnings("ignore")

from matplotlib.ticker import MultipleLocator, FormatStrFormatter , NullFormatter,LogLocator, LogFormatter,AutoLocator

from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
     # instance for invoking MPI relatedfunctions
     comm = MPI.COMM_WORLD
     # the node rank in the whole community
     comm_rank = comm.Get_rank()
     
     # the size of the whole community, i.e.,the total number of working nodes in the MPI cluster
     comm_size = comm.Get_size()

     if comm_rank !=0 and ('-h' in sys.argv or '--help' in sys.argv):
             exit(0)
     p = OptionParser()
     p.set_usage("mpirun -n <number of processor> python __main__.py  [options] ")
     p.set_description(__doc__)
     p.add_option('-f', '--file', dest='file', type='str',default='',
        help='Put filterbank file want to search')
     p.add_option('-t','--threshold', dest='threshold', type = 'float',default = 8.5,
        help='Threshold(sigma) for candidates pick')
     p.add_option('--dm', dest='dm', type = 'int',default = [50,2000],nargs=2,
        help='Set  DM range, Suggest use default [50,2000]')
     p.add_option('--pixel', dest='pixel', type = 'int',default = 5,
        help='Pixels number at 2nd 1DFFT map')
     p.add_option('--mask_cycle', dest='msk_cycle', type = 'int',default = 0,
        help='Mask abnormal lines at 1st 2DFFt map, only when RFI is terrible')
     p.add_option('--nsamps_gulp', dest='t_len', type = 'int',default = 0,
        help='Samples number for onece process, suggest use self-calculate(def)')
     p.add_option('--nbin', dest='nbin', type = 'int',default = 0,
        help='number of channels after re-bin step, suggest use self-calculate(def)')
     p.add_option('--wp', dest='Wp', type = 'int',default = 10,
        help='Set width of pulse to search, Suggest use default 10(ms).')
     p.add_option('--angle', dest='angle', type = 'float',default = [0,0],nargs=2,
        help='Angle range within [0,90], For de-buging.')

     p.add_option('-v', '--verbose', dest='verbose',action='store_true',
        help='Show details of process')
     p.add_option('-p', '--plot', dest='plot',action='store_true',
        help='Make overview plot for final result')

     p.add_option( '--Plot_proc', dest='plot_proc', type='str',default='',
        help='Input process step key words want to make signle plot.\
             \nKey words Including:\n\n{raw, rebin, 1stFFT, polar_sets_3D, polar_sets_2D, 2ndFFT_3D, 2ndFFT_2D}\n (This function Remain updates)')

     opts, args = p.parse_args(sys.argv[1:])

     if opts.file == '' or os.path.exists(opts.file)==False:
        if comm_rank == 0:
                print 'Please Checkout file input!'
                p.print_help()
        exit()
     comm.barrier()
     if opts.verbose:
         SHOW = 1
     else:
         SHOW = 0
     comm.barrier()


############
#Arguments #
############
     f_name    = opts.file
     plot_dir  = '../graph/' + f_name[-10:-4] + '/'
     if comm_rank == 0:
                dir_create(plot_dir)
                print 'directory build complete!'
     comm.barrier()

     DM_range  = opts.dm        # Min and Max DM
     Wp        = opts.Wp        # Wp means pulse width in (ms)
     t_len     = opts.t_len     # Time samples for each process chunk.(if 0, decided by DM_max)
     nbin      = opts.nbin      # Number of beams for rebin.if 0,nbin = N_f_chanel      
     ang       = opts.angle     # Angle range for search. If [0,0], Angle got from DM range.
     msk_cycle = opts.msk_cycle # Number of channels to be zeros in 2D-FFT(Noise remove).
     pixel     = opts.pixel     # Number of pixel to sum in 2ndFFT3D SNR compute.
     threshold = opts.threshold
     plot_proc = opts.plot_proc
######################
     Candidates_l = []   
     Gtran_l   	  = []
     t_consume	= 0
######################
#Read Data from file #
######################


     if comm_rank == 0:	 print 'Begin to load data from ' + f_name 
     fil, num, p_n, freq, t_rsl, t_len, t_gulp, nbin ,nch, T,fy,angle,n_deg,L_fft = read_data(f_name ,t_len, nbin,comm_rank, comm_size,DM_range,Wp,ang)

########################################
#Create New comm if time length too short     
#########################################
     comm =comm.Split(p_n)
     comm_rank = comm.Get_rank()
     comm_size = comm.Get_size()
     if p_n == 0:
	exit(0)

###################
#Begin to search  #
###################
     time_s = time.time()
     for  i_ch in range(p_n):  #i_chunk 
	     t_p    = comm_rank*p_n   #the thread position in total time in unit(chunk)
	     data   = fil.readBlock(t_len*(i_ch+t_p),int(t_len))
	     std_r  = data.std()	
	     data   = np.nan_to_num(data)
	     data   = data - data.mean()
	     t_ch_s = t_len*(i_ch+t_p)*t_rsl   #time of chunck start.
	     t_ch_e = t_len*(i_ch+t_p+1)*t_rsl 
	     t_axis = np.linspace(t_ch_s,t_ch_e,t_len) 
	     
	     if comm_rank == 0 and SHOW ==1:    print 'Load Data Over, Datasize for each chunk:',data.shape,'\nBegin to rebin... '
	     time_de_s = time.time()
	     re_data, f_axis  =  rebin_inter(data, fy, nbin,t_axis)

	     if comm_rank == 0 and SHOW ==1:    print 'Rebin over. \nBegin to do 1st 2-D FFT on rebin data...'
	
	     FFT1st_data = FFT(re_data, 2, L_fft, msk_cycle, t_gulp)
	
	     if comm_rank == 0 and SHOW ==1 :    print '1st FFT over.\nBegin to transform rectangular coordinates into polar coordinates...'
	
	     polar_data  = polar_coordinates_convert_inter( FFT1st_data, angle, n_deg, L_fft)
	     DM_axis = np.linspace(DM_range[0],DM_range[1],polar_data.shape[1])
	     

	     if comm_rank == 0 and SHOW ==1:    print 'Polar transform over,Polar data shape:',polar_data.shape,'\nBegin to do the 2nd 1-D FFT along radius direction...'
	     FFT2nd_data = FFT(polar_data, 1 )# 1 means 1 Dimension FFT
	     time_de_e = time.time()
	     t_consume +=  time_de_e - time_de_s
	     if comm_rank == 0 and SHOW ==1:
		    print '2nd FFT over.'
                    print '\n#############'
                    print 'Process matrix size:',nch,' * '+str(t_len)
                    print 'Dedispersion Time Cost:', t_consume ,'seconds,  ','equal',t_consume/60.,'minutes.'
                    print'Process:',i_ch,' of ',p_n,' for total:',p_n*t_len*comm_size,'samples'
                    print 'Angle range:',angle
                    print '###############\n'			

		    print 'Begin to locate the signal and calculate Significance...'
	     candidate, G_t  = Signal_finding(DM_axis,threshold,FFT2nd_data, pixel, DM_axis, comm_rank,p_n ,i_ch,std_r)
	     Candidates_l.extend(candidate)
	     Gtran_l.extend(G_t)
	
             if plot_proc != '':
   	                 print '\nBegin to plot...\n'
	  	         plot(comm_rank,t_axis,data,re_data,polar_data,FFT1st_data,FFT2nd_data,plot_proc,freq,f_axis,n_deg,plot_dir,pixel,angle,i_ch,p_n,SNR,DM,A_f)
	                 if comm_rank == 0 and SHOW ==1 and opts.Plot_proc: print 'Plot Over...\n\n'	
#########################################
# gather the results from all processes #
#########################################
     comm.barrier()
     time_e= time.time()

     if comm_rank ==0 and SHOW ==1 :
	   print '\n****************************************'	
	   print 'Total consume:',(time_e-time_s)/60.,'mins'
	   print '****************************************\n'
     combine_candidates  = comm.gather(Candidates_l,root=0)
     combine_Gtransient  = comm.gather(Gtran_l,root=0)
     
     if comm_rank == 0: 
		c_Candidates = []
		c_Gtransient = []
		for i in range(len(combine_candidates)):
			if len(combine_candidates[i]) > 1:
				for ii in range(len(combine_candidates[i])):
					c_Candidates.extend(combine_candidates[i][ii])

		
                for i in range(len(combine_Gtransient)):
                        if len(combine_Gtransient[i]) > 1:
                                for ii in range(len(combine_Gtransient[i])):
                                        c_Gtransient.extend([combine_Gtransient[i][ii]])


		c_Candidates = np.array(c_Candidates).reshape(-1,4).T
		c_Gtransient = np.array(c_Gtransient).reshape(-1,3).T
		if len(c_Candidates[0,:]) == 0:
			print 'Found no transient signal above threshold :('
			exit(0)
		print '\n\n****************************'
                print '*multiprocess plot over....*'
		print '****************************\n\n'
		if opts.plot:
			print 'Making Result...'
		else:	
			exit(1)
		###########################
		#Begin to make result Plot#
		###########################
                fig = plt.figure(figsize=(14,8))

		font = {'family' : 'serif',  
		        'color'  : 'black',  
		        'weight' : 'normal',  
		        'size'   : 16,  
		        }
		
                font1 = {'family' : 'sans-serif',
                        'color'  : 'black',
                        'weight' : 'normal',
                        'size'   : 10,
                        }		

                cm = plt.cm.get_cmap('Greens')
		if len(c_Gtransient[0]) != 0:
			G_snr	= c_Gtransient[2,:]
			lo      = np.where(G_snr == G_snr.max())
			M_seq   = int(c_Gtransient[1,lo[0][0]])
			dm_G    = c_Gtransient[0,lo[0][0]]
			lo_dm	= np.where(DM_axis == dm_G)
			G_data	= np.load('../data/2nd_FFT_'+str(M_seq)+'.npy')
			lo_max	= np.where(G_data[:,lo_dm[0]]==G_data[:,lo_dm[0]].max())
			y_axis	= np.linspace(-G_data.shape[0]/2,G_data.shape[0]/2,G_data.shape[0])
			ax1 = fig.add_subplot(2,2,1)
			A	= ax1.pcolormesh(DM_axis,y_axis, G_data,cmap=cm)		
			plt.colorbar(A)
			lo_max = y_axis[lo_max[0]]
			dm_G_p = np.zeros(len(lo_max))+dm_G
                #################################
                #             Plot 1            #
                #################################

			ax1.scatter(dm_G_p,lo_max,marker='o',c='',s=80,edgecolors='r')
			ax1.set_xlabel('Result DM',fontdict=font1)
			ax1.set_title('2nd FFT data of Tsamps:'+'\n('+str(M_seq*t_len)+'~'+str((M_seq+1)*t_len)+')',fontdict=font)
			ax1.set_ylim(-G_data.shape[0]/2,G_data.shape[0]/2)
			ax1.set_xlim(DM_axis[0]-1,DM_axis[-1]+1)
		else:
			print "Not Found any Giant Transient:("

		dm_c	= c_Candidates[0,:]
		seq_c	= c_Candidates[1,:]
		snr_c	= c_Candidates[2,:]
		n_count = c_Candidates[3,:]

		dm_bin	= DM_axis
#		print snr_c.shape,':snr_c'
#		print seq_c.shape,':seq_c'
#		print dm_c.shape,"dm_c"
#		print n_count.shape,'n_count'
#		print c_Candidates.shape,'c_Candidates'
		
		snr_bin	= np.linspace(0,snr_c.max(),20)
		Bins	= (snr_bin, dm_bin)

		d,x,y	= np.histogram2d(snr_c,dm_c,weights=n_count,bins = Bins)

		dm_bin	= np.linspace(dm_bin.min(),dm_bin.max(),d.shape[1])
		snr_bin	= np.linspace(snr_bin.min(),snr_bin.max(),d.shape[0])
		dm_bin ,snr_bin = np.meshgrid(dm_bin,snr_bin)

                #################################
                #             Plot 2            #
                #################################

#                ax2 = fig.add_subplot(2,2,2,projection='3d',azim=-79,elev=30, zlim=(0,n_count.max()))
		ax2 = fig.add_subplot(2,2,2,projection='3d',azim=-67,elev=30)		
		surface = ax2.plot_surface(dm_bin, snr_bin, d, linewidth=0, rstride=1, cstride=1, cmap=plt.cm.jet)
#                surface = ax2.plot_surface(snr_bin, dm_bin, d, linewidth=0, rstride=1, cstride=1, cmap=plt.cm.jet)		
#		plt.colorbar(surface)#,shrink=1)
		
		plt.xlim(DM_axis[0],DM_axis[-1])
		plt.ylim(0,snr_c.max()+1)

		xmajorLocator = MultipleLocator(400)
		xminorLocator = MultipleLocator(100)
		xmajorFormatter = FormatStrFormatter('%1d')

		ax2.xaxis.set_major_locator(xmajorLocator)
		ax2.xaxis.set_major_formatter(xmajorFormatter)

		ax2.xaxis.set_minor_locator(xminorLocator)

		ax2.set_xlabel('DM Value',fontdict=font1)
		ax2.set_ylabel('Significance',fontdict=font1)
		ax2.set_zlabel('Count Number',fontdict=font1)
		ax2.set_title('Distribution of Candidates',fontdict=font)
		ax2.xaxis.grid(True, which='minor')
		#################################
		#             Plot 3		#
		#################################
		ax3	= fig.add_subplot(2,1,2)
		
		cm = plt.cm.get_cmap('Reds')

		snr	= snr_c/snr_c.max()*120
		C 	= ax3.scatter(dm_c,seq_c,c=snr_c,alpha=0.5,edgecolors='bk',marker='*',s=snr,label='Candidates')
		if len(c_Gtransient[0]) !=0:
			dm_G    = c_Gtransient[0,:]
                	seq_G   = c_Gtransient[1,:]
        	        snr_G   = c_Gtransient[2,:]
	                snr     = snr_G/snr_G.max()*200
			G = ax3.scatter(dm_G,seq_G,c=snr_G,alpha=0.5,edgecolors='r',s=snr,marker='o',cmap=cm,label='Giant Transient')
				
			for i in range(len(dm_G)):
				cord = [dm_G[i],seq_G[i]]
				text = 'Significance:%1.1f'%(snr_G[i])
	                	plt.annotate(text, xy = cord, xytext = cord)
			
		plt.colorbar(C)
		ax3.set_xlim(DM_axis[0],DM_axis[-1])
#		ax3.set_xlim(10,DM_axis[-1])
		new_ticks = np.linspace(0,num,5,dtype=np.int)
		ticks_label =[]
		for i in new_ticks:
			label = '('+str(int(i*t_len))+'~'+str(int((i+1)*t_len))+')'
			ticks_label.append(label)
		
		xmajorLocator = MultipleLocator(400)
#		xmajorLocator = LogLocator(10)
		xmajorFormatter = FormatStrFormatter('%1.f')
		xminorLocator = MultipleLocator(100)
#		xminorLocator = LogLocator(1)

		ax3.xaxis.set_major_locator(xmajorLocator)
		ax3.xaxis.set_minor_locator(xminorLocator)
		ax3.xaxis.set_major_formatter(xmajorFormatter)		

		ymajorLocator = MultipleLocator(1)
		ax3.yaxis.set_major_locator(ymajorLocator)
		
#		ax3.semilogx()
		ax3.set_yticks(new_ticks)
		ax3.set_yticklabels(ticks_label)
		ax3.xaxis.grid(True,which = 'major')
		ax3.yaxis.grid(True,which = 'major')
		plt.legend(loc='best',scatterpoints=1,markerscale=0.6,framealpha=0.5,fontsize='xx-small')
		ax3.set_xlabel('DM',fontdict=font1)
		ax3.set_ylabel('Tsamps',fontdict=font1)
		
		plt.tight_layout()
		plt.savefig(plot_dir+'overview')
		plt.show()
     exit(1)
