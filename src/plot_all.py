import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
'''Plot the internal process with your demand. These could define in plot_proc variable'''
def plot(comm_rank,t_axis , data , re_data , polar_data , FFT1st_data , FFT2nd_data,\
         process , freq , f_axis ,  a_r ,dir , pixel = 5,angle=[5,88] ,i_ch=0,p_n=0,SNR=0,DM=0,A_f=0):
       '''Plot the internal process with your demand. These could define in plot_proc variable
	  Still in updating...
       '''
       ang_min = angle[0]
       ang_max = angle[1]
       tan_min        = np.tan(ang_min*np.pi/180)
       tan_max        = np.tan(ang_max*np.pi/180)
       tan_grid       = np.linspace(tan_min,tan_max,polar_data.shape[1])
       x_deg          = np.arctan(tan_grid)/np.pi*180
       SHOW    = 1
       Imag_show = 6
       if   process  == '':
            if comm_rank == 0 and SHOW ==1:    print 'Please claim which process to plot if you want to make plot!'
       if 'raw' in process:
                data  = np.ma.masked_invalid(np.abs(data))
                mean  = np.mean(data)
                sigma = np.var(data)
		t_x   = t_axis
                max   = mean + 2 * sigma
                min   = mean - 2 * sigma
                seq   = comm_rank * p_n  + i_ch
                plt.pcolormesh(t_axis,freq,data)#,vmax = max,vmin = min)
                plt.title('raw_data')
                plt.xlabel('time(s)')
                plt.ylabel('frequency(Mhz)')
                plt.xlim(t_x.min(),t_x.max())
                plt.ylim(freq.min(),freq.max())
                plt.colorbar()
                p_dir = dir + 'raw/'
                plt.savefig(p_dir + 'raw_'+str(seq))
                if seq == Imag_show and SHOW == 1:
                        plt.show()
                plt.close()
	        if comm_rank == 0: print 'raw data plot is over...'

       if 'rebin' in process:
                data  = np.abs(re_data)
                mean  = np.mean(data)
                sigma = np.var(data)
                max   = mean + 2 * sigma
                min   = mean - 2 * sigma
                seq   = comm_rank * p_n + i_ch
                plt.pcolormesh(t_axis,f_axis,data)#,vmax = max,vmin = min)
                plt.title('data after rebin')
                plt.xlabel('time(s)')
                plt.ylabel('frequency(Mhz)')
                plt.xlim(t_axis.min(),t_axis.max())
                plt.ylim(f_axis.min(),f_axis.max())
                plt.colorbar()
                p_dir = dir + 'rebin/'
                plt.savefig(p_dir + 'rebin_' + str(seq))
                if seq == Imag_show and SHOW == 1:
                        plt.show()
                plt.close()
	        if comm_rank==0: print 'rebin plot over...'

       if '1stFFT' in process:
                data  = np.abs(FFT1st_data)
                mean  = np.mean(data)
                sigma = np.var(data)
          #     max   = mean + 2 * sigma
          #     min   = mean - 2 * sigma
                seq   = comm_rank * p_n + i_ch
	#	data[-3:,:4]=data.mean()
		#data[:,0]=data.mean()
                plt.pcolormesh(data)#,vmax = max,vmin = min)
                plt.title('1st FFT')
                plt.xlabel('Time axis after 1st FFT')
                plt.ylabel('Frequency after 2nd FFT')
                plt.colorbar()
                index  = np.where(data == np.max(data))
                cord =(index[1][0],index[0][0])
#                if comm_rank == 0:    print index
#                np.savetxt('/home/nch/cord.txt',cord)
                plt.annotate('max:'+str(index[1][0]), xy = cord, xytext = cord)
                              #arrowprops = dict(facecolor = 'red', shrink = 0.01))
                p_dir = dir + '1stFFT/'
                plt.savefig(p_dir + '1stFFT_' + str(seq))
		if seq == Imag_show and SHOW==1:
			plt.show()	
                plt.close()
	        if comm_rank==0: print '1stFFT plot over....'

       if 'polar_sets_3D' in process:
                data  = np.abs(polar_data)
#                mean  = np.mean(data)
#                sigma = np.var(data)
#                max   = mean + 2 * sigma
#                min   = mean - 2 * sigma
                seq   = comm_rank * p_n + i_ch
                y_axis  = np.arange(data.shape[0])
                plt.pcolormesh(x_deg,y_axis,data)#,vmax = max,vmin = min)
#                plt.title('radius - angle(grid size:'+ str(r_r)+'*'+str(a_r)+')')
		plt.title('radius - angle(Angle found:'+str(int(A_f))+')')
                plt.xlabel('Angle(in degree)')
                plt.ylabel('Radius')
                plt.figtext(0.08,0.98,'SNR:'+str(SNR))
                plt.xlim(x_deg.min(),x_deg.max())
                plt.ylim(y_axis.min(),y_axis.max())
                plt.colorbar()
		
                p_dir = dir + 'polar_sets_3D/'
                plt.savefig(p_dir + 'polar_3D_' + str(seq) )
		if seq == Imag_show and SHOW == 1:
			plt.show()
                plt.close() 
	        if comm_rank==0: print 'polar_3D plot over...'

       if 'polar_sets_2D' in process:
                data  = polar_data
                data  = data.sum(axis = 0)
		data  = abs(data)
                seq   = comm_rank * p_n + i_ch
#                if seq == 7:
#                    np.save('/home/nch/temraw.npy',data)
                #Filter the profile of the FRB signal
#                prof_data = signal.medfilt(data,5)
#                data      = data - prof_data
                snr_2D  = np.nan_to_num((data.max()-data.mean())/data.std())
                dmax =  np.argmax(data)
                cord =  (x_deg[dmax] ,data[dmax])
                plt.plot([cord[0]],[data[dmax]],'ro')
                plt.figtext(0.08,0.98,'SNR:'+str(snr_2D))
#                plt.title('polar Sum along radius axis(grid size:'+ str(r_r)+'*'+str(a_r)+')')
		plt.title('polar Sum along radius axis(Angle found:'+str(int(A_f))+')')
                plt.xlabel('Angle(in degree)')
                plt.ylabel('Intensity')


                plt.annotate('angle:'+str(cord[0])+'deg', xy = cord, xytext = cord, \
                              arrowprops = dict(facecolor = 'black', shrink = 0.1))

                plt.grid()
                plt.xlim(x_deg.min(),x_deg.max())
                plt.ylim(data.min()-10,data.max()+10)
                plt.plot(x_deg,data)
                p_dir = dir + 'polar_sets_2D/'
                plt.savefig(p_dir + 'polar_2D_' + str(seq))
		
                if seq == Imag_show and SHOW==1:
                        plt.show()
                plt.close()
       		if comm_rank==0: print 'polar_2D plot over...'

       if '2ndFFT_3D' in process:
                data  = np.abs(FFT2nd_data)
#                mean  = np.mean(data)
#                sigma = np.var(data)
#                max   = mean + 2 * sigma
#                min   = mean - 2 * sigma
                seq   = comm_rank * p_n + i_ch
#                lo    = np.where(data == np.max(data))
#                d_max = data.max()
#                for i in np.arange(-pixel,pixel):
#                    for j in np.arange(-pixel,pixel):
#                        d_max += data[lo[0][0]+i,lo[1][0]+j]
#		print data.mean(),data.std(),data.max()
#                SNR   = (d_max - data.mean())/data.std()
#                for ii in range(3):
#                       ind  = np.where(data == data.max())
#                       y_max  = ind[0][0]
#                       if y_max  ==  data.shape[0]/2 or y_max == data.shape[0]/2 + 1:
#                          data[ind] = 0
                ind	= np.where(data == data.max())
                deg	= x_deg[(ind[1][0])]
		y_ax	= (ind[0][0]-data.shape[0]/2)
                y_axis  = np.arange(-data.shape[0]/2,data.shape[0]/2)
                plt.pcolormesh(x_deg,y_axis,data)#,vmax = max,vmin = min)
                plt.title('2nd FFT along radius axis(Angle found:'+str(int(A_f))+')')
                plt.xlabel('Angle(in degree)')
                plt.ylabel('Radius after FFT')
                plt.xlim(x_deg.min(),x_deg.max())
                plt.ylim(y_axis.min(),y_axis.max())
                plt.figtext(0.08,0.98,'SNR:'+ str(int(SNR))+'DM:'+str(int(DM)) + ' Location: ' + str(int(deg)) + ', y_axis:'+str(y_ax))
                plt.colorbar()
                p_dir = dir + '2ndFFT_3D/'
                plt.savefig(p_dir +'2ndFFT_3D_' + str(seq) )
                if seq == Imag_show and SHOW==1:
                        plt.show()
		
                plt.close()
		#exit(1)
	        if comm_rank==0: print '2nd FFT 3D plot over...'

       if '2ndFFT_2D' in process:
                data  = FFT2nd_data
                data  = abs(data)
                data  = data.sum(axis = 0)
                seq   = comm_rank * p_n + i_ch
                #Filter the profile of the FRB signal
#                prof_data = signal.medfilt(data,199)
#                data      = data - prof_data
                SNR_2nd_2D	= np.nan_to_num((data.max()-data.mean())/data.std())
                dmax =  np.argmax(data)
                cord =  (x_deg[dmax] ,data[dmax])
                plt.plot([cord[0]],[data[dmax]],'ro')
#                plt.title('Sum along radius axis(grid size:'+ str(r_r)+'*'+str(a_r)+')')
		plt.title('Sum along radius axis')
                plt.xlabel('Angle(in degree)')
                plt.ylabel('Intensity')

                plt.annotate('angle:'+str(cord[0])+'deg', xy = cord, xytext = cord, \
                              arrowprops = dict(facecolor = 'black', shrink = 0.1))

                plt.grid()
                plt.xlim(x_deg.min(),x_deg.max())
                plt.ylim(data.min()-10,data.max()+10)
                plt.figtext(0.08,0.98,'SNR:'+str(SNR_2nd_2D)+'Angle found:'+str(int(A_f)))
                plt.plot(x_deg,data)
                p_dir = dir + '2ndFFT_2D/'
                plt.savefig(p_dir + '2ndFFT_2D_' + str(seq))
                if seq == Imag_show and SHOW==1:
                        plt.show()
                plt.close()
	        if comm_rank==0: print '2ndFFT 2D plot over...'

if __name__ == '__main__':
                exit()          



