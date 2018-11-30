import os

def dir_create(plot_dir,file_name):
     '''
     Create plot directory
     '''    
     file_dir = '../data/'+file_name[:-4]
     if not os.path.exists(file_dir):
	       os.mkdir(file_dir)
     if not os.path.exists(plot_dir):
               print 'begin to make dir:'+ plot_dir 
               os.mkdir(plot_dir)
     if not os.path.exists(plot_dir + 'raw')  :
               print 'begin to make dir:'+ plot_dir + 'raw/' 
               os.mkdir(plot_dir+'raw/')
     if not os.path.exists(plot_dir + 'rebin/')  :
               print 'begin to make dir:'+ plot_dir + 'rebin/' 
               os.mkdir(plot_dir + 'rebin/')
     if not os.path.exists(plot_dir + '1stFFT/')  :
               print 'begin to make dir:'+ plot_dir + '1stFFT/' 
               os.mkdir(plot_dir + '1stFFT/')
     if not os.path.exists(plot_dir + 'polar_sets_2D/')  :
               print 'begin to make dir:'+ plot_dir + 'polar_sets_2D/' 
               os.mkdir(plot_dir + 'polar_sets_2D/')
     if not os.path.exists(plot_dir + 'polar_sets_3D')  :
               print 'begin to make dir:'+ plot_dir + 'polar_sets_3D/' 
               os.mkdir(plot_dir + 'polar_sets_3D/')
     if not os.path.exists(plot_dir + '2ndFFT_2D/')  :
               print 'begin to make dir:'+ plot_dir + '2ndFFT_2D/' 
               os.mkdir(plot_dir + '2ndFFT_2D/')
     if not os.path.exists(plot_dir + '2ndFFT_3D/')  :
               print 'begin to make dir:'+ plot_dir + '2ndFFT_3D/' 
               os.mkdir(plot_dir + '2ndFFT_3D/')
#     if not os.path.exists(plot_dir + 'SNR/')  :
#               print 'begin to make dir:'+ plot_dir + 'SNR/' 
#               os.mkdir(plot_dir + 'SNR/')
#     if not os.path.exists(plot_dir + 'Location/')  :
#               print 'begin to make dir:'+ plot_dir + 'Location/' 
#               os.mkdir(plot_dir + 'Location/')

