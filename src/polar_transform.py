import scipy.signal as signal
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def polar_coordinates_convert(data, angle ):
	  """
		Transform the rectanular coordinates into polar coordinates
    	  Note ..  Take histogram2D method
	  param:
		data:	Data want to transform 
		angle:	a list for [Angle_max, Angle_min]	  
	  Return: 
		data in polar-coordinate I(r,theta)
		Angle resolution
		Radius Resolution
	  """

	  ang_min = angle[0]
	  ang_max = angle[1]
          rang    = data.shape
          ang_rsl = (1./rang[1])*180/np.pi
	  rad_rsl =  1
	  
	  #take the left top of the matrix as center
          row     = np.arange(-rang[0]+1,1,dtype=np.float32) 
          rank    = np.arange(rang[1],dtype=np.float32)

	  # calculate the angle of each pixe
          angle   = np.nan_to_num(np.arctan(row[:,None]/rank[None,:])) / np.pi * 180 

	  # calculate radius of each pixel
          radius  = np.sqrt(row[:,None]**2 + rank[None,:]**2)
          ang     = -angle.reshape(-1)
          rad     = radius.reshape(-1)
          data    = data.reshape(-1)
	  rad_bin = np.arange(1,rang[1],rad_rsl)
	  ang_bin = np.arange(ang_min,ang_max,ang_rsl)
	  data_r,bin_eage_x,bin_eage_y   = np.histogram2d(rad, ang, weights=np.real(data), bins= (rad_bin,ang_bin))
	  data_i, bin_eage_x,bin_eage_y  = np.histogram2d(rad, ang, weights=np.imag(data), bins= (rad_bin,ang_bin))
	  polar_data   = data_r + data_i * 1j

          return  polar_data,ang_rsl,rad_rsl

def polar_coordinates_convert_inter(data, angle,n_deg,L_fft=0):
          """
                Transform the rectanular coordinates into polar coordinates
          Note ..  Take interpolation method
          param:
                data:   Data want to transform 
                angle:  a list for [Angle_max, Angle_min]         
		n_deg:	number of angle axis length
          Return: 
                data in polar-coordinate I(r,theta) \n with shape [n_rad,n_deg]
          """

	  rang     = data.shape
	  ang_min  = angle[0]
	  ang_max  = angle[1]
	  tan_min  = np.tan(ang_min*np.pi/180)
	  tan_max  = np.tan(ang_max*np.pi/180)
	  
	  n_deg	  = n_deg
	  n_rad   = max(data.shape)
	  rad_grid = np.linspace(1,rang[1]-1,n_rad)
          tan_min  = np.tan(ang_min*np.pi/180)
          tan_max  = np.tan(ang_max*np.pi/180)
	  tan_grid  = np.linspace(tan_min,tan_max,n_deg)
	  ang_grid  = np.arctan(tan_grid)/np.pi*180
	  grid_a,grid_r = np.meshgrid(ang_grid,rad_grid)

	  x_p     = grid_r * np.cos(grid_a*np.pi/180.) -1
	  y_p     = rang[0] - grid_r * np.sin(grid_a*np.pi/180.) 
	  x_p     = x_p.reshape(-1)
	  y_p     = y_p.reshape(-1)
	  cord    = [y_p,x_p]
	  polar_matrix_r  = ndimage.map_coordinates(np.real(data),cord,order=0)
	  polar_matrix_i  = ndimage.map_coordinates(np.imag(data),cord,order=0)
	  polar_data    = polar_matrix_r+polar_matrix_i*1j
 	  polar_data    = polar_data.reshape(grid_r.shape)	
	  return polar_data


if __name__ == '__main__':

	exit(0)

