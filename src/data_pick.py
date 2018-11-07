import numpy as np
data_dir = '/home/nch/pulsar_data/wiggle_cut/'
t_block  = 2048
blok_id  = [
            [0],					     #0
            [6,10,11,12,13,14,20,22,27,28,36,37,53,55],      #1
            [4,7,10,15,21,22,25,31,32,38,40],                #2
            [1,2,5,7,9,11,17,20,27,28,29,37,40,47],          #3
            [0,5,11,13,19,21,36,38,43,44,45,48,52,53,54,55], #4
            [4,12,19,20,26,38,39], 			     #5
            [8,9,16,17,23],				     #6
            [2,3,9,10,17,25,26],			     #7
            [0],					     #8
            [55],					     #9
            [23],					     #10
            [6],					     #11
            [33,44,46],					     #12
            [41],					     #13  strong pulsar
            [0],					     #14
            [5,11,13,22,54,59]				     #15
            ]

for i in range(16):
      
      data = np.load(data_dir + str(i)+'.npy')
      data = data[:,0,:]
      for j in blok_id[i]:
          print 'begin to save data :the ',j,' time block',' in ',i,' file'
#          print str(i)+'_'+str(j),' time block'
          data1 = data[:,t_block * j : t_block * (j + 1)]
          np.save('./wigglez_found/data_pick/'+str(i)+'_'+str(j)+'.npy', data1)

print 'save over'
