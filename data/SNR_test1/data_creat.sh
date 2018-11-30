#!/bin/bash
#This is a script generate simulated FRB test data for heavy test
#Make directory like SNR_test1

for (( i = 1 ; $i <= 10; i=i+1 )) ; do
   for (( j = 1 ; $j <= 100; j=j+1 )) ; do
	   fake -period 10000 -width 0.1 -nbits 8 -nchans 1024 -tsamp 1000 -tobs 10 -fch1 800 -foff 0.09765 -dm 300 -snrpeak $i > ./SNR$i/noise_level_$j.fil 
	done
   echo 'SNR:'$i 'done!'
done

