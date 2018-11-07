#!/bin/bash
for (( i = 1 ; $i <= 10; i=i+1 )) ; do
   for (( j = 1 ; $j <= 100; j=j+1 )) ; do
	   fake -period 5000 -width 0.02 -nbits 8 -nchans 1024 -tsamp 1000 -tobs 5 -fch1 800 -foff 0.098 -dm 100 -snrpeak $i > ./SNR$i/noise_level_$j.fil 
	done
   echo 'SNR:'$i 'done!'
done

