#!/bin/bash
for (( i = 1 ; $i <= 10; i=i+1 )) ; do
   for (( j = 1 ; $j <= 100; j=j+1 )) ; do
	#   fake -period 500 -width 0.2 -nbits 8 -nchans 1024 -tsamp 100 -tobs 0.5 -fch1 800 -foff 0.098 -dm 100 -snrpeak $i > ./SNR$i/noise_level_snr$i_$j.fil 
	rm ./SNR$i/noise_level_snr{$i}_{$j}.fil
	done
   echo 'SNR:'$i 'done!'
done

