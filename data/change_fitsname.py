#! /usr/bin/env python
import pyfits
import numpy as np
import sys
#hdulist = pyfits.open('guppi_FAST_0001_0002.fits',mode='update')
hdulist = pyfits.open('fake_test.fil',mode='update')
header	= hdulist[0].header
header.set('SRC_NAME','B0329+54')
header.set('PROJID','TianLai Pulsar')
header.set('TELESCOP','TianLai')
hdulist.close() 
