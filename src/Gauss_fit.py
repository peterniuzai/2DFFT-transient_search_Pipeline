import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from scipy.optimize import curve_fit
from scipy import optimize

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
#    mu,sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
#    return 1./((2*np.pi)**0.5*sigma)*np.exp(-(x-mu)**2/(2.*sigma**2))

def f_3(x, A, B, C, D):
    return A*x*x*x + B*x*x + C*x + D

def Multi_gaussian(x,*param):

    '''
    Multi-Peak Gauss Fit
    '''
    return param[0]*np.exp(-np.power(x - param[2], 2.) / (2 * np.power(param[4], 2.)))+\
           param[1]*np.exp(-np.power(x - param[3], 2.) / (2 * np.power(param[5], 2.)))

p0 = [1, 0., 1]
#p0 = [0,1]

d0 = np.load('../data/nosource.npy')
d1 = np.load('../data/source_5.npy')
d2 = np.load('../data/source_7.npy')
d3 = np.load('../data/source_10.npy')
#print d.size,d1.size,d2.size,d3.size
nbin=int(abs(d0.real.max()-d0.real.min()))
#plt.hist2d(d.reshape(-1).real,d.reshape(-1).imag,bins=100,normed=True)
#plt.colorbar()
#plt.show()
#exit()
plt.figure(figsize=(8,9))
N_test = 4
for i in range(N_test):
	exec ("d%s_r = d%s.real.reshape(-1)"%(i,i))
	exec ("d%s_i = d%s.imag.reshape(-1)"%(i,i))
	exec ("d%s = abs(d%s)"%(i,i))
	exec ("a%s_r, b%s_r =np.histogram(d%s_r,bins=nbin,density=True)"%(i,i,i))
	exec ("a%s_i, b%s_i =np.histogram(d%s_i,bins=nbin,density=True)"%(i,i,i))
	#Fit Gauss
	exec ("b%s_r_centers = (b%s_r[:-1] + b%s_r[1:])/2"%(i,i,i))
	exec ("b%s_i_centers = (b%s_i[:-1] + b%s_i[1:])/2"%(i,i,i))	
	exec ("a%s_r_coeff,a%s_r_var_matrix = curve_fit(gauss,b%s_r_centers,a%s_r,p0=p0)"%(i,i,i,i))
	#, maxfev=500000)
	#print a0_r_coeff
	#exit()
	exec ("a%s_i_coeff,a%s_i_var_matrix = curve_fit(gauss,b%s_i_centers,a%s_i,p0=p0)"%(i,i,i,i))
	# Get the fitted curve
	exec ("a%s_r_fit  = gauss(b%s_r_centers,*a%s_r_coeff)"%(i,i,i))
	exec ("a%s_i_fit  = gauss(b%s_i_centers,*a%s_i_coeff)"%(i,i,i))
	print '###A%s Test####'%i
	exec ("print 'Fitted real part mean =', a%s_r_coeff[1]"%(i))
	exec ("print 'Fitted imag part mean =', a%s_i_coeff[1]"%(i))
	exec ("print 'Fitted standard deviation(real part) = ', a%s_r_coeff[2]"%(i))
	exec ("print 'Fitted standard deviation(imag part) = ', a%s_i_coeff[2]"%(i))
	print '---------'
	
	alfa = 0.99
	
	threshold = 8*a0_r_coeff[2]
	alfa = norm.cdf(threshold,loc=a0_r_coeff[1],scale=a0_r_coeff[2])

plt.subplot(221)
plt.title("Real part")
for i in range(N_test):
	exec("plt.plot( b%s_r_centers, a%s_r, '.',label='a%s')"%(i,i,i))
	exec("plt.plot( b%s_r_centers, a%s_r_fit,label='a%s-Fit')"%(i,i,i))

plt.axhline(y=0,linestyle='-.',color='black')
plt.axvline(x=norm.ppf(alfa,loc=a0_r_coeff[1],scale=a0_r_coeff[2]),linestyle='--',color='red')
plt.axvline(x=norm.ppf((1-alfa),loc=a0_r_coeff[1],scale=a0_r_coeff[2]),linestyle='--',color='red')
plt.legend(frameon=False)

plt.subplot(222)
plt.title("Imaginary part")
for i in range(N_test):
        exec("plt.plot( b%s_i_centers, a%s_i, '.',label='a%s')"%(i,i,i))
	exec("plt.plot( b%s_i_centers, a%s_i_fit,label='a%s-Fit')"%(i,i,i))

plt.axhline(y=0,linestyle='-.',color='black')
plt.axvline(x=norm.ppf(alfa,loc=a0_i_coeff[1],scale=a0_i_coeff[2]),linestyle='--',color='red')
plt.axvline(x=norm.ppf((1-alfa),loc=a0_i_coeff[1],scale=a0_i_coeff[2]),linestyle='--',color='red')
plt.legend(frameon=False)

for i in range(N_test):
        exec ("a%s_r_fit_nor = (d%s_r-a%s_r_coeff[1]) / (a%s_r_coeff[2])"%(i,i,i,i))
	exec ("a%s_i_fit_nor = (d%s_i-a%s_i_coeff[1]) / (a%s_i_coeff[2])"%(i,i,i,i))

plt.subplot(223)
plt.title("Real part(Normal)")
for i in range(N_test):
#        exec("plt.plot( b%s_r_centers, a%s_r_fit, '.',label='a%s')"%(i,i,i))
#        exec("plt.plot( b%s_r_centers, a%s_r_fit_nor,label='a%s-Fit')"%(i,i,i))
	exec("plt.hist( a%s_r_fit_nor,histtype='step',bins=nbin,label='a%s-Fit',density=True)"%(i,i))

#        alfa = norm.cdf(8)
	alfa = 0.95

plt.axhline(y=0,linestyle='-.',color='black')
plt.axvline(x=0,linestyle='-.',color='yellow')
plt.axvline(x=norm.ppf(alfa),linestyle='--',color='red')
plt.axvline(x=norm.ppf(1-alfa),linestyle='--',color='red')
plt.legend(frameon=False)

plt.subplot(224)
plt.title("Imaginary part(Normal)")
for i in range(N_test):
#        exec("plt.plot( b%s_i_centers, a%s_i_fit, '.',label='a%s')"%(i,i,i))
#        exec("plt.plot( b%s_i_centers, a%s_i_fit_nor,label='a%s-Fit')"%(i,i,i))
	exec("plt.hist( a%s_i_fit_nor,bins=nbin,histtype='step',density=True, label='a%s-Fit')"%(i,i))
plt.axhline(y=0,linestyle='-.',color='black')
plt.axvline(x=0,color='yellow',linestyle='-.')
plt.axvline(x=norm.ppf(alfa),linestyle='--',color='red')
plt.axvline(x=norm.ppf(1-alfa),linestyle='--',color='red')
plt.legend(frameon=False)


plt.show()














#Fit Gauss
#a_coeff,a_var_matrix = curve_fit(f_3,b_centers,a,p0=p1)
#a_coeff,a_var_matrix = curve_fit(Multi_gaussian,b_centers,a,p0=p2)
#a_fit   = f_3(b_centers,*a_coeff)
#a_fit	= Multi_gaussian(b_centers,*a_coeff)








exit(1)
plt.figure(figsize=(7,17))
plt.subplot(321)
res = stats.probplot(d1.reshape(-1).real, plot=plt)
plt.subplot(322)
res = stats.probplot(d1.reshape(-1).imag, plot=plt)
plt.subplot(323)
res = stats.probplot(d2.reshape(-1).real, plot=plt)
plt.subplot(324)
res = stats.probplot(d2.reshape(-1).imag, plot=plt)
plt.subplot(325)
res = stats.probplot(d3.reshape(-1).real, plot=plt)
plt.subplot(326)

plt.plot(b_centers, a, label='No source')
plt.plot(b_centers,a_fit,label='No source,Fit')
plt.plot(b1_centers, a1, label='SNR: 5')
plt.plot(b1_centers,a1_fit,label='SNR:5,Fit')
plt.plot(b2_centers, a2, label='SNR: 7')
plt.plot(b2_centers,a2_fit,label='SNR:7,Fit')
plt.plot(b3_centers, a3, label='SNR:10')
plt.plot(b3_centers,a3_fit,label='SNR:10,Fit')

plt.legend(frameon=False)
plt.show()
