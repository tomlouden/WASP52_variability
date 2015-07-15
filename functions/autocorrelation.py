import numpy as np
import itertools
import multiprocessing
import random as rnd
from util import *

def autocorrelate(x,t):
	result = np.correlate(x, x, mode='full')
	result = result[(result.size/2)+1:]
	periods = np.arange(1,len(result)+1)*min(np.diff(t))
	return result, periods

def aUDCF(a,a_e,t,nproc=16):
	udcf, udcf_err, dt = UDCF(a,a,a_e,a_e,t,nproc)	
	return udcf, udcf_err, dt

def UDCF(a,b,a_e,b_e,t,nproc=16):

	a_mean = np.mean(a)
	b_mean = np.mean(b)
	a_sig = np.std(a)
	b_sig = np.std(b)

	a_m_err = np.mean(a_e)
	b_m_err = np.mean(b_e)

	p = multiprocessing.Pool(nproc)

	curry = []
	for i in range(0,len(a)):
		curry += [[i,a,a_e,t,a_mean,b_mean,a_sig,b_sig,a_m_err,b_m_err]]
	output = p.map(uncurry_mUDCF2,curry)
	output = np.array(output)
	udcf = []
	udcf_err = []
	dt = []

	for i in range(0,len(output[:,0])):
		udcf += list(output[:,0][i])
		udcf_err += list(output[:,1][i])
		dt += list(output[:,2][i])
	udcf = np.array(udcf)
	udcf_err = np.array(udcf_err)
	dt = np.array(dt)
	p.close()

	return udcf, udcf_err, dt

def uncurry_mUDCF2(x):
	return mUDCF2(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])

def mUDCF2(k,x,err,t,a_mean,b_mean,sig_a,sig_b,a_m_err,b_m_err):

	a = x[:len(x)-k]
	b = x[k:]
	a_e = err[:len(x)-k]
	b_e = err[k:]
	t1 = t[:len(x)-k]
	t2 = t[k:]

#	a =  (list(x[-k:]) +list(x[:len(x)-k]))
#	b = x.copy()
#	a_e = (list(err[-k:]) +list(err[:len(x)-k]))
#	b_e = err.copy()
#	t1 = (list(t[-k:]) +list(t[:len(x)-k]))
#	t2 = t.copy()

	udcf = []
	udcf_err = []
	delta_t = []

	for i in range(0,len(a)):
		result = mUDCF(a[i],b[i],a_e[i],b_e[i],a_mean,b_mean,sig_a,sig_b,t1[i],t2[i],a_m_err,b_m_err)
		udcf += [result[0]]
		udcf_err += [result[1]]
		delta_t += [result[2]]

	return udcf, udcf_err, delta_t

def uncurry_mUDCF(x):
	return mUDCF(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7])

def mUDCF(a,b,a_e,b_e,a_mean,b_mean,sig_a,sig_b,t_i,t_j,a_m_err,b_m_err):

	norm = np.sqrt((sig_a**2 - a_m_err**2)*(sig_b**2 - b_m_err**2))

	udcf = (a -a_mean)*(b -b_mean)/norm

	delta_t = t_j - t_i

	udcf_err = np.sqrt(((a - a_mean)*b_e)**2 + ((b - b_mean)*a_e)**2)/norm
	udcf_err2 = np.sqrt(udcf**2)*((a_e/a)**2 + (b_e/b)**2)**0.5

#	if delta_t == 0:
#		print a, b, a_e,b_e,sig_a,sig_b, 'values'
#		print udcf, udcf_err, udcf_err2, 'errors'

	return udcf, udcf_err, delta_t

def unweighted_DCF(udcf,udcf_err,delta_t,nbins):

	delta_tau = max(delta_t)/nbins

	tau = []
	dcf = []
	dcf_err = []

	for i in range(0,(nbins-1)):
		lower = (i)*delta_tau
		upper = (i+1)*delta_tau
		selection = (delta_t > lower) & (delta_t < upper)
		if len(udcf[selection] > 0):
			dcf += [np.mean(udcf[selection])]
			dcf_err += [np.std(udcf[selection]) / np.sqrt(len(udcf[selection]))]
			tau += [(lower + upper)/2]

	tau = np.array(tau)
	dcf = np.array(dcf)
	dcf_err = np.array(dcf_err)

	return dcf, dcf_err, tau

def test():
	print '3'

def DCF(udcf,udcf_err,delta_t,nbins):

	delta_tau = max(delta_t)/nbins

	tau = []
	dcf = []
	dcf_err = []

	for i in range(0,(nbins-1)):
		lower = (i)*delta_tau
		upper = (i+1)*delta_tau
		selection = (delta_t >= lower) & (delta_t <= upper)
		if len(udcf[selection] > 0):
			w_mean, w_err = weighted_mean(udcf[selection],udcf_err[selection])
			dcf += [w_mean]
			dcf_err += [w_err]
			tau += [(lower + upper)/2]

	tau = np.array(tau)
	dcf = np.array(dcf)
	dcf_err = np.array(dcf_err)

	return dcf, dcf_err, tau