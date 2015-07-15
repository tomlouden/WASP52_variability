import numpy as np

def weighted_mean(x,errors):

	weighted_sum = 0.0
	c = 0.0
	weighted_error_sum = 0.0

	for i in range(0,len(x)):
		c += (1.0/errors[i])**2
		weighted_sum += x[i]/errors[i]**2

	for i in range(0,len(x)):
		weighted_error_sum += (1.0/(c*errors[i]))**2

	w_mean = weighted_sum/c
	w_error = np.sqrt(weighted_error_sum)

	return w_mean, w_error

def time_binning(x,xerr,t,dt,ends=False):


	if ends == False:
		time_dif = max(t) - min(t)
		nbins = int(time_dif/dt)
		start = min(t)
	else:
		time_dif = ends[1] - ends[0]
		nbins = int(time_dif/dt)
		start = ends[0]

	bin_x =[]
	bin_x_err =[]
	bin_t =[]

	for i in range(0,(nbins-1)):
		lower = (i)*dt + start
		upper = (i+1)*dt + start
		selection = (t >= lower) & (t < upper)
		if len(x[selection] > 0):
			w_mean, w_err = weighted_mean(x[selection],xerr[selection])
			bin_x += [w_mean]
			bin_x_err += [w_err]
			bin_t += [(lower + upper)/2]
		else:
			bin_x += [0.0]
			bin_x_err += [0.0]
			bin_t += [(lower + upper)/2]

	bin_x = np.array(bin_x)
	bin_x_err = np.array(bin_x_err)
	bin_t = np.array(bin_t)

	return bin_t, bin_x, bin_x_err

def phase_binning(x,xerr,t,period,epoch,dp,ends=False):

	phase = (t-epoch)/period

	for i in range(0,len(phase)):
		phase[i] -= int(phase[i])

	if ends == False:
		phase_diff = max(phase) - min(phase)
		nbins = int(phase_diff/dp)
		start = 0.0
	else:
		phase_diff = ends[1] - ends[0]
		nbins = int(phase_diff/dp)
		start = ends[0]

	bin_x =[]
	bin_x_err =[]
	bin_phase =[]

	for i in range(0,(nbins-1)):
		lower = (i)*dp + start
		upper = (i+1)*dp + start
		selection = (phase >= lower) & (phase < upper)
		if len(x[selection] > 0):
			w_mean, w_err = weighted_mean(x[selection],xerr[selection])
			bin_x += [w_mean]
			bin_x_err += [w_err]
			bin_phase += [(lower + upper)/2]
		else:
			bin_x += [0.0]
			bin_x_err += [0.0]
			bin_phase += [(lower + upper)/2]

	bin_x = np.array(bin_x)
	bin_x_err = np.array(bin_x_err)
	bin_phase = np.array(bin_phase)

	return bin_phase, bin_x, bin_x_err
