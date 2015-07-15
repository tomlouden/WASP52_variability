import numpy as np

def sum_duplicates(hjd,mag,err):

	o = np.argsort(hjd)
	hjd = hjd[o]
	mag = mag[o]
	err = err[o]
	diffs = np.diff(hjd)

	delete_indexes = []

	for i in range(0,len(diffs)):
		if diffs[i] == 0:
			c = ((1.0/err[i])**2 + (1.0/err[i+1])**2)
			newval = (mag[i]/err[i]**2 + mag[i+1]/err[i+1]**2)/c
			newerr = np.sqrt((1.0/(c*err[i]))**2 + (1.0/(c*err[i+1]))**2)
			err[i+1] = newerr
			mag[i+1] = newval
			delete_indexes += [i]

	hjd = np.delete(hjd, delete_indexes)
	mag = np.delete(mag, delete_indexes)
	err = np.delete(err, delete_indexes)

	return hjd, mag, err