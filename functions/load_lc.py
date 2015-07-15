# -*- coding: utf-8 -*-
import numpy as np

def load_wasp_lc(fname):

	wasp_dict = {'HJD':[],'mag':[],'err':[]}
	i = 0
	for line in open(fname):
  		if i > 4:
	  		if line[0] == '#':
	  			splitted = line.strip('\n').strip('#').split(':')
	  			if (len(splitted[0]) > 0):
	  				if (len(splitted) > 1):
	  					if splitted[0][:7] != ' Column':
		  					wasp_dict[splitted[0].replace(' ','')] = splitted[1]
		  	else:
		  		splitted = line.strip('\n').split()
		  		wasp_dict['HJD'] += [float(splitted[0])]
		  		wasp_dict['mag'] += [float(splitted[1])]
		  		wasp_dict['err'] += [float(splitted[2])]
		i += 1

	wasp_dict['HJD'] = np.array(wasp_dict['HJD'])
	wasp_dict['mag'] = np.array(wasp_dict['mag']) + float(wasp_dict['MeanSWmag'])
	wasp_dict['err'] = np.array(wasp_dict['err'])


	return wasp_dict