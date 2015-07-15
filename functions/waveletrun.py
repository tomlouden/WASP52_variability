#takes stitched kepler data, interpolates onto an even grid, runs wavelets package on result


import wavelets
import pyfits
import numpy as np
import math
import sys
import scipy
import pylab as p
p.ion()
import os

def Run():
	
	
	#set up inputs
	infile = sys.argv[1]
	
	optionalinputs = {}	
	# Set up default values
	optionalinputs['inputcol'] = 'PDCSAP_FLUX'
	optionalinputs['inputcol_err'] = 'PDCSAP_FLUX_ERR'
	optionalinputs['initialperiod'] = None
	optionalinputs['minscale'] = 1.
	optionalinputs['binning'] = 4
	optionalinputs['cutneargaps'] = False  #otherwise is amount to cut after gaps
	
	for inputval in sys.argv[2:]:
	    key,val = inputval.split('=')
	    if key == 'initialperiod' or key == 'minscale':
	        val = float(val)
	    elif key == 'binning':
	        val = int(val)
	    elif key == 'cutneargaps':
	        val = float(val)
	    optionalinputs[key] = val
 	
	inputcol = optionalinputs['inputcol']
	inputcol_err = optionalinputs['inputcol_err']
	initialperiod = optionalinputs['initialperiod']
	minscale = optionalinputs['minscale']
	binning = optionalinputs['binning']
	cutneargaps = optionalinputs['cutneargaps']
	
	wavelet = raw_input('Input wavelet (0=Ricker, 1=Morlet): ')
	if wavelet=='1':
	    wavfunc = wavelets.Morlet
	else:
	    wavfunc = wavelets.Ricker
	    
	#read data
	if infile[-5:] == '.fits':
	    time, flux, err, t0 = ReadLCFITS(infile,inputcol,inputcol_err)
	else:
	    time, flux, err, t0 = ReadLCTXT(infile)
	
	#centre to zero
	flux -= np.median(flux)   #flux is already median normalised in each quarter, so need to subtract here
	
	print 'File Read'
	
	cadence = np.median(np.diff(time))
	
	
	#Cut near gaps
	if cutneargaps:
	    time,flux = CutAfterGaps(time,flux,cutneargaps,0.8)
	
	#fill gaps
	timezeros, fluxzeros = FillGapsZeros(time, flux, cadence)
	time, flux = FillGapsInterp(time, flux, cadence,20,10.)  #last two are points to average over before and after gap , and maximum gap length. Zeros are filled for gaps over max gap length.
	
	print 'Gaps Filled'
	
	
	#remove transits?
	
	
	#bin points up for speedier calculation
	time,flux = BinLC(time,flux,binning)
	timezeros,fluxzeros = BinLC(timezeros,fluxzeros,binning)
	print 'Lightcurve Binned'
	#interpolate onto even grid?
	
	#run wavelets
	wa = wavelets.WaveletAnalysis(flux,wavelet=wavfunc(),dt=cadence*binning,dj=0.01,unbias=True)
	
	print 'Wavelet Object Created'
	print wa.scales
	karray, acf = ACF(fluxzeros)
	
	print 'ACF Calculated'
	
	
	#plot/save?
	
	#wa.plot_power()
	wa.mask_coi = True
	
	#calculate global power spectrum for fitting
	globspec = wa.global_wavelet_spectrum
	
	gws_fitparams = FitGWS(globspec,wa.scales)
	
	amps = gws_fitparams[2::3]
	widths = gws_fitparams[1::3]
	locs = gws_fitparams[::3]
	realpeaks = locs>0
	print locs[realpeaks]
	print amps[realpeaks]
	print widths[realpeaks]
	maxpeak = np.argmax(amps[realpeaks])
	maxpeak_scale = locs[realpeaks][maxpeak]
	maxpeak_period = wa.fourier_period(maxpeak_scale)
	maxpeak_scalewidth = widths[realpeaks][maxpeak]
	maxpeak_width_plus = wa.fourier_period(maxpeak_scale + maxpeak_scalewidth) - maxpeak_period
	maxpeak_width_minus = maxpeak_period - wa.fourier_period(maxpeak_scale - maxpeak_scalewidth) 

	print 'GWS Max Peak Location = '+str(maxpeak_period) + ' + ' + str(maxpeak_width_plus)+ ' - ' + str(maxpeak_width_minus)  #half width half max
	
	acfsmoothed = SmoothACF(acf,maxpeak_period)
	
	print 'ACF Smoothed'
	
	acfperfit, acfcov, peaksused = GetACFPer(karray,acfsmoothed,cadence,binning,initialperiod)
	
	acfper,acferr = TranslateToValues(acfperfit,acfcov)
	
	print 'ACF Period = '+str(acfper) + ' +- '+str(acferr)
	
	#plot wavelet power
	plot_power_plus(wa,globspec, minscale,gws_fitparams)
	
	
	
	
	p.figure(10)
	p.clf()
	p.plot(karray*cadence*binning,acfsmoothed,'b.-')
	#p.plot(karray*cadence*binning,acf,'r.-')
	p.plot([peaksused,peaksused],[np.min(acf)-0.2,1.],'g--')
	p.plot([acfper,acfper],[np.min(acf)-0.2,1.],'r--')
	p.xlim(0,500.)
	p.ylim(np.min(acf)-0.2,1)
	p.xlabel('Correlation Timescale (d)')
	p.ylabel('ACF')
	
	p.figure(11)
	p.clf()
	p.plot(time,flux,'b.')
	p.xlabel('Time (d)')
	p.ylabel('Relative Flux')
	raw_input('Press enter to exit')



def plot_power_plus(wavobj, globspec, minscale, gws_fitparams, ax=None, coi=True):
    if not ax:
        fig = p.figure(figsize=(12,6))
        ax = p.subplot2grid((1, 4), (0, 0), colspan=3)
        ax2 = p.subplot2grid((1, 4), (0, 3),sharey=ax)

        #fig, (ax, ax2) = p.subplots(1,2,sharey=True)

    
    t, s = wavobj.time, wavobj.scales
    power = wavobj.wavelet_power
    #globspec = wavobj.global_wavelet_spectrum
    
    
    #cut to scales over 1d
    globspec = globspec[s>minscale]
    power = power[s>minscale,:]
    s = s[s>minscale]
    
    Time, Scale = np.meshgrid(t, s)

    #ax.contourf(Time, Scale, np.log10(power), 100)
    ax.contourf(Time, Scale, power, 100)

    ax.set_yscale('log')
    ax.grid(True)

    if coi:
        coi_time, coi_scale = wavobj.coi
        ax.fill_between(x=coi_time,
                        y1=coi_scale,
                        y2=s.max(),
                        color='gray',
                        alpha=0.3)

    ax2.semilogy(globspec,s,'b')
    ax2.semilogy(Get_MultiGauss(gws_fitparams,s,len(gws_fitparams[::3])),s,'r--')
    
    maxscale = np.argmax(globspec)
    print 'Wavelet GPS Peak Scale: ' + str(s[maxscale])
    print 'Wavelet GPS Peak Period: ' + str(wavobj.fourier_period(s[maxscale]))
    ax2.plot([0,np.max(globspec)],[s[maxscale],s[maxscale]],'r--')
    
    p.setp(ax2.get_yticklabels(),visible=False)
    p.setp(ax2.get_xticklabels(),visible=False)
    #ax2.yaxis.tick_right()
    
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(s.min(), s.max())

    
    fig.subplots_adjust(wspace=0)
    
    ax.set_ylabel('Scales (d)')
    ax.set_xlabel('Time (d)')

    ticks = np.unique(2 ** np.floor(np.log2(s[s>=1.])))[1:]
    ax.yaxis.set_ticks(ticks)
    ax.yaxis.set_ticklabels(ticks.astype(str))
    #ax.set_ylim(64, 0.5)
        
    ax_fourier = ax2.twinx()
    ax_fourier.set_yscale('log')
    # match the fourier ticks to the scale ticks
    ax_fourier.set_yticks(ticks)
    

    fourier_labels = []
    for tick in ticks:
        fourier_labels.append(np.round(wavobj.fourier_period(tick),decimals=2))
    
    ax_fourier.set_yticklabels(fourier_labels)
    ax_fourier.set_ylabel('Equivalent Fourier Period (d)')
    fourier_lim = [wavobj.fourier_period(i) for i in ax2.get_ylim()]
    ax_fourier.set_ylim(fourier_lim)   
    

    savecheck = raw_input('Save? Type in filename if so: ')
    #fig.set_size_inches(40.5, 10.5, forward=True)
    if savecheck:
        p.gcf().set_rasterized(True)
    
        p.savefig(os.path.join('/Users/davidarmstrong/Data/Kepler/Habitables/Plots/',savecheck),bbox_inches='tight',pad_inches=0.2)
    
    
    return ax

def CutAfterGaps(time,flux,cutamount,gapthreshold):
    outtime = [time[0]]
    outflux = [flux[0]]
    lastgap = 0
    
    for point in range(len(flux))[1:]:
        if time[point] - time[point-1] > gapthreshold:
            lastgap = time[point]
        
        if time[point]-lastgap>cutamount and lastgap != 0:
            outtime.append(time[point])
            outflux.append(flux[point])
    return np.array(outtime),np.array(outflux)
            
        
    

def ACF(sig,kmin=1,kmax=None):
    if not kmax:
        kmax = np.floor(len(sig)/2.)
    sigmean = np.mean(sig)
    normaliser = np.sum(np.power(sig-sigmean,2))
    output = []
    for k in np.arange(kmin,kmax,1):
        output.append( np.sum( (sig[:len(sig)-k] - sigmean) * (sig[k:] - sigmean) ) / normaliser )
    return np.arange(kmin,kmax,1),np.array(output)

def GetACFPer(karray, acf, cadence, binning, initialperiod=None):
    peaks = FindPeaks(acf)
    peakvalues = karray[peaks]*cadence*binning
    
    if initialperiod:
        peakindex = np.argmin(np.abs(peakvalues - initialperiod))
    else:
        peakindex = 0
    
    peakstofit = [0.]
    peakindices = [0.]
    
    for i in np.arange(5)+1:
        peakmultiple = peakvalues[peakindex] * i
        closestpeak = np.argmin(np.abs(peakmultiple - peakvalues))
        print peakmultiple
        print closestpeak
        print peakvalues[closestpeak]
        if math.fabs(peakvalues[closestpeak]-peakmultiple) < peakmultiple * 0.1:  #10% may be too lenient
            peakstofit.append(peakvalues[closestpeak])
            peakindices.append(i)
        
    print np.array(peakindices)
    print np.array(peakstofit)
    if len(peakindices) > 2:
        perfit,cov = np.polyfit(np.array(peakindices),np.array(peakstofit),1,cov=True)
    else:
        perfit = [peakvalues[peakindex],0]
        cov = np.array([[0,0],[0,0]])
    print perfit, cov
    return perfit,cov, np.array(peakstofit)
            
def FindPeaks(acf):

    peaks = []
 
    for val in range(len(acf))[1:-1]:
        if (acf[val-1:val] < acf[val]).all() and (acf[val+1:val+2] < acf[val]).all():
            peaks.append(val)
    return peaks

def TranslateToValues(perfit,cov):
    per = perfit[0]
    pererr = np.sqrt(np.abs(cov[0,0])) #assumes diagonal, which is not really true
    
    return per, pererr

def SmoothACF(acf,sigma):
    return scipy.ndimage.filters.gaussian_filter(acf,sigma,truncate=3.1)          

def ReadLCFITS(infile,inputcol,inputcol_err):
    dat = pyfits.open(infile)
    time = dat[1].data['TIME']
    t0 = time[0]
    time -= t0
    nanstrip = time==time
    time = time[nanstrip]
    flux = dat[1].data[inputcol][nanstrip]
    err = dat[1].data[inputcol_err][nanstrip]
    fluxnanstrip = flux==flux
    time = time[fluxnanstrip]
    flux = flux[fluxnanstrip]
    err = err[fluxnanstrip]
    return time, flux, err, t0

def ReadLCTXT(infile):
    dat = np.genfromtxt(infile)
    t0 = dat[0,0]
    nanstrip = np.isnan(dat[:,1]) | np.isinf(dat[:,1])
    return dat[~nanstrip,0]-t0,dat[~nanstrip,1],dat[~nanstrip,2],t0



def FillGaps(time,flux,cadence):
    insertedflag = np.zeros(len(flux))

    #interpolate data onto even grid (fill flat line of ones into gaps?)
    diffs = np.diff(time)

    ntoinsert = np.round(diffs/cadence) - 1
    for i in range(len(diffs)):
        reversedindex = len(diffs) - i - 1
        if ntoinsert[reversedindex] > 0:
            flux = np.insert(flux,np.ones(ntoinsert[reversedindex])* (reversedindex+1), np.ones(ntoinsert[reversedindex])) #inserts ones into gaps
            time = np.insert(time,np.ones(ntoinsert[reversedindex])* (reversedindex+1), time[reversedindex]+(np.arange(ntoinsert[reversedindex])+1)*cadence)

    return time,flux

def FillGapsInterp(time,flux,cadence,averaging,maxgaplength):
    outtime = [time[0]]
    outflux = [flux[0]]

    for point in range(len(flux))[1:]:
        timerange = time[point] - time[point-1] 
        
        if timerange > maxgaplength:
            ntoinsert = int(np.round((time[point]-time[point-1]) / cadence) - 1)
            for insertion in range(ntoinsert):
                inserttime = time[point-1]+cadence*(insertion+1)
                outtime.append(inserttime)
                outflux.append(0.0)
 
         
        elif timerange > cadence*1.5:
            ntoinsert = int(np.round((time[point]-time[point-1]) / cadence) - 1)
            if point>averaging:
                startflux = np.mean(flux[point-averaging-1:point-1])
            else:
                startflux = flux[point-1]
            if point<len(flux)-averaging-1:
                endflux = np.mean(flux[point:point+averaging])  #this and above section means that interpolation is not as biased by one odd point
            else:
                endflux = flux[point]
            


            for insertion in range(ntoinsert):
                inserttime = time[point-1]+cadence*(insertion+1)
                outtime.append(inserttime)
                outflux.append((inserttime-time[point-1])/timerange * (endflux-startflux) + startflux)

        outtime.append(time[point])
        outflux.append(flux[point])
    return np.array(outtime),np.array(outflux)

def FillGapsZeros(time,flux,cadence):
    outtime = [time[0]]
    outflux = [flux[0]]

    for point in range(len(flux))[1:]:
        timerange = time[point] - time[point-1]  
        if timerange > cadence*1.5:
            ntoinsert = int(np.round((time[point]-time[point-1]) / cadence) - 1)

            for insertion in range(ntoinsert):
                inserttime = time[point-1]+cadence*(insertion+1)
                outtime.append(inserttime)
                outflux.append(0.0)

            outtime.append(time[point])
            outflux.append(flux[point])       
  
        else:
            outtime.append(time[point])
            outflux.append(flux[point])
    return np.array(outtime),np.array(outflux)

def BinLC(time,flux,npoints):  #assumes no gaps, i.e. run one of the FillGaps algorithms first

    outtime = []
    outflux = []
    for i in range(int(np.floor(len(flux)/npoints))):
        outflux.append(np.mean(flux[i*npoints:(i+1)*npoints]))
        outtime.append(np.mean(time[i*npoints:(i+1)*npoints]))
    return np.array(outtime),np.array(outflux)


def FitGWS(globspec,scales):

    peaks = FindPeaks(globspec)
    
    nGauss = len(peaks)

    fitparams = FitGaussians(nGauss,peaks,scales,globspec,scales[peaks]*0.25,globspec[peaks])
    
    #offset = globspec - Get_MultiGauss(fitparams,scales,nGauss)
    
    
    return fitparams
    #smallestgauss = np.argmin(fitparams[::3])
    
    #params = []


def FitGaussians(N,peaks,scales,globspec,width,amps):
    params = []
    for i in range(N):
        params.append(scales[peaks[i]])  #initial peak locations
        params.append(width[i])  #initial guess
        params.append(amps[i])   #initial guess

    fitted_params,_ = scipy.optimize.leastsq(MultiGauss, params, args=(globspec,scales,N),maxfev=10000)
    return fitted_params


def MultiGauss(p, gws, x, N):
    locs = p[::3]
    widths = np.abs(p[1::3])
    amps = np.abs(p[2::3])
    #if np.any(amps<0):
    #    return 1e10
    
    ret = 0
    for i in range(N):
        ret += amps[i]*np.exp(-(x-locs[i])**2/(2.*widths[i]**2))

    return np.abs(gws - ret)

def Get_MultiGauss(p,x,N):
    locs = p[::3]
    widths = p[1::3]
    amps = np.abs(p[2::3])
    
    ret = 0
    for i in range(N):
        ret += amps[i]*np.exp(-(x-locs[i])**2/(2.*widths[i]**2))
    return ret

if __name__=='__main__':
    Run()