import wavelets
import pyfits
import numpy as np
import math
import sys
import scipy
import matplotlib.pyplot as plt
import os
from functions import *

def plot_wavelet(wavobj, globspec, minscale, gws_fitparams, ax=None, coi=True, savename=False,cmap='option_b'):

    if not ax:
        fig = plt.figure(figsize=(12,6))
        ax = plt.subplot2grid((1, 4), (0, 0), colspan=3)
        ax2 = plt.subplot2grid((1, 4), (0, 3),sharey=ax)

        #fig, (ax, ax2) = p.subplots(1,2,sharey=True)

    
    t, s = wavobj.time, wavobj.scales
    power = wavobj.wavelet_power
    #globspec = wavobj.global_wavelet_spectrum
    
    
    #cut to scales over 1d
    globspec = globspec[s>minscale]
    power = power[s>minscale,:]
    s = s[s>minscale]
    
    Time, Scale = np.meshgrid(t, s)
    
    ax.contourf(Time, Scale, power, 100,cmap=cmap)
    ax.contour(Time, Scale, power, 100,cmap=cmap, lw=0.1)

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
    
    if savename != False:
        p.savefig(savename,bbox_inches='tight',pad_inches=0.2)
        
    return ax

def get_peak_period(wavobj,globspec,minscale):
    s = wavobj.scales[wavobj.scales>minscale]
    globspec = globspec[wavobj.scales>minscale]
    return wavobj.fourier_period(s[np.argmax(globspec)])

def plot_wavelet_slice(wavobj, globspec, minscale, gws_fitparams,minper,maxper, ax=None, coi=True, savename=False):
    t, s = wavobj.time, wavobj.scales
    power = wavobj.wavelet_power
    #globspec = wavobj.global_wavelet_spectrum
    
    
    #cut to scales over 1d
    globspec = globspec[(s>minper) & (s<maxper)]
    power = power[(s>minper) & (s<maxper)]
    s = s[(s>minper) & (s<maxper)]

    power_sum = np.sum(power,axis=0)

    p.plot(t,power_sum)
    p.show()

def my_wavelet_variance(wavobj):
    """Equivalent of Parseval's theorem for wavelets, S3.i.
    The wavelet transform conserves total energy, i.e. variance.
    Returns the variance of the input data.
    """
    # TODO: mask coi for calculation of wavelet_variance
    # is this possible? how does it change the factors?
    dj = wavobj.dj
    dt = wavobj.dt
    C_d = wavobj.C_d
    N = wavobj.N
    s = np.expand_dims(wavobj.scales, 1)
    A = dj * dt / C_d
    var = A * np.sum(np.abs(wavobj.wavelet_transform) ** 2 / s,axis=0)
    return var

def scale_averaged_power(wavobj, globspec, dj, gws_fitparams,minper,maxper, sigma, ax=None, coi=True, savename=False):
    t, s = wavobj.time, wavobj.scales
    power = abs(wavobj.wavelet_transform) ** 2
#    power = wavobj.wavelet_power
    
    dt = np.diff(t)[0]
    
    #cut to scales over 1d
    globspec = globspec[(s>minper) & (s<maxper)]
    power = power[(s>minper) & (s<maxper)]
    s = s[(s>minper) & (s<maxper)]

    weighted_avg = []

    for i in range(0,len(t)):
        weighted_avg += [sum(((abs(power[:,i])))/s)]

    weighted_avg = np.array(weighted_avg)

    dj = wavobj.dj
    dt = wavobj.dt
    C_d = wavobj.C_d

    A = dj * dt / C_d

    weighted_avg = A * weighted_avg

    return weighted_avg
