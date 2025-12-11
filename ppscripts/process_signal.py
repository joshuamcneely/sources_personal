#!/usr/bin/env python
#
# process_signal.py
#
# There is no warranty for this code
#
# @version 0.1
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     2017/12/04
# @modified 2017/12/07
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate


#---

def Blackman_Harris_w(x):
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    N = len(x)
    n = np.arange(N)
    w = a0 - a1*np.cos(2*np.pi*n/N) + a2*np.cos(4*np.pi*n/N) - a3*np.cos(6*np.pi*n/N)
    return w*x

def resample_const_time_step(time,field,dt=None,intp='linear'):
    if not dt:
        dt = min(np.diff(time))
    if intp=='linear':
        fieldintp = interpolate.interp1d(time,field)
        time = np.arange(time[0],time[-1],dt)
        field = fieldintp(time)
    elif intp in['none','crop']:
        idxmin = np.argmin(np.diff(np.diff(time)))
        print(idxmin)
        time  =  time[idxmin:]
        field = field[idxmin:]
    else:
        raise RuntimeError()

    return time,field,dt 

def zero_crop(time,field,tcrackid):
    #timenew =  np.zeros(tcrackid*2)
    #fieldnew = np.zeros(tcrackid*2)
    n=len(time)
    
    return time[:-1],field[:-1] 

def get_dt(time):
    return min(np.diff(time))

def differentiate(field,dt):
    return np.diff(np.array(field))*1.0/dt

def to_frequency_domain(time,field):
    fieldfft = np.fft.fft(field)

    n = time.shape[0]
    dt = get_dt(time)
    fs = 1/dt #sampling frequency
    print('sampling period',dt,'frequency',fs)
    s = np.array([i*fs/n for i in range(time.shape[0])])
    print('f0,f1,fs',s[0],s[1],s[-1])

    # discard redundant data
    idxmax = n//2
    fieldfft = fieldfft[:idxmax]
    s = s[:idxmax]    
    fieldamp = np.sqrt(fieldfft.real**2 + fieldfft.imag**2 )
    fieldphs = np.arctan(fieldfft.imag/fieldfft.real)

    return s,fieldamp,fieldphs


def analyse_time_series(time,field,**kwargs):
    """
    window  = False
              True => Blackman_Harris
    butterw = False
              [order, fcut] => ButterWorth
    crop    = False
              timeframe (float)
    pad     = False
              True => pad with constant
    asymmetry = False
                True => asymmetry
    """
    print(field.shape, time.shape)
    
    if field.shape!=time.shape:
        if field.shape[0]==time.shape[0]:
            field=field[:,0]
            print(field.shape,time.shape)
    if kwargs.get('crop',False):
       
        if kwargs.get('pad',False):
            print('pad')
            Dt = kwargs['crop']
            dt = get_dt(time)
            time_pad = np.arange(-Dt,Dt,dt)
            field_pad = np.zeros(len(time_pad))
            start = np.argmin(np.abs(time+Dt))
            end   = np.argmin(np.abs(time-Dt))
            start_pad = np.argmin(np.abs(time_pad-time[0]))
            end_pad   = np.argmin(np.abs(time_pad-time[-1]))                   
            print('crop',Dt,start, end, len(field))

            field_pad[start_pad:end_pad] = field[start:end]

            field_pad[end_pad:] = [field[-1] for i in range(len(field_pad[end_pad:]))]
            time=time_pad
            field=field_pad
        else:
            Dt = min(kwargs['crop'],max(time))
            start = np.argmin(np.abs(time+Dt))
            end   = np.argmin(np.abs(time-Dt))
            print('crop',Dt,start, end)
            time  =  time[start:end]
            field = field[start:end]


    if kwargs.get('asymmetry',False):
        Dt = min(max(time),max(-time))
        start = np.argmin(np.abs(time+Dt)) 
        end   = np.argmin(np.abs(time-Dt)) 
        time = time[start:end]
        field = field[start:end]
        field = field - field[-1::-1]

        print('start end field',field[0],field[-1])


    if kwargs.get('window',False):
        print('Blackman Harris window')
        field = Blackman_Harris_w(field)

    if kwargs.get('butterw',False):
        print('ButterWorth filter')
        order,fcut = kwargs['butterw']
        fny = 1.0/2.0/get_dt(time)
        b, a = signal.butter(order,fcut/fny)
        if False:
            zi = signal.lfilter_zi(b, a)
            field, _ = signal.lfilter(b, a, field, zi = zi*field[0])
        else:
            field = signal.filtfilt(b, a, field)

    print(field.shape, time.shape)

    s, fieldamp, fieldphs = to_frequency_domain(time,field)

    return time, field, s, fieldamp, fieldphs

def freq_dom_deconv(x,y):
    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    Z = Y/X
    Z = np.fft.ifft(Z)

    return Z

def read_csv(fname):
    f = open(fname,'r')
    dl=', '
    x=[]
    y=[]
    for line in f:
        if line.split(dl)[0]=='time':
            continue
        x.append(float(line.split(dl)[0]))
        y.append(float(line.split(dl)[1]))
    return np.array(x),np.array(y)

def main():
    """
    python process_signal.py

    this is an example
    """

    rawtime, rawfield = read_csv('b0_dyn_0039_x0.125_y0_z0.0025_raw_data.csv')
    raws, rawfamp,rawfphs = to_frequency_domain(rawtime,rawfield)

    time,field,s,fieldamp,fieldphs = analyse_time_series(
        rawtime,
        rawfield,
        crop=6e-5,
        pad=True,
        window=True,
        butterw=[2,2e6],# [order, fcut]        
        )

    # other useful functions

    restime, resfield, dt = resample_const_time_step(time,field,dt=1e-6,intp='linear')

    diff_field = differentiate(resfield,dt)    

    h = freq_dom_deconv(field,rawfield)

    fig, axes = plt.subplots(nrows=2)

    axes[0].plot(rawtime,rawfield,label='raw')
    axes[0].plot(time,field,label='processed')
    axes[1].semilogx(raws, rawfamp)
    axes[1].semilogx(s,fieldamp)


    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('amplitude (-)')
    
    axes[1].set_xlabel('frequency (Hz)')
    axes[1].set_ylabel('amplitude (-)')
    axes[0].legend(loc='best')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
