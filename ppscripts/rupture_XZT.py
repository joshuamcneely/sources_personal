#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt

import ifasha.datamanager as idm
from ifasha.rupturehunter import RuptureHunter

import ppscripts.postprocess as pp
from ppscripts.save_ruptures import load_ruptures

def get_rupture_xzt(sname, **kwargs):
    
    # in cases snames are actually sim-ids
    sname = pp.sname_to_sname(sname)
    
    rpts=load_ruptures(sname,**kwargs)

    X,Y,T=rpts[0].get_sorted_front_XYT()
    return X,Y,T

def plot_rupture_contour(snames,**kwargs):
    wdir = kwargs.get('wdir','./data')
    start_time = kwargs.get('start_time',0.0)
    end_time = kwargs.get('end_time',np.inf)
    for sname in snames:
        dma = idm.DataManagerAnalysis(sname,wdir)
        group='interface'
        data = dma(group)
        t=data.get_full_field(idm.FieldId('time'))
        tmax = min(tmax,np.nanmax(t))
    times = np.linspace(start_time,min(end_time,tmax),10)

    for sname,c in zip(snames,plt.rcParams['axes.prop_cycle'].by_key()['color']):
        X,Y,T=get_rupture_xzt(sname,**kwargs)
        plt.contour(X,Y,T,colors=c,linestyles='-',levels=times)
        plt.plot([],[],'-',color=c,label=sname)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.legend(loc='best')

# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.exit('Missing argument! usage: ./rupture_XZT.py <sname1>,<sname2>,<...>')

    argv1 = str(sys.argv[1]).split(',')
    print(len(argv1))
    plot_rupture_contour(argv1)

    plt.show()

