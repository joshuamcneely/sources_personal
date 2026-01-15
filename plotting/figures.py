#!/usr/bin/env python3
from __future__ import print_function
from __future__ import division
import sys
import os
sys.path.insert(0, '/Users/joshmcneely/sources')

from matplotlib import rcParams

from ppscripts.space_time import *
from ppscripts.at_position import *
from ifasha.datamanager import FieldId

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import numpy as np

import solidmechanics.definitions as smd
from solidmechanics import linearelasticity as matlaw
from scipy import signal
from matplotlib.colors import ListedColormap

def plot_xt(bname,group,fld,ax,my_cmap,tlim=[0,5],wdir='/Users/joshmcneely/sources/ppscripts/data/'):
    X,T,F=get_space_time(bname,group,fld,end_fct=0.5,start_time=tlim[0]*1e-3,end_time=tlim[1]*1e-3,wdir=wdir)
    fmin=1e-6
    F[F<fmin]=fmin
    pcm= ax.pcolormesh(T*1e3,X,F,
                       norm=colors.LogNorm(vmin=3e-4, vmax=F.max()),
                       cmap=my_cmap,
                       linewidth=0,
                       edgecolors='face',
                       shading='gouraud',
                       rasterized=True)
    return pcm
    
def plot_stations(bname,group,fld,scale,stations,ax,wdir='/Users/joshmcneely/sources/ppscripts/data/'):
    for pos in stations:
        time,field = get_at_position(bname, group, fld, [pos,0], wdir=wdir)
        ax.plot(time*1e3,-(field - field[0])*scale + pos,'-k', rasterized=True)

    f=field-field[0]
    fstd=np.std(f)
    
    trpt=0
    for t,y in zip(time,f):
        if y>0.05*fstd:
            trpt=t
            tlim = np.array([-1,2])+trpt*1e3
            return tlim
            
        
def plot_event(fig3, ax3,bname,title,scale=5e-8,    stations=np.arange(0.1,3,0.2),tlim=[0,5],wdir='/Users/joshmcneely/sources/ppscripts/data/'):
    group ='interface'

    cmap = plt.get_cmap('cividis')
    alpha_min = 0.3

    ###
    # raw eps xy

    tlimauto=plot_stations(bname,group,FieldId('cohesion'),scale,stations,ax3,wdir=wdir)

    if tlim=='auto':
        tlim=tlimauto
            
    ###
    # slip rate
        
    my_cmap = cmap(np.arange(cmap.N))
    # fake alpha
    my_cmap[:, :-1] = alpha_min+my_cmap[:, :-1]*(1.0 - alpha_min)
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    f3=plot_xt(bname,group,FieldId('top_velo'),ax3,my_cmap,tlim=tlim,wdir=wdir)

    cb = fig3.colorbar(f3, ax=ax3, pad=1e-2)

    cb.set_label('$\dot{\delta}$ (m/s)', rotation=270, labelpad=9)
    cb.ax.tick_params(axis='y', which='major', pad=0, rotation=270)

    ###
    # wave speeds
    cs=2700.0
    cp=4340.0 #m/s
    for c,l in zip([cs,cp],['c_s','c_p']):
        t=tlim[0]+np.array([0,1]) #mus
        x=np.array([0,-1])*c/1000.0+1.5
        plt.plot(t,x,c='w')

    ax3.set_xlim(tlim)
    ax3.invert_yaxis()
    ax3.set_xlabel(r'$t$ (ms)')
    ax3.set_ylabel(r'$x$ (m) $ \quad $ ')
    ax3.tick_params(axis='y', which='major', pad=0, rotation=90)
    ax3.set_ylim(1e-3,3)
    ax3.invert_yaxis()
    ax3.set_title(r'\texttt{{{}}}'.format(title))
    

params = {
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': 7,
    'lines.linewidth': 0.5,
    'text.usetex': False
}  
plt.rcParams.update(params)

#####################################3
# do a figure for each simulation
scl=5e-8*3

# Edit these lists with your simulation names
snames = [
    'rs_slip_asp_XL_v1_0_480',
]
tlims = ['auto']  # Use 'auto' to auto-detect, or specify [tmin, tmax] in ms

for sname,tlim in zip(snames,tlims):
    print(f"Plotting {sname}...")
    # initialize figure
    fig1,ax = plt.subplots(1,figsize=[2.5,2])
    # plot
    plot_event(fig1, ax, sname, sname.replace('_','-'),scale=scl,tlim=tlim)

    # save
    outname = '{}_xt_stations.pdf'.format(sname)
    fig1.savefig(outname, dpi=1200, transparent=True)
    print(f"Saved to {outname}")

