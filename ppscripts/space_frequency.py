#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d
from matplotlib.colors import LogNorm

import ifasha.datamanager as ifm

import ppscripts.postprocess as pp

def Blackman_Harris_w(x):
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    N = len(x)
    n = np.arange(N)
    w = a0 - a1*np.cos(2*np.pi*n/N) + a2*np.cos(4*np.pi*n/N) - a3*np.cos(6*np.pi*n/N)
    return w*x

def const_time_step(X,T,Z,intp='none'):
    t=T[:,0]
    dt = min(np.diff(t))
    if intp in['none','crop']:
        idxmin = np.argmin(np.diff(np.diff(t)))
        T=T[idxmin:]
        X=X[idxmin:]
        Z=Z[idxmin:]
    else:
        t_cst=np.arange(min(t),max(t),dt)
        if intp=='linear':
            
            raise RuntimeError('Not implemented')
        
    return X,T,Z,dt 

def to_frequency_domain(X,T,Z):
    X,T,Z,dt = const_time_step(X,T,Z,intp='none')
    # to frequency domain    
    Z=Z-np.average(Z)
    w=Blackman_Harris_w(np.ones(Z.shape[0]))
    Zfft = np.array([np.fft.fft(z*w) for z in Z.T]).T
    n = T.shape[0]
    fs = 1/dt #sampling frequency
    print('sampling period',dt,'frequency',fs)
    S = np.array([[i*fs/n for i in range(T.shape[0])] for j in range(T.shape[1]) ]).T
    print('f0,f1,fs',S[0,0],S[1,0],S[-1,0])

    # discard redundant data
    idxmax=n//2
    Zfft=Zfft[:idxmax]
    S=S[:idxmax]    
    X=X[:idxmax]
    Zamp = np.sqrt(Zfft.real**2 + Zfft.imag**2 )
    Zphs = np.arctan(Zfft.imag/Zfft.real)
    maxZ = np.max(Zamp)
    Zphs = np.array([[Zphs[i,j] if Zamp[i,j]>maxZ*1e-8 else 0 for i in range(Zamp.shape[0])] for j in range(Zamp.shape[1])]).T

    return X,S,Zamp,Zphs

def space_frequency(sname, group, fldid, **kwargs):

    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)

    # number of points along space
    nb_x_elements = kwargs.get('nb_x_elements',100)
    start_fct = kwargs.get('start_fct',0.)
    end_fct = kwargs.get('end_fct',1.)

    start_time = kwargs.get('start_time',0)
    end_time = kwargs.get('end_time',None) # None if until end of sim
    nb_t_points = kwargs.get('nb_t_points',0)

    zmax = kwargs.get('zmax',None)
    zmin = kwargs.get('zmin',None)

    # get input data of simulation
    input_data = pp.get_input_data(sname,**kwargs)

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False
    if kwargs.get('no_fig'): new_figure = False 
    if new_figure:
        fig = plt.figure()
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)  
        ax2 = fig.add_subplot(133)  
    
    # ---------------------------------------------------------------------
    # load the simulation data
    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)

    is_3d_in_2d = False    
    if 'spatial_dimension' in input_data.keys():  
        if input_data['spatial_dimension']==3:
            z_coord_tmp = kwargs.get('zcoord',0.0)
            z_coords = np.array(sorted(set(data.get_field_at_t_index(idm.FieldId('position',2),0)[0])))
            z_coord = z_coords[np.argmin(np.abs(z_coords-z_coord_tmp))]
            is_3d_in_2d = True
            print('is 3d in 2d')

    # ---------------------------------------------------------------------
    start = data.get_index_of_closest_time(idm.FieldId('time'),start_time)
    if end_time is None:
        last = data.get_t_index('last')
    else:
        last = data.get_index_of_closest_time(idm.FieldId('time'),end_time)

    delta_t = max(1,(last - start) / nb_t_points)
    print('start',start,'last',last,'delta_t',delta_t)

    # position
    if data.has_field(idm.FieldId('position',0)):
        pos_fldid = idm.FieldId('position',0)
    elif data.has_field(idm.FieldId('coord',0)):
        pos_fldid = idm.FieldId('coord',0)
    else:
        raise('Does not have any position information')

    if is_3d_in_2d:
        print('z_coord',z_coord)
        idx = data.get_index_of_closest_position(idm.FieldId('position',2),z_coord)
        zpos = data.get_field_at_node_index(idm.FieldId('position',2),idx)[0][0]
        idcs = data.get_indices_of_nodes_on_line(idm.FieldId('position',2),zpos)
        nb_elements = len(idcs)
    else:
        nb_elements = len(data.get_field_at_t_index(pos_fldid,0)[0])
    st_x_elements = int(start_fct * nb_elements)
    e_x_elements = int(end_fct * nb_elements)
    int_x_elements = max(1,int((e_x_elements - st_x_elements) / nb_x_elements))
    print('xs',st_x_elements,'xe',e_x_elements,'delta_x',int_x_elements)

    if is_3d_in_2d:
        X,T,Z = data.get_sliced_x_sliced_t_plot_at_node_index(
            pos_fldid, 
            np.arange(st_x_elements,e_x_elements,int_x_elements),
            idm.FieldId("time"),
            np.arange(start,last,delta_t),
            fldid,
            idcs)
    else:
        X,T,Z = data.get_sliced_x_sliced_t_plot(
            pos_fldid,
            np.arange(st_x_elements,e_x_elements,int_x_elements),
            idm.FieldId("time"),
            np.arange(start,last,delta_t),
            fldid)

    print(X.shape,T.shape,Z.shape)
    print('maxZ',np.max(Z))
    print('minZ',np.min(Z))

    fg0 = ax0.pcolor(*const_time_step(X,T,Z,intp='none')[:3])

    X,S,Zamp,Zphs = to_frequency_domain(X,T,Z)
    
    print(X.shape,S.shape,Zamp.shape)
    print('maxZamp',np.max(Zamp))
    print('minZamp',np.min(Zamp))
    print('maxX',np.max(X))
    print('minX',np.min(X))
    print('maxS',np.max(S))
    print('minS',np.min(S))

    if kwargs.get('no_fig'):
        return X,S,Zamp,Zphs

    if zmin is not None and zmax is not None:
        fg1 = ax1.pcolor(X,S,Zamp,vmin=zmin,vmax=zmax)
    elif zmin is not None:
        fg1 = ax1.pcolor(X,S,Zamp,vmin=zmin)
    elif zmax is not None:
        fg1 = ax1.pcolor(X,S,Zamp,vmax=zmax)
    else:
        fg1 = ax1.pcolor(X,S,Zamp,norm=LogNorm())
    fg2 = ax2.pcolor(X,S,Zphs)
    
    #Z = np.gradient(Z)
    #Z = np.abs(Z)
    #Z = Z[1]
    #fg1 = ax.pcolor(X,T,Z, norm=LogNorm(vmin=1, vmax=Z.max()))
    if new_figure:
        for ax in [ax1,ax2]:
            ax.set_yscale('log')
            ax.set_ylim(S[1,0],S[-1,0])
        ax0.set_title(fldid.get_string())
        ax1.set_title(fldid.get_string()+' amplitude')
        ax2.set_title(fldid.get_string()+' phase')

        fig.colorbar(fg0,ax=ax0,orientation='horizontal')        
        fig.colorbar(fg1,ax=ax1,orientation='horizontal')
        fig.colorbar(fg2,ax=ax2,orientation='horizontal')

    fig.tight_layout()
    return fg1,fg2


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) not in  [4,5]:
        sys.exit('Missing argument! usage: ./supershear_arrest.py sname/sim-id group fldid')

    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])
    try: 
        z_coord=float(sys.argv[4])
    except:
        z_coord=0.0

    fid = idm.FieldId()
    fid.load_string(fldid)

    space_frequency(sname,group,fid,zcoord=z_coord)

    plt.show()
