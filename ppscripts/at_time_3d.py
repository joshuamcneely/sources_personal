#!/usr/bin/env python
# at_time_3d.py
#
# plot FieldId for a given time
#
# There is no warranty for this code
#
# @version 0.1
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     2018/09/12
# @modified 2018/09/12
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d

import ifasha.datamanager as idm

import ppscripts.postprocess as pp

def get_at_time_3d(sname, group, fldid, **kwargs):

    wdir = kwargs.get('wdir','./data')

    # in cases snames are actually sim-ids
    sname = pp.sname_to_sname(sname)

    # norm
    x_norm = kwargs.get('x_norm', 1.)
    y_norm = kwargs.get('y_norm', 1.)

    # ---------------------------------------------------------------------

    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)

    
    # position
    if data.has_field(idm.FieldId('position',0)):
        xfldid = idm.FieldId('position',0)
        zfldid = idm.FieldId('position',2)
    elif data.has_field(idm.FieldId('coord',0)):
        xfldid = idm.FieldId('coord',0)
        if data.has_field(idm.FieldId('coord',2)):
            zfldid = idm.FieldId('coord',2)
        else:
            zfldid = idm.FieldId('coord',1)
    else:
        raise('Does not have any position information')


    if 'time' in kwargs.keys():
        time = kwargs.get('time')
        tidx = data.get_index_of_closest_time(idm.FieldId('time'),time)
    elif 'time_idx' in kwargs.keys():
        tidx = kwargs.get('time_idx')
    else:
        raise RuntimeError("missing keyword argument: time or time_idx")

    X,Z,V = data.get_xy_plot(xfldid,zfldid,fldid,tidx)
    
    x_start_fct = kwargs.get('x_start_fct',0.)
    x_end_fct = kwargs.get('x_end_fct',1.)
    nb_x_elements = kwargs.get('nb_x_elements',600)
    
    z_start_fct = kwargs.get('z_start_fct',0.)
    z_end_fct = kwargs.get('z_end_fct',1.)
    nb_z_elements = kwargs.get('nb_z_elements',600)
    
    nbx,nbz=X.shape
    st_x = int(nbx*x_start_fct);
    e_x = int(nbx*x_end_fct);
    dx = int(max(nbx/nb_x_elements,1))
    xslice = np.arange(st_x,e_x,dx)
    
    st_z = int(nbz*z_start_fct);
    e_z = int(nbz*z_end_fct);
    dz = int(max(nbz/nb_z_elements,1))
    zslice = np.arange(st_z,e_z,dz)
    
    X,Z,V =  data.get_sliced_x_sliced_y_plot(xfldid, xslice, zfldid, zslice, fldid, tidx)

    print(X.shape,Z.shape,V.shape)
    return X,Z,V

def plot_at_time_3d(sname, group, fid, **kwargs):
    fldid = idm.FieldId()
    fldid.load_string(fid)

    X,Z,V = get_at_time_3d(sname,group,fldid,**kwargs)

    zmax = kwargs.get('zmax',None)
    zmin = kwargs.get('zmin',None)

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)   

    #X,Z = idm.FieldCollectionAnalysis.make_pretty(X,Z)

    print(X.shape,Z.shape,V.shape)

    print('min',np.min(V))
    print('max',np.max(V))

    if zmin is not None and zmax is not None:
        fg1 = ax.pcolor(X,Z,V,vmin=zmin,vmax=zmax)
    elif zmin is not None:
        fg1 = ax.pcolor(X,Z,V,vmin=zmin)
    elif zmax is not None:
        fg1 = ax.pcolor(X,Z,V,vmax=zmax)
    else:
        fg1 = ax.pcolor(X,Z,V)


    if 'time' in kwargs.keys():
        time = kwargs.get('time')
        
        wdir = kwargs.get('wdir','./data')

        # in cases snames are actually sim-ids
        sname = pp.sname_to_sname(sname)
        dma = idm.DataManagerAnalysis(sname,wdir)
        data = dma(group)
        tidx = data.get_index_of_closest_time(idm.FieldId('time'),time)
    else:
        tidx = kwargs.get('time_idx')
        
    ax.set_title("{} {} -- {}".format(fldid.name.replace('_',' '),fldid._i,tidx))

    cbar=plt.colorbar(fg1,ax=ax)
    cbar.set_label(fldid.get_string())
    fg1.axes.set_aspect(1)

    ax.legend(loc='best')
    return cbar

    
# --------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 5:
        sys.exit('Missing argument! usage: ./at_time_3d.py sname/sim-id group fldid time_idx')

    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])
    time = float(sys.argv[4])
    #tidx=int(sys.argv[4])
    

    # use a preceptually uniform colormap
    plt.rcParams['image.cmap'] = 'viridis'#, 'magma', 'plasma', 'inferno', 'cividis'
    
    plot_at_time_3d(sname, group, fldid, 
                    time=time)
                    #time_idx=tidx)
    
    plt.show()
    

