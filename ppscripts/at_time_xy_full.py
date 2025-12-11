#!/usr/bin/env python
#
# at_time.py
#
# plot FieldId for a given time
#
# There is no warranty for this code
#
# @version 0.1
# @author David Kammer <dkammer@ethz.ch>
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     2022/12/12
# @modified 2022/12/12
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt

import ifasha.datamanager as idm

import ppscripts.postprocess as pp

def get_at_time_xy_full(sname, group, fldid, **kwargs):

    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)
        
    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)

    if 'time' in kwargs.keys():
        time = kwargs.get('time')
        tidx = data.get_index_of_closest_time(idm.FieldId('time'),time)
    elif 'time_idx' in kwargs.keys():
        tidx = kwargs.get('time_idx')
    else:
        raise RuntimeError("missing keyword argument: time or time_idx")
    
    z_coord = kwargs.get('zcoord',0.0)
    z_fld   =  kwargs.get('zfld',2)
    
    if data.has_field(idm.FieldId('position',2))==False:
        raise("only for 3d")
    if group!='full':
        raise("only for full fieldcollection")
    
    # position
    if z_fld == 2:
        pos_x_fldid = idm.FieldId('position',0)
        pos_y_fldid = idm.FieldId('position',1)
        pos_z_fldid = idm.FieldId('position',2)
    elif z_fld == 1:
        pos_x_fldid = idm.FieldId('position',0)
        pos_y_fldid = idm.FieldId('position',2)
        pos_z_fldid = idm.FieldId('position',1)
    elif z_fld == 0:
        pos_x_fldid = idm.FieldId('position',2)
        pos_y_fldid = idm.FieldId('position',1)
        pos_z_fldid = idm.FieldId('position',0)

    
    # position
    tol=kwargs.get('tolerance',0.0)
        
    print('getting indices ...')
    idcs = data.get_indices_of_nodes_on_line(pos_z_fldid,z_coord,tolerance=tol)
    print(np.array(idcs).shape)
    print('getting data ...')
    X,Y,V = data.get_xy_plot_at_node_index(pos_x_fldid, 
                                           pos_y_fldid, 
                                           fldid, tidx, idcs)

    return X,Y,V

# ------------------------------------------------------------------------------
def plot_at_time_xy_full(snames, groups, fldids, times, **kwargs):

    # check if fldids are FieldId nor string, else convert
    fldids = idm.FieldId.string_to_fieldid(fldids)

    
    with_legend = kwargs.get('with_legend',True)

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)  

    for (sname,group) in zip(snames,groups):

        for (fldid,time) in zip(fldids,times):
            
            X,Y,F = get_at_time_xy_full(sname, group, fldid,time=time,**kwargs)
            print(np.min(F), np.max(F))
                  
            fg1 = ax.pcolormesh(X,Y,F)
    if new_figure:
        z_fld   =  kwargs.get('zfld',2)
        if z_fld==2:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        elif z_fld==1:
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        else:
            ax.set_xlabel('z')
            ax.set_ylabel('y')
    if new_figure or with_legend:
        ax.legend(loc='best')
    
    return fg1


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 5:
        sys.exit('Missing argument! usage: ./at_time.py sname/sim-id group fldid time')


    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])
    time = float(sys.argv[4])

    plot_at_time_xy_full([sname],[group],[fldid],[time],tolerance=1e-5, zfld=2)
    plot_at_time_xy_full([sname],[group],[fldid],[time],tolerance=1e-5, zfld=1)
    plt.show()
