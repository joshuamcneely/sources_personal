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
# @date     2017/05/25
# @modified 2019/07/10
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt

import ifasha.datamanager as idm

import ppscripts.postprocess as pp

def get_at_time(sname, group, fldid, **kwargs):

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

    dim = 2
    if data.has_field(idm.FieldId('position',2)) or data.has_field(idm.FieldId('coord',2)): 
        dim = 3

    # position
    if data.has_field(idm.FieldId('position',0)):
        pos_x_fldid = idm.FieldId('position',0)
        pos_z_fldid = idm.FieldId('position',2)
    elif data.has_field(idm.FieldId('coord',0)):
        pos_x_fldid = idm.FieldId('coord',0)
        pos_z_fldid = idm.FieldId('coord',2)
    else:
        raise('Does not have any position information')

    if dim==3:
        # position
        tol=kwargs.get('tolerance',0)
        if tol!=0:
            raise RuntimeError("Need to implement tolerance in get_indices_of_nodes_on_line(pos_z_fldid,zcoord,TOL)")
        print('getting indices ...')
        idcs = data.get_indices_of_nodes_on_line(pos_z_fldid,z_coord)
        print('getting position ...')
        pos_x  = data.get_field_at_node_index(pos_x_fldid,idcs)
        if pos_x.shape[0]>1:
            pos_x  = pos_x[tidx,:]
        else:
            pos_x = pos_x[0,:]
            print('getting field...')
        field_x = data.get_field_at_node_index(fldid,idcs)[tidx,:]
    else:
        # position
        print('getting pos...')
        pos_x = data.get_field_at_t_index(pos_x_fldid,0)[0]
        print('getting field...')
        field_x = data.get_field_at_t_index(fldid, tidx)[0]

    # sort
    fltr = pos_x.argsort()
    position = pos_x[fltr]
    field = field_x[fltr]

    return position,field

# ------------------------------------------------------------------------------
def plot_at_time(snames, groups, fldids, times, **kwargs):

    # check if fldids are FieldId nor string, else convert
    fldids = idm.FieldId.string_to_fieldid(fldids)

    # crop
    start_fct = kwargs.get('start_fct',0.)
    end_fct = kwargs.get('end_fct',1.)

    # norm
    x_norm = kwargs.get('x_norm', 1.)
    y_norm = kwargs.get('y_norm', 1.)

    with_legend = kwargs.get('with_legend',True)

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)  

    for (sname,group) in zip(snames,groups):

        for (fldid,time) in zip(fldids,times):
            
            position,field = get_at_time(sname,group,fldid,time=time,**kwargs)
            nb_elements = len(field)
            st_x_elements = int(start_fct * nb_elements)
            e_x_elements = int(end_fct * nb_elements)
            field = field[st_x_elements:e_x_elements]
            position = position[st_x_elements:e_x_elements]

            print('fld min max',np.min(field),np.max(field))
            plot_kwargs = kwargs.get('plot_kwargs',
                                     {'label':'{} - t={:1.2e}'.format(fldid.get_string(),time)})

            fg1 = ax.plot(position*x_norm,
                          field*y_norm,
                          **plot_kwargs)
    if new_figure:
        ax.set_xlabel('x')

    if new_figure or with_legend:
        ax.legend(loc='best')
    
    return fg1

# ------------------------------------------------------------------------------
# plot difference for two fields, provide full information for both fields
def plot_diff_at_time(snames, groups, fldids, times, **kwargs):

    if len(snames)!=2 or len(groups)!=2 or len(fldids)!=2 or len(times)!=2:
        print(len(snames),len(groups),len(fldids),len(times))
        raise RuntimeError('not enough information')

    # check if fldids are FieldId nor string, else convert
    fldids = idm.FieldId.string_to_fieldid(fldids)

    # norm
    x_norm = kwargs.get('x_norm', 1.)
    y_norm = kwargs.get('y_norm', 1.)

    with_legend = kwargs.get('with_legend',True)

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)  

    pos_0,fld_0 = get_at_time(snames[0],groups[0],
                              fldids[0],time=times[0],
                              **kwargs)
    pos_1,fld_1 = get_at_time(snames[1],groups[1],
                              fldids[1],time=times[1],
                              **kwargs)

    plot_kwargs = kwargs.get('plot_kwargs',
                             {'label':'t={:1.2e}'.format(times[0])})

    fg1 = ax.plot(pos_0*x_norm,
                  (fld_0-fld_1)*y_norm,
                  **plot_kwargs)
    
    ax.set_title(fldid.get_string().replace('_',' '))
    if kwargs.get('legend',True):
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

    plot_at_time([sname],[group],[fldid],[time])

    plt.show()
