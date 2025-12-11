#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d

import ifasha.datamanager as idm

import ppscripts.postprocess as pp

def space_space(sname, group, fldid, time, **kwargs):

    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)

    # number of points along x
    nb_x_elements = kwargs.get('nb_x_elements',100)

    # number of points along y
    nb_y_elements = kwargs.get('nb_y_elements',100)

    # check if fldids are FieldId nor string, else convert
    fldid = idm.FieldId.string_to_fieldid(fldid)

    # norm
    x_norm = kwargs.get('x_norm', 1.)
    y_norm = kwargs.get('y_norm', 1.)

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)  

    # ---------------------------------------------------------------------
    # load the simulation data
    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)

    # x position
    if data.has_field(idm.FieldId('position',0)):
        x_pos_fldid = idm.FieldId('position',0)
    elif data.has_field(idm.FieldId('coord',0)):
        x_pos_fldid = idm.FieldId('coord',0)
    else:
        raise('Does not have any position information')

    # y position
    if data.has_field(idm.FieldId('position',1)):
        y_pos_fldid = idm.FieldId('position',1)
    elif data.has_field(idm.FieldId('coord',1)):
        y_pos_fldid = idm.FieldId('coord',1)
    else:
        raise('Does not have any position information')

    # find number of points, need to do the same was as get_xy_plot function
    X0 = data.get_field_memmap(x_pos_fldid)[0,:]
    Y0 = data.get_field_memmap(y_pos_fldid)[0,:]
    p = X0.argsort()
    X0 = X0[p]
    Y0 = Y0[p]
    Xr = X0[::-1]
    Xmin = Xr[-1]
    lng = len(Xr) - np.where(Xr==Xmin)[0][0]
    X0.shape = [-1,lng]
    Y0.shape = [-1,lng]

    soi = np.argsort(Y0, axis=1)
    sti = np.indices(Y0.shape)
    X0 = X0[sti[0], soi]
    Y0 = Y0[sti[0], soi]


    nbxetot = X0.shape[0]
    print('nb nodes in x',nbxetot)
    nbyetot = X0.shape[1]
    print('nb nodes in y',nbyetot)

    st_x_el = 0
    if 'start_fct_x' in kwargs:
        start_fct_x = kwargs.get('start_fct_x')
        st_x_el = int(start_fct_x * nbxetot)
    elif 'start_x' in kwargs:
        start_x = kwargs.get('start_x')
        st_x_el = np.argmin(abs(X0[:,0] - start_x))
    st_x_el = max(0,st_x_el)

    e_x_el = nbxetot-1
    if 'end_fct_x' in kwargs:
        end_fct_x = kwargs.get('end_fct_x')
        e_x_el = int(end_fct_x * nbxetot)
    elif 'end_x' in kwargs:
        end_x = kwargs.get('end_x')
        e_x_el = np.argmin(abs(X0[:,0] - end_x))
    e_x_el = min(e_x_el,nbxetot-1)

    st_y_el = 0
    if 'start_fct_y' in kwargs:
        start_fct_y = kwargs.get('start_fct_y')
        st_y_el = int(start_fct_y * nbyetot)
    elif 'start_y' in kwargs:
        start_y = kwargs.get('start_y')
        st_y_el = np.argmin(abs(Y0[0,:] - start_y))
    st_y_el = max(0,st_y_el)

    e_y_el = nbyetot-1
    if 'end_fct_y' in kwargs:
        end_fct_y = kwargs.get('end_fct_y')
        e_y_el = int(end_fct_x * nbyetot)
    elif 'end_y' in kwargs:
        end_y = kwargs.get('end_y')
        e_y_el = np.argmin(abs(Y0[0,:] - end_y))
    e_y_el = min(e_y_el,nbyetot-1)

    int_x_el = max(1,int((e_x_el - st_x_el) / nb_x_elements))
    print('xs',st_x_el,'xe',e_x_el,'delta_x',int_x_el)
    int_y_el = max(1,int((e_y_el - st_y_el) / nb_y_elements))
    print('ys',st_y_el,'ye',e_y_el,'delta_y',int_y_el)

    tidx = data.get_index_of_closest_time(idm.FieldId('time'),time)
    actual_time = data.get_full_field(idm.FieldId('time'))[tidx]
    print('actual time = {}'.format(actual_time))

    X,Y,Z = data.get_sliced_x_sliced_y_plot(x_pos_fldid,
                                            np.arange(st_x_el,e_x_el,int_x_el),
                                            y_pos_fldid,
                                            np.arange(st_y_el,e_y_el,int_y_el),
                                            fldid,
                                            tidx)
    print(X.shape,Y.shape,Z.shape)
    print('max',np.max(Z))
    print('min',np.min(Z))

    # kwargs could include vmin or vmax
    plot_kwargs = kwargs.get('plot_kwargs',{})
    fg1 = ax.pcolor(X*x_norm,Y*y_norm,Z,**plot_kwargs)

    if new_figure:
        ax.set_title(fldid.get_string())
        ax.axis([X.min()*x_norm, X.max()*x_norm,
                 Y.min()*y_norm, Y.max()*y_norm])
        plt.colorbar(fg1)

    return fg1


# -------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 5:
        sys.exit('Missing argument! usage: ./space_space.py sname/sim-id group fldid')

    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])
    time = float(sys.argv[4])

    space_space(snamegroup,fldid,time)

    plt.show()
