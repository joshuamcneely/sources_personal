#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d
from matplotlib.colors import LogNorm

import ifasha.datamanager as idm

import ppscripts.postprocess as pp

def x_y_at_time(sname, group, fldid, **kwargs):

    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)

    tidx = kwargs.get('tidx',None)
    
    # number of points along space
    nb_x_elements = kwargs.get('nb_x_elements',100)
    start_x_fct = kwargs.get('start_x_fct',0.)
    end_x_fct = kwargs.get('end_x_fct',1.)
    nb_x = kwargs.get('nb_x',800)

    nb_y_elements = kwargs.get('nb_y_elements',100)
    start_y_fct = kwargs.get('start_y_fct',0.)
    end_y_fct = kwargs.get('end_y_fct',1.)
    nb_y = kwargs.get('nb_y',400)

    zmax = kwargs.get('zmax',None)
    zmin = kwargs.get('zmin',None)

    # check if fldids are FieldId nor string, else convert
    fldid = idm.FieldId.string_to_fieldid(fldid)

    # get input data of simulation
    input_data = pp.get_input_data(sname,**kwargs)

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False
    if kwargs.get('no_fig'): new_figure = False 
    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)  
    
    # ---------------------------------------------------------------------
    # load the simulation data
    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)


    is_3d_in_2d = False    
    if 'spatial_dimension' in input_data.keys():  
        if input_data['spatial_dimension']==3:
            raise RuntimeError("ERROR: Not implemented for 3D ")

    nb_x_elements = data.get_nb_points_in_direction(idm.FieldId('position',1))
    nb_y_elements = data.get_nb_points_in_direction(idm.FieldId('position',0))
    st_x_elements = int(start_x_fct * nb_x_elements)
    st_y_elements = int(start_y_fct * nb_y_elements)
    e_x_elements = int(end_x_fct * nb_x_elements)
    e_y_elements = int(end_y_fct * nb_y_elements)
    
    int_x_elements = max(1,int((e_x_elements - st_x_elements) / min(nb_x_elements,nb_x)))
    int_y_elements = max(1,int((e_y_elements - st_y_elements) / min(nb_y_elements,nb_y)))
    print('xs',st_x_elements,'xe',e_x_elements,'delta_x',int_x_elements)
    print('ys',st_y_elements,'ye',e_y_elements,'delta_y',int_y_elements)


    time = data.get_full_field(idm.FieldId('time'))
    
    if tidx == None:
        print(time)
        raise RuntimeError("choose time index tidx")
    else:
        print('time =',time[tidx])
    #---------------------------------------------------------------------------------------
    
    
    X,Y,Z = data.get_sliced_x_sliced_y_plot(
        idm.FieldId('position',0),
        np.arange(st_x_elements,e_x_elements,int_x_elements),
        idm.FieldId('position',1),
        np.arange(st_y_elements,e_y_elements,int_y_elements),
        fldid,
        tidx)


    print(X.shape,Y.shape,Z.shape)
    Xp,Yp = data.make_pretty(X,Y)
    print(Xp.shape,Yp.shape,Z.shape)
    print('max',np.max(Z))
    print('min',np.min(Z))

    if kwargs.get('no_fig'):
        return X,Y,Z

    if zmin is not None and zmax is not None:
        fg1 = ax.pcolor(X,Y,Z,vmin=zmin,vmax=zmax)
    elif zmin is not None:
        fg1 = ax.pcolor(X,Y,Z,vmin=zmin)
    elif zmax is not None:
        fg1 = ax.pcolor(X,Y,Z,vmax=zmax)
    else:
        fg1 = ax.pcolor(X,Y,Z)

    if new_figure:
        ax.set_title(fldid.get_string())

        plt.colorbar(fg1,orientation='horizontal')

    return fg1
 
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) not in  [4,5]:
        raise RuntimeError('Missing argument! usage: ./x_y_at_time.py sname/sim-id group fldid tidx')

    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])
    try:
        tidx  = int(sys.argv[4])
    except:
        tidx=None

    # use a preceptually uniform colormap
    plt.rcParams['image.cmap'] = 'viridis'#, 'magma', 'plasma', 'inferno', 'cividis'

    fg = x_y_at_time(sname,group,fldid,tidx=tidx)

    fg.axes.set_aspect(1)

    plt.show()

   #-------------------------------------------------------------------------------

