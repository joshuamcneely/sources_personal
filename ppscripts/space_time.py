#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d
from matplotlib.colors import LogNorm

import ifasha.datamanager as idm

import ppscripts.postprocess as pp

def get_space_time(sname, group, fldid, **kwargs):

    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)

    # number of points along space
    nb_x_elements = kwargs.get('nb_x_elements',400)
    start_fct = kwargs.get('start_fct',0.)
    end_fct = kwargs.get('end_fct',1.)

    start_time = kwargs.get('start_time',0)
    end_time = kwargs.get('end_time',None) # None if until end of sim
    nb_t_points = kwargs.get('nb_t_points',400)

    # ---------------------------------------------------------------------
    # load the simulation data
    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)

    # position
    if data.has_field(idm.FieldId('position',0)):
        pos_fdnm = 'position'
        z_coord_idx_def=2;
    elif data.has_field(idm.FieldId('coord',0)):
        pos_fdnm = 'coord'
        z_coord_idx_def=2;
    else:
        raise('Does not have any position information')

    z_coord_idx = kwargs.get('z_coord_idx',z_coord_idx_def)        
    if z_coord_idx==0:
        x_coord_idx = z_coord_idx_def
    else:
        x_coord_idx = 0

    pos_fldid = idm.FieldId(pos_fdnm,x_coord_idx)
    
    is_3d_in_2d = False    
    if (pos_fdnm == 'position' and data.has_field(idm.FieldId(pos_fdnm,2))) \
       or (pos_fdnm == 'coord' and data.has_field(idm.FieldId(pos_fdnm,2))):
        is_3d_in_2d = True
        print('is 3d in 2d')
        tol=1e-6
        z_coord_tmp = kwargs.get('z_coord',0.0)
        z_coords = data.get_field_at_t_index(idm.FieldId(pos_fdnm,z_coord_idx),0)[0]
        z_coords = np.array(sorted(set(z_coords)))
        z_coord = z_coords[np.argmin(np.abs(z_coords-z_coord_tmp+tol))\
                           :min(len(z_coords),1+np.argmin(np.abs(z_coords-z_coord_tmp-tol)))]
        print('z_coord ',z_coord)

    # ---------------------------------------------------------------------
    start = data.get_index_of_closest_time(idm.FieldId('time'),start_time)
    if end_time is None:
        last = data.get_t_index('last')
    else:
        last = data.get_index_of_closest_time(idm.FieldId('time'),end_time)

    delta_t = int(max(1,(last - start) / nb_t_points))
    print('start',start,'last',last,'delta_t',delta_t)
    
    if is_3d_in_2d:
        print('z_coord',z_coord)
        idcs = []
        for z in z_coord:
            print('z',z,'zidx',z_coord_idx,'xidx',x_coord_idx)
            idx = data.get_index_of_closest_position(idm.FieldId(pos_fdnm,z_coord_idx),z)
            zpos = data.get_field_at_node_index(idm.FieldId(pos_fdnm,z_coord_idx),idx)[0][0]
            idcs += data.get_indices_of_nodes_on_line(idm.FieldId(pos_fdnm,z_coord_idx),zpos)
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
    print('max',np.max(Z))
    print('min',np.min(Z))

    return X,T,Z



def plot_space_time(sname, group, fldid, **kwargs):

    sname = pp.sname_to_sname(sname)

    # check if fldids are FieldId nor string, else convert
    fldid = idm.FieldId.string_to_fieldid(fldid)

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)  

    X,T,Z = get_space_time(sname, group, fldid, **kwargs)

    zmax = kwargs.get('zmax',None)
    zmin = kwargs.get('zmin',None)

    if zmin is not None and zmax is not None:
        fg1 = ax.pcolor(X,T,Z,vmin=zmin,vmax=zmax)
    elif zmin is not None:
        fg1 = ax.pcolor(X,T,Z,vmin=zmin)
    elif zmax is not None:
        fg1 = ax.pcolor(X,T,Z,vmax=zmax)
    else:
        fg1 = ax.pcolor(X,T,Z)

    #Z = np.gradient(Z)
    #Z = np.abs(Z)
    #Z = Z[1]
    #fg1 = ax.pcolor(X,T,Z, norm=LogNorm(vmin=1, vmax=Z.max()))
    if new_figure:

        cbar=plt.colorbar(fg1)
        cbar.set_label(fldid.get_string().replace('_',' '))

    return fg1


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) not in  [4,5]:
        sys.exit('Missing argument! usage: ./space_time.py '
                 + 'sname/sim-id group fldid')

    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])
    try:
        tstart = float(sys.argv[4])
    except:
        tstart  = 0.0
    z_coord = 0.0    
    #z_coord=float(sys.argv[4])
    #except:
    

    # use a preceptually uniform colormap
    plt.rcParams['image.cmap'] = 'viridis'#, 'magma', 'plasma', 'inferno', 'cividis'

    plot_space_time(sname,group,fldid,z_coord=z_coord,start_time=tstart)

    plt.show()
