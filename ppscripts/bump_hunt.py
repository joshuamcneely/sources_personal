#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d
from matplotlib.colors import LogNorm

import ifasha.datamanager as idm
from ifasha.rupturehunter import BumpHunter

import ppscripts.postprocess as pp


def bump_hunt(sname,  **kwargs):
    group = 'interface'
    fldid = idm.FieldId('cohesion',0)

    sname = pp.sname_to_sname(sname)
    wdir = kwargs.get('wdir','data')

    # number of points along space
    start_fct = kwargs.get('start_fct',0.)
    end_fct = kwargs.get('end_fct',1.)

    start_time = kwargs.get('start_time',0)
    end_time = kwargs.get('end_time',None) # None if until end of sim
    
    zmax = kwargs.get('zmax',None)
    zmin = kwargs.get('zmin',None)

    # get input data of simulation
    input_data = pp.get_input_data(sname,**kwargs)
    
    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False
    no_fig = kwargs.get('no_fig',False)
    if no_fig: new_figure = False 
    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)  
    
    # ---------------------------------------------------------------------
    # load the simulation data
    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)


    z_coord_idx = kwargs.get('z_coord_idx',1)

    try:
        data.get_full_field(idm.FieldId('coord',1))
    except:
        is_3d_in_2d = False
    else:
        is_3d_in_2d = True
        z_coord_tmp = kwargs.get('z_coord',0.0)
        z_coords = np.array(sorted(set(data.get_field_at_t_index(idm.FieldId('coord',z_coord_idx),0)[0])))
        z_coord = z_coords[np.argmin(np.abs(z_coords-z_coord_tmp))]
        print('is 3d in 2d, z_coord_tmp = ',z_coord_tmp,'z_coord = ',z_coord)

    # ---------------------------------------------------------------------
    # position
    pos_fldid = idm.FieldId('coord',0)

    
    if is_3d_in_2d:
        return RuntimeError()
    else:
        time        = data.get_full_field(idm.FieldId('time'))
        if data.has_field(idm.FieldId('is_sticking',0)):
            is_sticking = data.get_full_field(idm.FieldId('is_sticking',0))       
        elif data.has_field(idm.FieldId('top_disp',0)):
            # convert displacement to is_sticking
            zero_precision=1e-12
            top_disp = data.get_full_field(idm.FieldId('top_disp',0))
            is_sticking = top_disp < zero_precision
            is_sticking.astype(np.int)
            
        position    = data.get_full_field(pos_fldid) 

        friction_traction = data.get_full_field(idm.FieldId('cohesion',0))

        bump_hunter = BumpHunter()
        bump_hunter.load(is_sticking, friction_traction, position, time)
        bump_hunter.hunt()
        bump = bump_hunter.get_bump() #[tip_pos,bump_pos,bump_time,bump_size]

        print(bump.shape)
    if no_fig:
        return bump
    else:
        ax.scatter(bump[1],bump[3],label=sname)

        if new_figure:
            ax.set_title(sname+" "+fldid.get_string())

        return fig


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) not in  [2,3,4]:
        sys.exit('Missing argument! usage: ./bump_hunt.py sname/sim-id')

    sname = str(sys.argv[1])

    try: 
        z_coord_idx=int(sys.argv[2])
    except:
        z_coord_idx=1
    
    try: 
        z_coord=float(sys.argv[3])
    except:
        z_coord=0.0
        
    # use a preceptually uniform colormap
    plt.rcParams['image.cmap'] = 'viridis'#, 'magma', 'plasma', 'inferno', 'cividis'

    print(z_coord_idx,z_coord)
    bump_hunt(sname,z_coord_idx=z_coord_idx,z_coord=z_coord,start_fct=0.5)

    plt.show()
