#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt

import ifasha.datamanager as idm
import ifasha.rupturehunter as irh

import ppscripts.postprocess as pp


def track_Xc(sname, **kwargs):
    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)

    group='interface'
    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)

    if data.has_field(idm.FieldId('friction_traction',0)):
        trac_fldid = idm.FieldId('friction_traction',0)
    else:
        trac_fldid = idm.FieldId('cohesion',0)

    if data.has_field(idm.FieldId('position',0)):
        pos_fldid = idm.FieldId('position',0)
    else:
        pos_fldid = idm.FieldId('coord',0)

    # get input data of simulation
    input_data = pp.get_input_data(sname,**kwargs)

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False
    if kwargs.get('no_fig'): 
        new_figure = False
        no_fig=True
    else: no_fig=False
    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(211)  
    
    # ---------------------------------------------------------------------
    
    if data.has_field(idm.FieldId('is_sticking',0)):
        is_sticking = data.get_full_field(idm.FieldId('is_sticking',0))       
    elif data.has_field(idm.FieldId('top_disp',0)):
        # convert displacement to is_sticking
        top_disp = data.get_full_field(idm.FieldId('top_disp',0))
        zero_precision = 1e-6*np.max(top_disp)
        is_sticking = top_disp < zero_precision
        is_sticking.astype(np.int)
        
    position          = data.get_full_field(pos_fldid) 
    friction_traction = data.get_full_field(trac_fldid)
    time              = data.get_full_field(idm.FieldId('time'))
    print(time.shape)
    print(position.shape)
    print(friction_traction.shape)
    # hunt cohesive zone
    #is_coh_zone = hunt_cohesive_zone(position,time,friction_traction,is_sticking)
       
    Xc_c_eqmot,Xc_r_eqmot = irh.hunt_cohesive_zone_boundary(position,time,friction_traction,is_sticking)
    if no_fig:
        return Xc_c_eqmot,Xc_r_eqmot
        
    ax.plot(Xc_c_eqmot[0],Xc_c_eqmot[1],label='leading')
    ax.plot(Xc_r_eqmot[0],Xc_r_eqmot[1],label='trailing')
    ax2=fig.add_subplot(212)
    ax2.plot(Xc_c_eqmot[0],(Xc_c_eqmot[0]-Xc_r_eqmot[0])/np.diff(position[0])[0],label='width (dx)')
    ax2.legend()
    ax.legend()
    if new_figure:
        ax.set_title(sname)

    return fig


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.exit('Missing argument! usage: ./track_Xc.py sname/sim-id')

    sname = str(sys.argv[1])
    
    track_Xc(sname)

    plt.show()

