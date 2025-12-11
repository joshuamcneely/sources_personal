#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt

import ifasha.datamanager as idm
from ifasha.rupturehunter import hunt_cohesive_zone

import ppscripts.postprocess as pp
from ppscripts.space_time import get_space_time


def track_max_field(sname, group, fldid, **kwargs):
    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)

    fldid = idm.FieldId.string_to_fieldid(fldid)

    # get input data of simulation
    input_data = pp.get_input_data(sname,**kwargs)

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)

    no_fig= kwargs.get('no_fig',False)
    new_figure = True if ax is None and no_fig is False else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)  
            
    # ---------------------------------------------------------------------
    # get space time data
    kwargs['no_fig']=True
    X,T,Z = get_space_time(bname,group,fldid,nb_t_points=np.inf,nb_x_elements=np.inf,**kwargs)

    times = T[:,0]
    position = X[0]
    sign = float(kwargs.get('sign',1))
    max_field = np.array([np.max(z*sign) for z in Z])*sign

    max_field_t_z = np.array([times,max_field])

    if no_fig:
        return max_field_t_z

    ax.plot(max_field_t_z[0], max_field_t_z[1],label='max_field')

    ax.legend()
    if new_figure:
        ax.set_title(fldid.get_string())

    return fig



# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) not in [4,5]:
        sys.exit('Missing argument! usage: ./supershear_arrest.py sname/sim-id group fldid')


    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])
    
    fid = idm.FieldId()
    fid.load_string(fldid)

    try: sign=float(sys.argv[4])
    except:
        sign=1.0
    track_max_field(sname,group,fid,sign=sign)

    plt.show()

