#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d

import ifasha.datamanager as idm

import ppscripts.postprocess as pp

def global_quantity(sname, group, fldid, **kwargs):

    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)

    # check if fldids are FieldId nor string, else convert
    fldid = idm.FieldId.string_to_fieldid(fldid)

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

    T = data.get_full_field(idm.FieldId('time'))
    V = data.get_full_field(fldid)

    fg1 = ax.plot(T,V)

    if new_figure:
        ax.set_title(fldid.get_string())

    return fg1


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 4:
        sys.exit('Missing argument! usage: ./global_quantity.py sname/sim-id group fldid')

    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])

    fid = idm.FieldId()
    if fldid[-2] != '_':
        fldid += '_0'
    fid.load_string(fldid)

    global_quantity(sname,group,fid)

    plt.show()
