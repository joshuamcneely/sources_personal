#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import sys
import numpy as np
from collections import Iterable

import ifasha.datamanager as idm

import ppscripts.postprocess as pp

# ---------------------------------------------------------------------
# at interface
flds = [
    [idm.FieldId('top_velo',0),'slipvelo'],
    #[idm.FieldId('top_disp',1),'normdisp'],
    #[idm.FieldId('cohesion',0),'frictrac'],
    #[idm.FieldId('cohesion',1),'contpres'], # not there
]

# -------------------------------------------------------------------------------------------
def extract_data_to_npz_file(snames, **kwargs):

    wdir = kwargs.get('wdir',"./data")
    base = kwargs.get('base',"defrig_nuc")

    group = kwargs.get('group','interface')

    sliced = kwargs.get('sliced',True)
    int_x_elements = kwargs.get('int_x_elements',4) # power of 2 for weak interface simulations
    nb_t_points = kwargs.get('nb_t_points',400)

    for sname in snames:
        # load the simulation data
        sname = pp.sname_to_sname(sname)
        dma = idm.DataManagerAnalysis(sname,wdir)
        data = dma(group)

        simid = data.sim_info.split('=')[1]

        # position
        if data.has_field(idm.FieldId('position',0)):
            pos_fldid = idm.FieldId('position',0)
        elif data.has_field(idm.FieldId('coord',0)):
            pos_fldid = idm.FieldId('coord',0)
        else:
            raise('Does not have any position information')

        # slice information
        # time
        start = 0
        last  = data.get_t_index('last')
        delta_t = max(1,(last - start) / nb_t_points)
        print('start',start,'last',last,'delta_t',delta_t)
        # space
        st_x_elements = 0
        nb_x_elements = data.get_field_shape(pos_fldid)[1]

        for fld in flds:

            name = base+"_"+simid+"_"+group+"_"+fld[1]

            # get content of (first) supplementary file with *.in
            ifl_cont = dma.get_supplementary_content([supp for supp in dma.supplementary if supp.endswith('.in')][0])
            with open('{}.in'.format(name),'w') as ifl:
                print(ifl_cont, file=ifl)

            # get content of (first) supplementary file with *.out
            ofl_cont = dma.get_supplementary_content([supp for supp in dma.supplementary if supp.endswith('.out')][0])
            with open('{}.out'.format(name),'w') as ofl:
                print(ofl_cont, file=ofl)

            if sliced:
                X,T,Z = data.get_sliced_x_sliced_t_plot(pos_fldid,
                                                        np.arange(st_x_elements,nb_x_elements,int_x_elements),
                                                        idm.FieldId("time"),
                                                        np.arange(start,last,delta_t),
                                                        fld[0])
            else:
                X,T,Z = data.get_xt_plot(pos_fldid,
                                         idm.FieldId("time"),
                                         fld[0])

            np.savez_compressed(name, x=X, t=T, v=Z)

# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.exit('Missing argument! usage: ./extract_data_to_npz_file.py sim-id/sname')

    argv1 = str(sys.argv[1])

    extract_data_to_npz_file([argv1])
        
