#!/usr/bin/env python

# crop.py
#
# crop all fields in all groups. This does change the stored data!!
#
# There is no warranty for this code
#
# @version 0.1
# @author David Kammer <kammer@cornell.edu>
# @date     2017/09/08
# @modified 2017/09/08
from __future__ import print_function, division, absolute_import

import sys

import ifasha.datamanager as idm

import ppscripts.postprocess as pp

# if no group provided: does apply on all field collections
def crop(sname, time, **kwargs):

    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)

    dm = idm.DataManager(sname,wdir)
    fcs = dm.get_all_field_collections()

    for fc in fcs:
        if not fc.is_packed():
            tidx = idm.FieldCollectionAnalysis(fc).get_index_of_closest_time(idm.FieldId('time'),time)
            tidx = tidx + 1 # to be conservative
            
            fc.crop_fields_at_time_step(tidx)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 3:
        sys.exit('Missing argument! usage: ./crop.py sname time \n does not work with sim-id')

    sname = str(sys.argv[1])
    time = float(sys.argv[2])

    crop(sname,time)
