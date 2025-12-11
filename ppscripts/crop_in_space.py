#!/usr/bin/env python

# crop_in_space.py
#
# crop all fields in all groups. This does change the stored data!!
#
# There is no warranty for this code
#
# @version 0.1
# @author David Kammer <dkammer@ethz.ch>
# @date     2019/10/18
# @modified 2019/10/18
from __future__ import print_function, division, absolute_import

import sys

import ifasha.datamanager as idm

import ppscripts.postprocess as pp

# apply to all fields in group
def crop(sname, group, fltr, **kwargs):

    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)
    
    dm = idm.DataManager(sname,wdir)
    fc = dm(group)
    fc.filter_fields(fltr)

    
def get_filter(sname, group, fldid, **kwargs):

    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)

    # check if fldids are FieldId nor string, else convert
    fldid = idm.FieldId.string_to_fieldid(fldid)

    dm = idm.DataManagerAnalysis(sname,wdir)
    data = dm(group)
    field = data.get_field_at_t_index(fldid,0)[0]

    boundary_type = kwargs.get('boundary_type')
    boundary_value = kwargs.get('boundary_value')

    if boundary_type == 'max':
        fltr = field <= boundary_value
    elif boundary_type == 'min':
        fltr = field >= boundary_value
    else:
        raise RuntimeError('Do not know boundary type: {}'.format(boundary_type))

    return fltr
    
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    error_message = """
    Usage: ./crop_in_space.py simid[s] group field-for-filter type value
      type (boundary provided), options: max , min
      value (float for boundary)
      Note: this will crop all fields in the group based on filter
    """
    
    if len(sys.argv) != 6:
        sys.exit(error_message)

    simid = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])
    btype = str(sys.argv[4])
    bvalue = float(sys.argv[5])

    # simid can be 22,24,30-33 -> [22,24,30,31,32,33]
    simids = pp.string_to_simidlist(simid)

    # execute the filtering
    for simid in simids:
        print('filter simid={}'.format(simid))
        fltr = get_filter(simid,group,fldid,
                          boundary_type=btype,boundary_value=bvalue)
        crop(simid,group,fltr)
