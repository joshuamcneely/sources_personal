#!/usr/bin/env python

# separate_at_dist_data.py
#
# separates data along one dimension into separate fieldcollections
#
# There is no warranty for this code
#
# @version 0.1
# @author David Kammer <kammer@cornell.edu>
# @date     
# @modified 2017/09/14
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d

import ifasha.datamanager as idm

import ppscripts.postprocess as pp

def separate_at_dist_data(sname, group, pos_y, **kwargs):

    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)

    new_group_base = kwargs.get('new_group_base',group+'-')

    # check if fldids are FieldId nor string, else convert
    pos_y = idm.FieldId.string_to_fieldid(pos_y)

    # of type FieldId
    old_fields_ids = kwargs.get('fields',None) # if you want to provide the fields

    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)
    data.unpack()

    # get y position at initial step
    tidx       = data.get_index_of_closest_time(idm.FieldId('time'),0.)
    pos_y_mmap = data.get_field_at_t_index(pos_y, tidx)

    # find all heights present
    height_set = set()
    for p in pos_y_mmap[0,:]:
        height_set.add(p)
    print(height_set)

    # loop over all heights
    for h in height_set:

        print('height =',h)

        # create mask finding data needed for this height
        count = 0
        mask = np.zeros_like(pos_y_mmap)
        for (i,j), val in np.ndenumerate(pos_y_mmap):
            if val == h:
                count += 1
                mask[i,j] = count # new index (which starts with 1)
        print('shape',mask.shape)
        print('count',count)

        # create new FieldCollection
        nfc = dma.get_new_field_collection(new_group_base+'{0:1.2e}'.format(h))
        nfc.sim_info = data.sim_info

        new_nbts = data.get_full_field(idm.FieldId('time')).shape[0]

        if old_fields_ids is None:
            old_fields = data.get_all_fields()
        else:
            old_fields = list()
            for ofi in old_fields_ids:
                old_fields.append(data.get_field(ofi))

        # loop over all fields 
        for old_field in old_fields:
            print('Field',old_field.identity)

            # get old MemMap
            old_mmap = old_field.get_memmap('r')

            new_N = 1 # global field
            if old_mmap.shape[1] > 1:
                new_N = count # nodal field

            # create new Field
            new_field = nfc.get_new_field(old_field.identity,
                                          new_N,new_nbts,
                                          old_field.data_type,old_field.NEG)
            new_mmap = new_field.get_memmap('w+')

            # copy data
            if new_N > 1: # Nodal fields (needed to be filtered)
                for (i,j), val in np.ndenumerate(old_mmap):
                    if mask[0,j] > 0:
                        new_mmap[i,int(mask[0,j]-1)] = val 
            else: # Global fields (no need)
                for (i,j), val in np.ndenumerate(old_mmap):
                    new_mmap[i,j] = val

            print('max',np.amax(new_mmap))
            print('min',np.amin(new_mmap))

    data.pack()

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    error_message = """Usage: ./separate_at_dist_data.py sname group field-along-sep [base-name-of-new-groups]"""

    nb_args = len(sys.argv)
    if nb_args < 4 or nb_args > 5:
        print(error_message)
        sys.exit('Wrong arguments')

    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])

    if nb_args == 4:
        separate_at_dist_data(sname, group, fldid)
    else:
        gsname = str(sys.argv[4])
        separate_at_dist_data(sname, group, fldid, new_group_base=gsname)


