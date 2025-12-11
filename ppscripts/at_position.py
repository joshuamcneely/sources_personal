#!/usr/bin/env python

# at_position.py
#
# plot FieldId for a given position
#
# There is no warranty for this code
#
# @version 0.1
# @author Gabriele Albertini <ga288@cornell.edu>
# @author David Kammer <dkammer@ethz.ch>
# @date     2017/07/21
# @modified 2019/06/18
from __future__ import print_function, division, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import sys

import ifasha.datamanager as idm

import ppscripts.postprocess as pp

def get_at_position(sname, group, fldid, position, **kwargs):

    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)
    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)

    # position
    if data.has_field(idm.FieldId('position',0)):
        pos_fdnm = 'position'
    elif data.has_field(idm.FieldId('coord',0)):
        pos_fdnm = 'coord'
    else:
        raise('Does not have any position information')


    pflds = [idm.FieldId(pos_fdnm,i) for i in range(len(position))]
    pidx = data.get_index_of_closest_position(pflds,position);
    print('pos',position,'index',pidx)
            
    time = data.get_full_field(idm.FieldId('time'))
    field = data.get_field_at_node_index(fldid, pidx)
    
    return time, field


def plot_at_position(snames, groups, fldids, positions, **kwargs):

    with_legend = kwargs.get('with_legend',True)

    # check if fldids are FieldId nor string, else convert
    fldids = idm.FieldId.string_to_fieldid(fldids)

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)  
        
    for (sname,group) in zip(snames,groups):
        sname = pp.sname_to_sname(sname)

        for (position,fldid) in zip(positions,fldids):
            
            time, field = get_at_position(sname, group,
                                          fldid, position,
                                          **kwargs)

            print(field.shape, time.shape)
            print('fld min max',
                  np.min(field[0,:]),
                  np.max(field[0,:]))

            ax.plot(time,
                    field,
                    '.-',
                    label='{} - x={}'.format(fldid.get_string(),
                                             position))
    
    if new_figure or with_legend:
        ax.legend(loc='best')

# ------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 5:
        sys.exit('Missing argument! usage: ./at_position.py '
                 + 'sname/sim-id group fldid posx,posy,posz')

    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])
    pos_tmp = sys.argv[4].split(',')
    position = [float(p) for p in pos_tmp]

    plot_at_position([sname],[group],[fldid],[position])

    plt.show()
