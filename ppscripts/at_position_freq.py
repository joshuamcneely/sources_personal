#!/usr/bin/env python

# at_position_freq.py
#
# plot FieldId for a given position
#
# There is no warranty for this code
#
# @version 0.1
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     2017/07/21
# @modified 2017/07/21
from __future__ import print_function, division, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import sys

import ifasha.datamanager as idm

import ppscripts.postprocess as pp
from ppscripts.process_signal import *

#---
def get_tcrack(dma,position):
    
    data = dma('interface')
        
    pflds = [idm.FieldId('position',i) for i in range(len(position))]
    pidx = data.get_index_of_closest_position(pflds,position);print('pos',position,'index',pidx)
    t = np.array(data.get_full_field(idm.FieldId('time')))
    is_stick = np.array(data.get_field_at_node_index(idm.FieldId('is_sticking',0), pidx))[:,0]
 
    tid_crack = np.argmin(np.diff(is_stick))
    print(i)
    for i in range(len(is_stick)):
        if is_stick[i]==0:
            tid_crack = i
            print(i)
            t_crack = t[i]
            return t_crack,i
#---

def write_to_csv(fname, time, field):
    f = open(fname,'w')
    dl=', '
    nl='\n'
    f.write('time'+dl+'field'+nl)
    for t, y in zip(time,field):
        f.write(str(t)+dl+str(y)+nl)
    
#--- 


def at_position_freq(snames, groups, fldids, positions, **kwargs):
    """
    kwargs:
        wdir = ./data
        with_legend = True
        axes = None
        no_figure = False
        window = False
                 True : Blackman_Harris
        butterw = False
                 [order, fcut] : ButterWorth
    """
    wdir = kwargs.get('wdir','./data')

    with_legend = kwargs.get('with_legend',True)

    # check if fldids are FieldId nor string, else convert
    fldids = idm.FieldId.string_to_fieldid(fldids)

    # add plot on ax of figure if already provided
    axes = kwargs.get('axes', None)
    no_figure = kwargs.get('no_figure', False)
    if axes is None and no_figure is False:
        new_figure = True 
    else:
        new_figure = False

    if new_figure:
        fig = plt.figure()
        ax0 = fig.add_subplot(311)  
        ax1 = fig.add_subplot(312)  
        ax2 = fig.add_subplot(313)  
        axes=[ax0,ax1,ax2]
    # ---------------------------------------------------------------------
    for (sname,group) in zip(snames,groups):

        sname = pp.sname_to_sname(sname)
        dma = idm.DataManagerAnalysis(sname,wdir)
        data = dma(group)

        for (position,fldid) in zip(positions,fldids):

            pflds = [idm.FieldId('position',i) for i in range(len(position))]
            pidx = data.get_index_of_closest_position(pflds,position);print('pos',position,'index',pidx)
            time = np.array(data.get_full_field(idm.FieldId('time')))

            field = np.array(data.get_field_at_node_index(fldid, pidx))[:,0]

            print(field.shape, time.shape)
            print('fld min max',np.min(field),np.max(field))

            fix_dt = kwargs.get('fix_dt',None)
            time,field,dt = resample_const_time_step(time,field,dt=fix_dt,intp='linear')

            
            print(field.shape, time.shape)
            
            # take away initial condition
            field = field-field[0]
            
            
            # center
            tcrack,tcrackid = get_tcrack(dma,position)
            time = time - tcrack
            xyz=''
            for i, p in zip(['x','y','z'],position):
                xyz+='_{}{}'.format(i,p)

            write_to_csv('{}{}_raw_data.csv'.format(sname,xyz),time,field)
            
            time, field, s, fieldamp, fieldphs = analyse_time_series(time,field,**kwargs)
            dpa=''
            for k in kwargs.keys():
                dpa+='_{}{}'.format(k[0],str(kwargs[k]))
            write_to_csv('{}{}_ppp_data.csv'.format(sname,xyz),time,field)

            if no_figure:
                return time,field,s,fieldamp,fieldphs

            axes[0].plot(time,
                         field,
                         '.-',
                         label='{}, x='.format(fldid.get_string())+str(position))

            axes[1].plot(s,
                         fieldamp,
                         '.-',
                         label='{}, x='.format(fldid.get_string())+str(position))
            if len(axes)==3:
                axes[2].plot(s,
                             fieldphs,
                             '.-',
                             label='{}, x='.format(fldid.get_string())+str(position))
            axes[1].set_yscale('log')
            for ax in axes[1:]: ax.set_xscale('log')
    if new_figure or with_legend:
        axes[0].legend(loc='best')

# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) is not 5:
        sys.exit('Missing argument! usage: ./at_position_freq.py sname/sim-id group fldid posx,posy,posz')

    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])
    pos_tmp = sys.argv[4].split(',')
    position = [float(p) for p in pos_tmp]

    at_position_freq([sname],[group],[fldid],[position])

    plt.show()
