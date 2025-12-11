#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

from ppscripts.at_time import *
import copy

def get_at_time_diff(bname, group, fid, **kwargs):
    x,f = get_at_time(bname,group,fid,**kwargs)
    #kwargs0=copy.deepcopy(kwargs)
    kwargs0=kwargs;
    if 'time' in kwargs0.keys():
        kwargs0['time']=0
    else:
        kwargs0['time_idx']=0
    x,f0 = get_at_time(bname,group,fid,**kwargs0)

    return x,f-f0

def plot_at_time_diff(bname,group,fldid, **kwargs):

    # in cases bnames are actually sim-ids
    bname = simid_to_bname(bname,**kwargs)

    # add plot on ax of figure if already provided 
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)


    wdir = kwargs.get('wdir','./data')
    bname = simid_to_bname(bname,**kwargs)
    dma = DataManagerAnalysis(bname,wdir)
    data = dma(group)
        
    if 'time' in kwargs.keys():
        time = kwargs.get('time')
        
        # in cases bnames are actually sim-ids
        tidx = data.get_index_of_closest_time(FieldId('time'),time)
    else:
        tidx = kwargs.get('time_idx')
        time=tidx
    
    pos,field = get_at_time_diff(bname,group,fldid,**kwargs)
    ax.plot(pos,
            field,
            label='t={:1.2e}'.format(time))

    ax.set_title(fldid.get_string().replace('_',' '))
    if kwargs.get('legend',True):
        ax.legend(loc='best')


# --------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 5:
        sys.exit('Missing argument! usage: ./at_time.py bname/sim-id group fldid time')

    bname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])
    #time = float(sys.argv[4])
    tidx=int(sys.argv[4])
    
    fid = FieldId()
    fid.load_string(fldid)

    plot_at_time_diff(bname,group,fid,time_idx=tidx)

    plt.show()
