#!/usr/bin/env python
# at_time_3d_diff.py
#
# plot FieldId for a given time
#
# There is no warranty for this code
#
# @version 0.1
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     ?
# @modified 2018/09/12
from __future__ import print_function, division, absolute_import

from ppscripts.at_time_3d import *
import copy

def get_at_time_3d_diff(bname,group,fid,**kwargs):
    Xi,Zi,Vi = get_at_time_3d(bname,group,fid,**kwargs)

    if 'time' in kwargs.keys():
        t=kwargs['time']
        kwargs['time']=0
    else:
        t=kwargs['time_idx'];
        kwargs['time_idx']=0
    X0,Z0,V0 = get_at_time_3d(bname,group,fid,**kwargs)

    if 'time' in kwargs.keys():
        kwargs['time']=t
    else:
        kwargs['time_idx']=t
    
    return Xi,Zi,Vi-V0

def plot_at_time_3d_diff(bname,group,fid,**kwargs):
    X,Z,V = get_at_time_3d_diff(bname,group,fid,**kwargs)

    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)  

    # use a preceptually uniform colormap
    plt.rcParams['image.cmap'] = 'viridis'#, 'magma', 'plasma', 'inferno', 'cividis'
    
    Xp,Zp = X,Z;#FieldCollectionAnalysis.make_pretty(X,Z)

    print(Xp.shape,Zp.shape,V.shape)

    print('min',np.min(V))
    print('max',np.max(V))

    fg1 = ax.pcolor(Xp,Zp,V)
    if 'time' in kwargs.keys():
        time = kwargs.get('time')
        
        wdir = kwargs.get('wdir','./data')

        # in cases bnames are actually sim-ids
        bname = simid_to_bname(bname,**kwargs)
        dma = DataManagerAnalysis(bname,wdir)
        data = dma(group)
        tidx = data.get_index_of_closest_time(FieldId('time'),time)
    else:
        tidx = kwargs.get('time_idx')
    ax.set_title("{} {} -- {}".format(fid.name,fid._i,tidx))

    plt.colorbar(fg1)

    fg1.axes.set_aspect(1)

    ax.legend(loc='best')
    
# --------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 5:
        sys.exit('Missing argument! usage: ./at_time_3d.py bname/sim-id group fldid time_idx')

    bname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])
    #time = float(sys.argv[4])
    tidx=int(sys.argv[4])
    
    fid = FieldId()
    fid.load_string(fldid)

    # use a preceptually uniform colormap
    plt.rcParams['image.cmap'] = 'viridis'#, 'magma', 'plasma', 'inferno', 'cividis'

    plot_at_time_3d_diff(bname,group,fid,time_idx=tidx)
    
    plt.show()
    

