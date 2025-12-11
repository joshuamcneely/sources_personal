#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

from ppscripts.space_time import *

def get_space_time_diff(bname, group, fldid, **kwargs):
    X,T,Z = get_space_time(bname, group, fldid, **kwargs)

    Z[:,:]-=Z[0,:]
    return X,T,Z

def plot_space_time_diff(bname, group, fldid, **kwargs):
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)  

    zmax = kwargs.get('zmax',None)
    zmin = kwargs.get('zmin',None)

    X,T,Z = get_space_time_diff(bname,group,fldid, **kwargs)
    
    if zmin is not None and zmax is not None:
        fg1 = ax.pcolor(X,T,Z,vmin=zmin,vmax=zmax)
    elif zmin is not None:
        fg1 = ax.pcolor(X,T,Z,vmin=zmin)
    elif zmax is not None:
        fg1 = ax.pcolor(X,T,Z,vmax=zmax)
    else:
        fg1 = ax.pcolor(X,T,Z)

    if new_figure:
        ax.set_title(fldid.get_string().replace('_',' '))

        plt.colorbar(fg1)

    return fg1


# -----------------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) not in  [4,5]:
        sys.exit('Missing argument! usage: ./space_time_diff.py '
                 + 'bname/sim-id group fldid (zcoord)')
        
    bname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldid = str(sys.argv[3])
    try: 
        z_coord=float(sys.argv[4])
    except:
        z_coord=0.0

    fid = idm.FieldId()
    fid.load_string(fldid)

    # use a preceptually uniform colormap
    plt.rcParams['image.cmap'] = 'viridis'#, 'magma', 'plasma', 'inferno', 'cividis'

    plot_space_time_diff(bname,group,fid,zcoord=z_coord)

    plt.show()
