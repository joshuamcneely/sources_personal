#!/usr/bin/env python
from ppscripts.at_time_3d import * 
import ppscripts.postprocess as pp
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
pretty_label={'cohesion_1':r'interface traction $\tau_y$ (Pa)',
              'top_velo_1':r'particle velocity $\dot u_y$ (m/s)',
              'tau_max':r'yield strength $\sigma_Y$ (Pa)'}

def plot_at_time_3d(sname, group, fid, **kwargs):
    fldid = idm.FieldId()
    fldid.load_string(fid)

    input_data = pp.get_input_data(sname,**kwargs)  

    X,Z,V = get_at_time_3d(sname,group,fldid,**kwargs)

    X = X-input_data['x_length']/2.0
    Z = Z-input_data['z_length']/2.0

    zmax = kwargs.get('zmax',None)
    zmin = kwargs.get('zmin',None)

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)   

    #X,Z = idm.FieldCollectionAnalysis.make_pretty(X,Z)

    print(X.shape,Z.shape,V.shape)

    print('min',np.min(V))
    print('max',np.max(V))

    if zmin is not None and zmax is not None:
        fg1 = ax.pcolor(X,Z,V,vmin=zmin,vmax=zmax)
    elif zmin is not None:
        fg1 = ax.pcolor(X,Z,V,vmin=zmin)
    elif zmax is not None:
        fg1 = ax.pcolor(X,Z,V,vmax=zmax)
    else:
        fg1 = ax.pcolor(X,Z,V)


    if 'time' in kwargs.keys():
        time = kwargs.get('time')
        
        wdir = kwargs.get('wdir','./data')

        # in cases snames are actually sim-ids
        sname = pp.sname_to_sname(sname)
        dma = idm.DataManagerAnalysis(sname,wdir)
        data = dma(group)
        tidx = data.get_index_of_closest_time(idm.FieldId('time'),time)
    else:
        tidx = kwargs.get('time_idx')
        
    ax.set_title("{} {} -- {}".format(fldid.name.replace('_',' '),fldid._i,tidx))

    cbar=plt.colorbar(fg1,ax=ax)
    cbar.set_label(fldid.get_string())
    fg1.axes.set_aspect(1)

    ax.legend(loc='best')
    return cbar


#------------------------
if __name__=="__main__":
    sname = sys.argv[1]#'stick_break_3d_'+sys.argv[1]#3d_02_restart'
    notch=0.01

    group='interface'
    fldid='cohesion_1'

    input_data = pp.get_input_data(sname)  
    start_time = input_data["nuc_tstart"] 
    end_time = input_data["nuc_tstart"]+200e-6

    z_coord = input_data['z_length']/2.0 
    x_coord = input_data['x_length']/2.0+input_data['het_xstart']
    print('x_coord',x_coord)
    start_time = input_data['nuc_tstart']
    nbx = 2*1024

    if 1:
        try:
            fdir=sname
            os.mkdir(fdir)
        except FileExistsError:
            pass
        for time in np.linspace(start_time,end_time,20):
            fig,axes = plt.subplots(1,3,figsize=(12,8))

            for fldid,ax in zip(['tau_max',
                                 'cohesion_1',
                                 'top_velo_1'],
                                axes):
                try:
                    cbar = plot_at_time_3d(sname, group, fldid,time=time,ax=ax,
                                           x_start_fct=0.5)
                except:
                    print('did not work')
                else:
                    cbar.set_label(pretty_label[fldid])
                ax.set_title('')
                ax.set_xlabel('$x$')
                ax.set_ylabel('$z$')
            fig.suptitle(r'{} -- $t$={:1.0f} $\mu$s'.format(sname.replace('_',' '),time*1e6))

            plt.savefig('{}/xz_at_t{:1.0f}.png'.format(fdir,time*1e6),dpi=300)
            #plt.show()
    if 1:
        try:
            fdir=sname+'_zoom'
            os.mkdir(fdir)
        except FileExistsError:
            pass
        for time in np.linspace(start_time,end_time,20):
            fig,axes = plt.subplots(1,3,figsize=(12,8))

            for fldid,ax in zip(['tau_max',
                                 'cohesion_1',
                                 'top_velo_1'],
                                 axes):

                cbar = plot_at_time_3d(sname, group, fldid,time=time,ax=ax,
                                       z_start_fct=0.5,
                                       x_start_fct=0.5,
                                       x_end_fct=0.5+(notch*2)/input_data['x_length'])
                
                cbar.set_label(pretty_label[fldid])
                ax.set_title('')
                ax.set_xlabel('$x$')
                ax.set_ylabel('$z$')
            fig.suptitle(r'{} -- $t$={:1.0f} $\mu$s'.format(sname.replace('_',' '),time*1e6))

            plt.savefig('{}/xz_at_t{:1.0f}.png'.format(fdir,time*1e6),dpi=300)
            #plt.show()    
