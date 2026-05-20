#!/usr/bin/env python
from plot_3d import plot_space_time,plot_rupture_contour_one_sim, pretty_label
import ppscripts.postprocess as pp
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
#------------------------
if __name__=="__main__":
    sname = sys.argv[1]#'stick_break_3d_'+sys.argv[1]#3d_02_restart'
    notch=0.01

    group='interface'
    fldid='cohesion_1'

    input_data = pp.get_input_data(sname)  
    start_time = input_data["nuc_tstart"] 
    end_time = input_data["duration"]#nuc_tstart"]+200e-6

    z_coord = input_data['z_length']/2.0 
    x_coord = input_data['x_length']/2.0+input_data['het_xstart']
    print('x_coord',x_coord)
    start_time = input_data['nuc_tstart']
    nbx = 2*1024

    # fig 1
    if 1:
        fig,axes=plt.subplots(5,figsize=(8,12))
        plt.suptitle(sname.replace('_',' '))


        print('plotting space time tau max')
        ax = axes[0]
        plot_space_time(sname,group,'tau_max',#fldid,
                        z_coord=z_coord,
                        z_coord_idx=2,
                        ax=ax,nb_x_elements=nbx,start_fct=0.0, end_fct=1,zmin=0,start_time=start_time)
        print('plotting space time velo')
        ax = axes[1]
        plot_space_time(sname,group,'top_velo_1',#fldid,
                        z_coord=z_coord,
                        z_coord_idx=2,
                        ax=ax,nb_x_elements=nbx,start_fct=0.0, end_fct=1,zmin=0,start_time=start_time)
        print('plotting space time cohesion xt')
        ax = axes[2]
        plot_space_time(sname,group,fldid,
                        z_coord=z_coord,
                        z_coord_idx=2,
                        ax=ax,nb_x_elements=nbx,start_fct=0.0, end_fct=1,zmin=0,start_time=start_time)
        print('plotting space time cohesion zt')
        ax = axes[3]
        plot_space_time(sname,group,fldid,
                        z_coord=x_coord,
                        z_coord_idx=0,
                        ax=ax,nb_x_elements=nbx,start_fct=0.0, end_fct=1,zmin=0,start_time=start_time)

        print('plotting rpt contour')
        ax = axes[4]
        plot_rupture_contour_one_sim(sname,
                                     start_time=start_time,
                                     end_time=end_time, ax=ax)
        ax.set_ylim(notch+input_data['nuc_xc0']/2.0,None)

        plt.tight_layout()

        plt.savefig('{}_xt_zt_xz.png'.format(sname),dpi=300)
        pickle.dump(fig,open('{}_xt_zt_xz.pkl'.format(sname),'wb'))
        #plt.show()
