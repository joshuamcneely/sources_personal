#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

import ifasha.datamanager as idm
import solidmechanics as sm
import solidmechanics.lefm as lefm

import ppscripts.postprocess as pp
from ppscripts.input_parameters import get_material_properties
from ppscripts.save_ruptures import load_ruptures

dsif_default_name = 'dyn_sif'

# --------------------------------------------------------------------------
def get_dyn_sif(sname, group, **kwargs):

    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)

    prop_dir = kwargs.get('prop_dir','to_right')

    top_mat_name = kwargs.get('top_mat_name','slider')
    bot_mat_name = kwargs.get('bot_mat_name','base')

    # get material properties
    materials = get_material_properties(sname,**kwargs)

    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)

    # find name of position
    if data.has_field(idm.FieldId('position',0)):
        posname = 'position'
    elif data.has_field(idm.FieldId('coord',0)):
        posname = 'coord'
    else:
        raise RuntimeError('Do not know position name')

    # find tip position
    rpt = load_ruptures(sname)[0]
    rpt = rpt.first()
    front = rpt.get_sorted_front()
    nuc_p, nuc_t = rpt.get_nucleation()
    nuc_i = np.argmin(abs(front[:,0]-nuc_p[0]))
    xnuc = front[nuc_i,0]

    # rupture speed
    avg_dist = kwargs.get('avg_dist',0.01)
    prop_speed = rpt.get_propagation_speed(0,avg_dist,**kwargs)

    # name of field to use for K2 determination
    fldnm = kwargs.get('fldnm','strain_2')
    fldid = idm.FieldId.string_to_fieldid(fldnm)
    window = kwargs.get('window',0.01)

    # coordinates
    x_pos = data.get_field_at_t_index(idm.FieldId(posname,0),0)
    y_pos = data.get_field_at_t_index(idm.FieldId(posname,1),0)
    # sort
    soi = np.argsort(x_pos, axis=1)
    sti = np.indices(x_pos.shape)
    x_pos = x_pos[sti[0],soi][0,:]
    y_pos = y_pos[sti[0],soi][0,:]

    if np.min(y_pos) != np.max(y_pos):
        raise RuntimeError('group must have y=const')

    # analytical solution for arbitrary sif
    K2 = kwargs.get('K2',1e3)

    K2s = list()

    times = data.get_full_field(idm.FieldId('time'))
    for tidx in range(len(times)):
        actual_time = times[tidx]

        if prop_dir == 'to_right':
            psidx = nuc_i + np.argmin(abs(front[nuc_i:,2] - actual_time))
        elif prop_dir == 'to_left':
            psidx = np.argmin(abs(front[:nuc_i,2] - actual_time))
        xtip = front[psidx,0]

        # if crack too short, this does not work
        if xtip - xnuc < window:
            continue

        # find rupture speed
        psidx = np.argmin(abs(prop_speed[:,0] - xtip))
        vtip = prop_speed[psidx,2] # rupture speed

        field = data.get_field_at_t_index(fldid,tidx)
        field = field[sti[0],soi][0,:]

        # focus on small window around crack tip
        lidx = np.argmin(abs(x_pos - xtip + window))
        ridx = np.argmin(abs(x_pos - xtip - window))
        x_pos_w = x_pos[lidx:ridx] - xtip
        y_pos_w = y_pos[lidx:ridx]
        field = field[lidx:ridx]

        # find local min and max of simulation
        izero =  np.argmin(abs(x_pos_w - 0.))
        # first max behind crack tip
        imax_sim = np.argmax(field[:izero])
        imin_sim = imax_sim + np.argmin(field[imax_sim:2*izero+imax_sim])
        dEPS_sim = field[imax_sim] - field[imin_sim]

        # analytical solution
        sigxx,sigyy,sigxy = lefm.mode_2_bimaterial_singular_stress_field(
            vtip,
            materials[top_mat_name],
            materials[bot_mat_name],
            x=x_pos_w,
            y=y_pos_w,
            K2=K2,
            rformat='element-wise')

        if y_pos[0] >= 0:
            local_mat = materials[top_mat_name]
        else:
            local_mat = materials[bot_mat_name]
            
        epsxx,epsyy,epsxy = local_mat.stresses_to_strains_2d(sigxx,
                                                             sigyy,
                                                             sigxy)
        
        eps = {'strain_0':epsxx,
               'strain_1':epsyy,
               'strain_2':epsxy}
        
        # find local min and max of theory
        izero =  np.argmin(abs(x_pos_w - 0.))
        # first max behind crack tip
        imax_theo = np.argmax(eps[fldnm][:izero])
        imin_theo = imax_theo + np.argmin(eps[fldnm][imax_theo:2*izero+imax_theo])
        dEPS_theo = eps[fldnm][imax_theo] - eps[fldnm][imin_theo]

        K2s.append([actual_time,K2 * dEPS_sim / dEPS_theo])
    
        if False:
            sigxx,sigyy,sigxy = lefm.mode_2_bimaterial_singular_stress_field(
                vtip,
                materials[top_mat_name],
                materials[bot_mat_name],
                x=x_pos_w,
                y=y_pos_w,
                K2=K2,
                rformat='element-wise')
            
            if y_pos[0] >= 0:
                local_mat = materials[top_mat_name]
            else:
                local_mat = materials[bot_mat_name]
                
            epsxx,epsyy,epsxy = local_mat.stresses_to_strains_2d(sigxx,
                                                                 sigyy,
                                                                 sigxy)
            
            eps = {'strain_0':epsxx,
                   'strain_1':epsyy,
                   'strain_2':epsxy}
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x_pos_w, field - field[0])
            ax.plot(x_pos_w[imax_sim],field[imax_sim]-field[0],'o')
            ax.plot(x_pos_w[imin_sim],field[imin_sim]-field[0],'o')
            #ax.plot(x_pos_w, eps[fldnm],'k--')
            plt.show()

    return np.array(K2s)

# -----------------------------------------------------------------------------
def save_dyn_sif(snames,group,**kwargs):

    wdir = kwargs.get('wdir','./data')

    for sname in snames:
        sname = pp.sname_to_sname(sname)
        fname = kwargs.get('dsif_fname',dsif_default_name+'.ifa')
        
        dsifs = get_dyn_sif(sname,group,**kwargs)

        # save to temporary file
        tmp_fname = 'save_dyn_sif.tmp'
        with open(tmp_fname, 'w') as fl:
            pickle.dump(dsifs, fl)

        # copy temporary file into datamanager
        dm = idm.DataManager(sname,wdir)
        dm.add_supplementary(fname,tmp_fname,True)
        
        
# ------------------------------------------------------------------------------
def load_dyn_sif(sname,**kwargs):

    wdir = kwargs.get('wdir','./data')
    sname = pp.sname_to_sname(sname)
    fname = kwargs.get('dsif_fname',dsif_default_name+'.ifa')
    
    # use data manager to load ruptures
    dm = idm.DataManager(sname,wdir)
    
    with open(dm.get_supplementary(fname),'rb') as fl:
        dsifs = pickle.load(fl)

    return dsifs


# --------------------------------------------------------------------------------
if __name__ == "__main__":

    #if len(sys.argv) != 5:
    #    sys.exit('Missing argument! usage: ./save_dyn_sif.py sname group')

    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fname = str(sys.argv[3])

    save_dyn_sif([sname], group, dsif_fname=fname)
