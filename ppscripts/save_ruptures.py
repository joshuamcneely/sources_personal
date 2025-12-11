#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import collections
from math import isnan
import numpy as np
import matplotlib.pyplot as plt
import pickle

import ifasha.datamanager as idm
import ifasha.rupturehunter as irh

import ppscripts.postprocess as pp


rpt_default_fname = 'ruptures.txt'
# -----------------------------------------------------------------------------
def get_is_sticking(data,d_slip):
    # stick slip information
    zero_precision = 1e-12

    if data.has_field(idm.FieldId('is_sticking',0)):
        is_sticking = data.get_full_field(idm.FieldId('is_sticking',0))
    elif data.has_field(idm.FieldId('binary_shear_info',0)):
        is_sticking = data.get_full_field(idm.FieldId('binary_shear_info',0))
    elif (data.has_field(idm.FieldId('top_disp',0)) and data.has_field(idm.FieldId('bot_disp',0)) and
          data.has_field(idm.FieldId('top_disp',1)) and data.has_field(idm.FieldId('bot_disp',1)) and d_slip):
        print("top bot sqrt(d0^2 + d1^2)>d_slip")
        top_disp_0 = data.get_full_field(idm.FieldId('top_disp',0))
        bot_disp_0 = data.get_full_field(idm.FieldId('bot_disp',0))
        top_disp_1 = data.get_full_field(idm.FieldId('top_disp',1))
        bot_disp_1 = data.get_full_field(idm.FieldId('bot_disp',1))
        is_sticking = np.sqrt((top_disp_0 - bot_disp_0)**2 + (top_disp_1 - bot_disp_1)**2) < d_slip
        is_sticking = is_sticking.astype(np.int)
        
    elif (data.has_field(idm.FieldId('top_disp',1)) and d_slip):
        print("top only sqrt(d1^2)>d_slip")
        top_disp_1 = data.get_full_field(idm.FieldId('top_disp',1))
        is_sticking = np.sqrt(((top_disp_1)**2)) < d_slip/2.0
        is_sticking = is_sticking.astype(np.int)
    elif (data.has_field(idm.FieldId('top_velo',1)) and d_slip):
        print("top only sqrt(d1^2)>d_slip")
        top_disp_1 = data.get_full_field(idm.FieldId('top_velo',1))
        is_sticking = np.sqrt(((top_disp_1)**2)) < d_slip/2.0
        is_sticking = is_sticking.astype(np.int)
        
    elif data.has_field(idm.FieldId('top_disp',0)) and data.has_field(idm.FieldId('bot_disp',0)):
        # convert displacement to is_sticking
        top_disp = data.get_full_field(idm.FieldId('top_disp',0))
        bot_disp = data.get_full_field(idm.FieldId('bot_disp',0))
        is_sticking = np.abs(top_disp - bot_disp) < zero_precision
        is_sticking = is_sticking.astype(np.int)
    elif data.has_field(idm.FieldId('top_disp',0)):
        # convert displacement to is_sticking
        top_disp = data.get_full_field(idm.FieldId('top_disp',0))
        is_sticking = top_disp < zero_precision
        is_sticking = is_sticking.astype(np.int)
    elif data.has_field(idm.FieldId('displacement',0)):
        print("disp")

        top_disp = data.get_full_field(idm.FieldId('displacement',0))
        print(top_disp.shape)
        top_disp=np.array([top_disp[i]-top_disp[0] for i in range(top_disp.shape[0])])
        print(top_disp.shape)
        is_sticking = top_disp < zero_precision
        is_sticking = is_sticking.astype(np.int)
    else:
        raise RuntimeError("no sticking data")
    return is_sticking
    
# -----------------------------------------------------------------------------
def get_ruptures(sname, **kwargs):
    
    d_slip = kwargs.get('d_slip', 0)
    d_slip = float(d_slip)    

    wdir = kwargs.get('wdir', './data')
    group = kwargs.get('group','interface')
    sname = pp.sname_to_sname(sname)

    # get data
    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)

    dim = 2
    if data.has_field(idm.FieldId('position',2)) or data.has_field(idm.FieldId('coord',2)): 
        dim =3
    if data.has_field(idm.FieldId('position',0)):
        posfld='position'
    elif data.has_field(idm.FieldId('coord',0)):
        posfld='coord'
    else:
        raise('Does not have any position information')

    # position
    if dim == 2:
        position = data.get_field_at_t_index(idm.FieldId(posfld,0),0) # position at beginning
    else:
        position_x = data.get_field_at_t_index(idm.FieldId(posfld,0),0) # position at beginning      
        position_z = data.get_field_at_t_index(idm.FieldId(posfld,2),0) # position at beginning      
    is_sticking = get_is_sticking(data,d_slip)
    #plt.pcolormesh(is_sticking)
    #plt.show()
    print(np.max(is_sticking),np.min(is_sticking))
    time = data.get_full_field(idm.FieldId('time'))
    print(time.shape)
    hunter = irh.RuptureHunter()
    if dim==2: # 1D interface
        hunter.load(position,time,is_sticking)
    else:# 2D interface
        hunter.load2D(position_x,position_z,time,is_sticking)
        
    hunter.hunt()

    ruptures = list()
    rptidxs  = hunter.get_rupture_indexes()
    for rptidx in rptidxs:
        ruptures.append(hunter.get_rupture(rptidx))

    return ruptures


# -----------------------------------------------------------------------------
def save_ruptures(snames,**kwargs):

    wdir = kwargs.get('wdir','./data')

    
    for sname in snames:
        sname = pp.sname_to_sname(sname)
        fname = kwargs.get('rpt_fname',rpt_default_fname)

        # get ruptures
        ruptures = get_ruptures(sname, **kwargs)

        # save to temporary file
        tmp_fname = 'save_ruptures.tmp'
        with open(tmp_fname, 'wb') as fl:
            pickle.dump(ruptures, fl)
        print('save '+fname)
        # copy temporary file into datamanager
        dm = idm.DataManager(sname,wdir)
        dm.add_supplementary(fname,tmp_fname,True)

# ------------------------------------------------------------------------------
def load_ruptures(sname,**kwargs):

    sname = pp.sname_to_sname(sname)

    # use data manager to load ruptures
    wdir = kwargs.get('wdir','./data')
    fname = kwargs.get('rpt_fname',rpt_default_fname)
    dm = idm.DataManager(sname,wdir)

    print(fname)
    try:
        open(dm.get_supplementary(fname),'rb')
        print('found')
    except:
        save_ruptures([sname], **kwargs)
        dm = idm.DataManager(sname,wdir)
        
    with open(dm.get_supplementary(fname),'rb') as fl:
        ruptures = pickle.load(fl)

    return ruptures


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.exit('Missing argument! usage: ./save_ruptures.py simid[s]')

    # simid can be 22,24,30-33 -> [22,24,30,31,32,33]
    simids = pp.string_to_simidlist(str(sys.argv[1]))
    save_ruptures(simids)

