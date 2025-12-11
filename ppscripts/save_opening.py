#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import collections
from math import isnan
import numpy as np
import matplotlib.pyplot as plt
import pickle

import ifasha.datamanager as idm
import ifasha.rupturehunter as rh

import ppscripts.postprocess as pp

op_default_fname = 'opening.txt'

# -----------------------------------------------------------------------------
def get_opening(sname, **kwargs):
    
    zero_precision = 1e-12

    wdir = kwargs.get('wdir', './data')
    group = kwargs.get('group','interface')
    sname = pp.sname_to_sname(sname)
    input_data = pp.get_input_data(sname,**kwargs)

    # get data
    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)

    dim = 2
    if 'spatial_dimension' in input_data.keys():
        if input_data['spatial_dimension'] == 3 :
            dim =3

    # position
    if dim == 2:
        if data.has_field(idm.FieldId('position',0)):
            position = data.get_field_at_t_index(idm.FieldId('position',0),0) # position at beginning
        elif data.has_field(idm.FieldId('coord',0)):
            position = data.get_field_at_t_index(idm.FieldId('coord',0),0) # position at beginning
        else:
            raise('Does not have any position information')
    else:
        position_x = data.get_field_at_t_index(idm.FieldId('position',0),0) # position at beginning      
        position_z = data.get_field_at_t_index(idm.FieldId('position',2),0) # position at beginning      

    # stick slip information
    if data.has_field(idm.FieldId('is_in_contact',0)):
        is_in_contact = data.get_full_field(idm.FieldId('is_in_contact',0))
    elif data.has_field(idm.FieldId('top_disp',1)) and \
         data.has_field(idm.FieldId('bot_disp',1)):
        # convert displacement to is_sticking
        top_disp = data.get_full_field(idm.FieldId('top_disp',1))
        bot_disp = data.get_full_field(idm.FieldId('bot_disp',1))
        is_in_contact = np.abs(top_disp - bot_disp) < zero_precision
        is_in_contact.astype(np.int)
    elif data.has_field(idm.FieldId('top_disp',1)):
        # convert displacement to is_sticking
        top_disp = data.get_full_field(idm.FieldId('top_disp',1))
        is_in_contact = top_disp < zero_precision
        is_in_contact.astype(np.int)
    elif data.has_field(idm.FieldId('cohesion',1)):
        # convert displacement to is_sticking
        cohesion = data.get_full_field(idm.FieldId('cohesion',1))
        is_in_contact = cohesion < -zero_precision
        is_in_contact.astype(np.int)
    else:
        print('do not have necessary fields')
        raise RuntimeError

    time = data.get_full_field(idm.FieldId('time'))

    hunter = rh.RuptureHunter()
    if dim==2: # 1D interface
        hunter.load(position,time,is_in_contact)
    else:# 2D interface
        hunter.load2D(position_x,position_z,time,is_in_contact)
        
    hunter.hunt()
    if len(hunter.get_rupture_indexes()):
        hunter.renumber()

    opening = list()
    rptidxs  = hunter.get_rupture_indexes()
    for rptidx in rptidxs:
        opening.append(hunter.get_rupture(rptidx))

    return opening


# -----------------------------------------------------------------------------
def save_opening(snames,**kwargs):

    wdir = kwargs.get('wdir','./data')

    for sname in snames:
        sname = pp.sname_to_sname(sname)
        fname = kwargs.get('rpt_fname',op_default_fname)

        # get opening
        try:
            opening = get_opening(sname, **kwargs)
        except Exception as e:
            print('Could not get opening for: {}'.format(sname))
            print("Exception: {}".format(e.message))
            continue
        
        # save to temporary file
        tmp_fname = 'save_opening.tmp'
        with open(tmp_fname, 'w') as fl:
            pickle.dump(opening, fl)

        # copy temporary file into datamanager
        dm = idm.DataManager(sname,wdir)
        dm.add_supplementary(fname,tmp_fname,True)

# ------------------------------------------------------------------------------
def load_opening(sname,**kwargs):

    sname = pp.sname_to_sname(sname)

    # use data manager to load ruptures
    wdir = kwargs.get('wdir','./data')
    fname = kwargs.get('rpt_fname',op_default_fname)
    dm = idm.DataManager(sname,wdir)

    try:
        open(dm.get_supplementary(fname),'rb')
    except:
        save_opening([sname], **kwargs)
        dm = idm.DataManager(sname,wdir)

    with open(dm.get_supplementary(fname),'rb') as fl:
        opening = pickle.load(fl)

    return opening


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.exit('Missing argument! usage: ./save_opening.py simid[s]')
        
    simids = pp.string_to_simidlist(str(sys.argv[1]))
    save_opening(simids)

