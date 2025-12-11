#!/usr/bin/env python
from __future__ import print_function

import numpy as np
from collections import Iterable

import ifasha.datamanager as idm

import ppscripts.postprocess as pp
from ppscripts.space_time import get_space_time


def save_txt(fname,fld):
    print('save', fname)
    np.savetxt(fname,fld,delimiter=",")

def extract_xt_data_to_text_file(sname,group,fldids,**kwargs):
    wdir = kwargs.get('wdir','./data')

    renameflds = kwargs.get('fldrename',None)
    if renameflds is None:
        renameflds =fldids

    if 'zcoord' in kwargs.keys():
        raise RuntimeError('Need to implement is 3d in 2d')
    
    for f in range(len(fldids)):
        X,T,F = get_space_time(sname,group,fldids[f],**kwargs)
       
        save_txt("{}_{}_{}.csv".format(sname,group,renameflds[f]), F)
    
    positions_x = X[0,:]
    time        = T[:,0]
    #------------------------------------------

    save_txt("{}_{}_time.csv".format(sname,group), time)
    save_txt("{}_{}_x.csv".format(sname,group), positions_x)


    print('done printing')

# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    if len(sys.argv) <4:
        sys.exit('Missing argument! usage: ./extract_data_to_text_file.py sname/sim-id group fldid0 fldid1 fldid2 ... ')
    
    sname = str(sys.argv[1])
    group = str(sys.argv[2])
    fldstrs = [str(sys.argv[i]) for i in range(3,len(sys.argv))]
    
    # check if fldids are FieldId nor string, else convert
    fldids = idm.FieldId.string_to_fieldid(fldstrs)

    extract_data_to_text_file(sname,group,fldids)
