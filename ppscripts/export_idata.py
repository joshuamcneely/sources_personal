#!/usr/bin/env python

# export_idata.py
#
# export input and submission files from data folder
#
# There is no warranty for this code
#
# @version 0.1
# @author David Kammer <dkammer@ethz.ch>
# @date     2020/05/13
# @modified 2020/05/13
from __future__ import print_function, division, absolute_import

import glob, os, sys
import shutil as shtl

from ifasha.datamanager import DataManager

import ppscripts.postprocess as pp

# ------------------------------------------------------------------------------
# get all snames from wdir
def get_all_snames(**kwargs):

    wdir = kwargs.get('wdir','data/')
    snames = list()
    for fl in glob.glob(wdir+"*.datamanager.info"):
        snames.append(fl.strip().split(wdir)[1].split('.datamanager.info')[0])
    return snames

# copy supp files from data to idata folder
def copy_supp_files(snames,**kwargs):

    wdir = kwargs.get('wdir','./data')
    odir = kwargs.get('odir','./idata')
    if not os.path.exists(odir):
        raise RuntimeError('odir does not exist: '+odir)

    ext = kwargs.get('ext','')
    
    for sname in snames:
        sname = pp.sname_to_sname(sname)
        dm = DataManager(sname,wdir)

        to_copy = [i for i in dm.supplementary if i.endswith(ext)]

        for f in to_copy:
            fpath = dm.get_supplementary(f)
            shtl.copyfile(fpath, odir+'/'+f)

                    
# -------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.exit('Missing argument! usage: ./export_idata.py '
                 + 'comma-separated-extensions (e.g.: .in,.sub,.run)')

    exts = sys.argv[1].split(',')
        
    snames = get_all_snames()
    for ext in exts:
        copy_supp_files(snames,ext=ext)
