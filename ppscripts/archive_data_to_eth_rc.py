#!/usr/bin/env python3

# archive_data_to_eth_rc.py
#
# archive data for ETH Research Collection
#
# There is no warranty for this code
#
# @version 0.1
# @author David Kammer <dkammer@ethz.ch>
# @date     2021/04/20
# @modified 2021/04/20

import sys, os #glob
import shutil as shtl
from pathlib import Path
import numpy as np
import h5py

import ifasha.datamanager as idm
import ppscripts.postprocess as pp

# ------------------------------------------------------------------------------
# get simids from README file
# everything that follows a simtag are simids (separated by commas)
def get_simid_from_readme(fname,**kwargs):
    simtag=kwargs.get('simtag','#simids')
    simids = []
    with open(fname) as fp:
        lines = fp.readlines()
        for line in lines:
            if simtag in line:
                simids.append([s.strip() for s in line.split(simtag)[1].split(',')])
    return simids
    
    
# copy supp files from data to idata folder
def copy_supp_files(snames,**kwargs):

    wdir = kwargs.get('wdir','./data')
    odir = kwargs.get('odir','./idata')
    if not os.path.exists(odir):
        raise RuntimeError('odir does not exist: '+odir)

    ext = kwargs.get('ext','')
    excl = kwargs.get('excl',None) # e.g. ['.v.','.p.']
    
    for sname in snames:
        sname = pp.sname_to_sname(sname)
        dm = idm.DataManager(sname,wdir)

        to_copy = [i for i in dm.supplementary if i.endswith(ext)]

        if excl:
            for e in excl:
                to_copy = [i for i in to_copy if e not in i]

        for f in to_copy:
            fpath = dm.get_supplementary(f)
            shtl.copyfile(fpath, odir+'/'+f+'.txt')


# ---------------------------------------------------------------------
def extract_group(sname,group,**kwargs):

    sname = pp.sname_to_sname(sname)
    wdir = kwargs.get('wdir','./data')
    odir = kwargs.get('odir',wdir)

    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)

    h5f = h5py.File(odir+'/'+group+'.h5', 'w')

    flds = data.get_all_fields()
    for fld in flds:
        print(fld)
        mmap = fld.get_memmap()
        h5f.create_dataset(fld.get_id_string(), data=mmap)

    h5f.close()
    # can be accessed with
    # h5f = h5py.File('data.h5','r')
    # t = h5f['time'][:]

                    
# -------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.exit('Missing argument! usage: ./archive_data_to_eth_rc.py <README file>')

    # readme file contain simids
    readme = sys.argv[1]

    # each line starting with #simids will be packed in a tar file
    simids_lists = get_simid_from_readme(readme)
    countfmt = '{:0'+str(int((len(simids_lists)+1)/10)+1)+'d}' # for leading zeros

    # loop over tar files
    tarcount = 0
    for simids in simids_lists:

        # folder name
        tarcount+=1
        odir = readme.strip('.txt')+'-data-'+countfmt.format(tarcount)

        # create folder
        Path(odir).mkdir(parents=True, exist_ok=True)

        print('tar {} with simids: '.format(tarcount)+' '.join(simids))
        for simid in simids:
            sname = pp.simid_to_sname(simid)
            oodir = odir+"/"+simid

            # create subfolder for simulation
            Path(oodir).mkdir(parents=True, exist_ok=True)

            # copy .in files
            copy_supp_files([sname],odir=oodir,ext='.in',excl=['.v.','.p.'])

            # extract all the fields from a given group
            extract_group(sname,'interface',odir=oodir)

        # create a tar file
        print("start building tar {}".format(tarcount))
        import tarfile
        tar = tarfile.open(odir+".tar", "w")
        tar.add(odir)
        tar.close()
