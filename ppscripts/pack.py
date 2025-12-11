#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import os, sys

from ifasha.datamanager import DataManager

import ppscripts.postprocess as pp

# -------------------------------------------------------------------------
def pack(snames, **kwargs):

    wdir = kwargs.get('wdir', './data')

    for sname in snames:
        sname = pp.sname_to_sname(sname)
        try:
            dma = DataManager(sname,wdir)
        except:
            print('did not exist:',sname)
        else:
            print('pack:',sname)
            dma.pack()
    
# -------------------------------------------------------------------------
if __name__ == "__main__":

    error_message = """Usage: ./pack.py option fname
         options:
             list: file is list of snames
             exclude: file of list of snames to exclude from packing
         fname: file name to consider
    """

    nb_args = len(sys.argv)
    if nb_args != 3:
        print(error_message)
        sys.exit('Wrong arguments')

    action = str(sys.argv[1])
    fname = str(sys.argv[2])

    with open(fname) as f:
        file_snames = f.readlines()
        file_snames = [l.strip() for l in file_snames]
        file_snames = set(file_snames)

    if action == 'list':
        snames_to_pack = file_snames
    elif action == 'exclude':
        all_snames = pp.get_all_snames()!!! issue function does not exist!!!
        snames_to_pack = [n for n in all_snames if not n in file_snames]
    else:
        print(error_message)
        sys.exit()

    pack(snames_to_pack)
