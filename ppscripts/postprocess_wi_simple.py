#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import os

import ifasha.datamanager as idm

# input file name
def get_input_fname(sname):
    return '{}.in'.format(sname)
# output file name
def get_output_fname(sname):
    return '{}.out'.format(sname)

def add_io_files(dmioh,sdir,new_bname):
    # add input file to datamanager
    input_file = os.path.join(sdir,new_bname+'.in')
    dmioh.add_supplementary(get_input_fname(new_bname),input_file,True)
    
    progress_file = os.path.join(sdir,new_bname+'.progress')
    dmioh.add_supplementary(get_output_fname(new_bname),progress_file,True)

usage="""./postprocess.py <raw_data_dir/bname>"""
if len(sys.argv) != 2:
    sys.exit(usage)

# Parse the path to get simulation directory and basename
input_path = sys.argv[1]
if '/' in input_path:
    # Split path into directory and basename
    sdir = os.path.dirname(input_path)
    bname = os.path.basename(input_path)
else:
    bname = input_path
    sdir = '.'

wdir = 'data/'
new_bname = bname
groups = ['interface']

# verify it does not yet exist
try:
    dmioh = idm.DataManager(new_bname,wdir,False)
except IOError: # it does not yet exist
    dmioh = idm.DataManager(new_bname,wdir,True)

    # add io files to datamanager
    add_io_files(dmioh,sdir,new_bname)

else:
    answer = input('DataManager named "{}" exists already.\n'.format(dmioh.name) 
                       + 'Want to: [m] modify it, [r] replace it, [s] stop here? ')
    answer = answer.strip().lower()
    if answer == 's':
        sys.exit()
    elif answer == 'r':
        dmioh.destroy()
        dmioh = idm.DataManager(new_bname,wdir,True)
        # add io files to datamanager
        add_io_files(dmioh,sdir,new_bname)
    elif answer == 'm':
        # add io files to datamanager
        add_io_files(dmioh,sdir,new_bname)
        pass
    else:
        print('Incorrect answer!')
        sys.exit()


for group in groups:

    # check if FieldCollection exist already
    if group in dmioh:
        answer = input('FieldCollection named "{}" exists already.\n'
                           + 'Want to: [m] modify it, [r] replace it, [s] stop here? '.format(group))
        answer = answer.strip().lower()

        if answer == 's':
            sys.exit()
        elif answer == 'r':
            dmioh.remove_field_collection(group)
            fc = dmioh.get_new_field_collection(group)
        elif answer == 'm':
            fc = dmioh.get_field_collection(group)
        else:
            print('Incorrect answer!')
            sys.exit()

    else:
        fc = dmioh.get_new_field_collection(group)

    # "cast" FieldCollection to IOHelperReader
    fcwi = idm.FieldCollectionWeakInterface(fc)
    print(bname)
    if os.path.exists(os.path.join(sdir,bname+".info")):
        fcwi.read_simulation_output(bname+'.info', sdir)
    else:
        fcwi.read_simulation_output(bname+'-interface.info', sdir)
