#!/usr/bin/env python

# update_simulation_data.py
#
# Code to udpate data structure from simulations
#
# There is no warranty for this code
#
# @version 0.1
# @author David Kammer <davekammer@gmail.com>
# @date     2015/12/01
# @modified 2015/12/01

import sys
import glob
import os.path
import shutil as shtl

from ifasha.datamanager import DataManager
from ifasha.datamanager import FieldId
from ifasha.datamanager import Field

from SimulationData import SimulationData
from SimulationData import FieldId as OldFieldId
from SimulationData import Field as OldField

# -----------------------------------------------------------------------------
bname = 'subRayleigh_dynamic_nucD2.0e-04_taui4.10e+06_sigi5.00e+06_E5.65e+09nu0.33rho1180psss1_lswnhs1.06k0.74d1.40e-06_L0.2H0.2_elem3200'

# old name and new name of groups
groups = [
    ['', 'interface'],
    #['-global', 'global'],
    #['-at-distance', 'at-distance'],
    #['-at-dist-7.5mm', 'at-dist-7.5mm'],
    #['-at-dist-3.5mm', 'at-dist-3.5mm']
]

wdir   = ''
job_id = 'unknown_job_id'


# -----------------------------------------------------------------------------
# verify it does not yet exist
try:
    dmioh = DataManager(bname,wdir,False)
except IOError: # it does not yet exist
    dmioh = DataManager(bname,wdir,True)
else:
    answer = raw_input('DataManager named "{}" exists already.\n'.format(dmioh.name)
                       + 'Want to: [m] modify it, [r] replace it, [s] stop here? ')
    answer = answer.strip().lower()
    if answer == 's':
        sys.exit('you stopped here.')
    elif answer == 'r':
        dmioh.destroy()
        dmioh = DataManager(bname,wdir,True)
    elif answer == 'm':
        pass
    else:
        sys.exit('Incorrect answer!')


# add input file to datamanager
dmioh.add_supplementary(job_id+'.in',wdir+'/'+bname+'.in',True)


# postprocessing each group
for group_names in groups:

    old_group = SimulationData()
    old_group.load(bname+group_names[0]+'.mmap.info',wdir)

    all_fields = old_group.fields.keys()
    print(all_fields)

    group = group_names[1]
    
    # check if FieldCollection exist already
    if group in dmioh:
        answer = raw_input('FieldCollection named "{}" exists already.\n'.format(group)
                           #+ 'Want to: [m] modify it, [r] replace it, [s] stop here? ')
                            + 'Want to: [p] pass it, [r] replace it, [s] stop here? ')
        answer = answer.strip().lower()

        if answer == 's':
            sys.exit('you stopped here')
        elif answer == 'p':
            print('passed!')
            continue
        elif answer == 'r':
            dmioh.remove_field_collection(group)
            fc = dmioh.get_new_field_collection(group)
        # does not work:
        #elif answer == 'm':
        #    fc = dmioh.get_field_collection(group)
        else:
            sys.exit('Incorrect answer!')

    else:
        fc = dmioh.get_new_field_collection(group)

    fc.sim_info = 'sim-id={}'.format(job_id)


    for fld in all_fields:
        fldcmps = fld.split('_')
        oldfldid = OldFieldId('_'.join(fldcmps[:-1]),fldcmps[-1])
        oldfld = old_group.getField(oldfldid)
        
        print(oldfld)

        newfldid = FieldId()
        if 'step' in fld or 'time' in fld:
            fld = '_'.join(fldcmps[:-1])
        newfldid.load_string(fld)
        

        newfld = Field(newfldid, oldfld.N, oldfld.nbts,
                       oldfld.type, oldfld.NEG)

        # this is dirty, but I am lazy
        newfld.mmap = oldfld.file.split('/')[-1]
        newfld.set_path('/'.join(oldfld.path.split('/')[:-1]))

        print(newfld)
        print(newfld.path)
        print('')

        fc.add_field(newfld)


