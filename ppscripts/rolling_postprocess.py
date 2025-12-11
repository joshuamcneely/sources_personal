#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import datetime
import sys
import time
import glob

import ppscripts.postprocess as pp
import ppscripts.simid_bname_data as sbd

# ------------ CHECK INPUT
nb_argv = len(sys.argv)
if nb_argv < 4 or nb_argv > 5:
    sys.exit('Missing argument! usage: ./rolling_postprocess.py nb-days-to-run path-to-check group [file-to-save]')

nb_days = 1+int(sys.argv[1])
path = sys.argv[2]

# default file
if nb_argv == 4:
    group = sys.argv[3]
    fname = sbd.default_fname.format(group)
    print('use simid-bname file: {}'.format(fname))
else:
    fname = str(sys.argv[4])

# ---------------------------------

# check and set dates
print('Today is {}'.format(datetime.date.today()))
end_date = str(datetime.date.today() + datetime.timedelta(days=nb_days))
print('End date is {}'.format(end_date))

# load file with info about simulations already postprocessed
id_to_name = sbd.get_simid_data(fname)

# problem simulations: exist already or other problems
banned = set()

while not str(datetime.date.today()) == end_date:
    time.sleep(3)
    finished_runs = set()
    for fl in glob.glob(path+'/*.o????'):
        finished_runs.add(fl.strip().split('.o')[1])

    for finished_run in finished_runs:

        # already postprocessed
        if finished_run in id_to_name or finished_run in banned:
            continue
    
        success,bname = pp.postprocess(finished_run,'save')
        #success,bname = postprocess(finished_run,'forced')

        if success:
            id_to_name[finished_run] = bname
            sbd.write_simid_file(id_to_name, fname)
        else:
            banned.add(finished_run)
            
            # write banned sim-id to file
            with open('banned.txt', 'w') as f:
                for value in banned:
                    print(value, file=f)
