#!/usr/bin/env python

# simulation_status_update.py
#
# Code to manage simulations input and outputs
#
# There is no warranty for this code
#
# @version 0.1
# @author Gabriele Albertini <galberti819@gmail.com>
# @date     2016/03/04
# @modified 2016/03/04

from __future__ import print_function
import sys
import getopt
from ifasha.simulationmanager import SimulationManager
from ifasha.simulationmanager.utilities import Usage
import my_simulationmanager.sm_info as sm_info
import numpy as np

import time
from insert_comment import main
from subprocess import call


def main(argv=None):
    """ 
    Usage: launch.py <database_name> <sim_id> <sim_id_n>
    """

    wdir = sm_info.wdir
    source_dir = sm_info.source_dir
    
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hov:", ["help", "output="])
        except getopt.error, msg:
             raise Usage(msg)
            
        # option processing
        for option, value in opts:
            if option in ("-h", "--help"):
                raise Usage(main.__doc__)

        if len(args) not in [2,3]:
            raise Usage(main.__doc__)
        

    except Usage, err:
        print(sys.argv[0].split("/")[-1] + ": " + str(err.msg))
        print("\t for help use --help")
        return 2

    #-----------------------------------------------------------------------------

    database_name   = args[0]
    sim_id_1        = int(args[1])
    
    if len(args) == 3:
        sim_id_n = int(args[2])
    else:
        sim_id_n = sim_id_1


    sim_ids = np.arange(sim_id_1, sim_id_n + 1)
    
    #print(sim_ids)

    command = 'qsub'
    for sim_id in sim_ids:
        sub_file ='{0}_{1:0>4}.sub'.format(database_name, sim_id)
        print(sub_file)

        sm = SimulationManager(database_name, wdir, source_dir) 
        if not sm.launch_simulation_check(sim_id):
            if not sm.user_input('Do you want to continue?'):
                continue

        call([command, sub_file])
    

        new_status = sm.status.submitted
        sm.update_status(sim_id, new_status, True)
        time.sleep(0.5)

if __name__ == "__main__":
    sys.exit(main())
    os.system("echo '" + lines + " ' >> " + scriptname)

if __name__ == "__main__":
    sys.exit(main()) 
