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

import sys
import getopt
from ifasha.simulationmanager import SimulationManager
from ifasha.simulationmanager.utilities import Usage
import my_simulationmanager.sm_info


def main(argv=None):
    """ 
    Usage: status_update_simulation.py <database_name> <sim_id> <new_status>
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

        if len(args) < 3:
            raise Usage(main.__doc__)
        

    except Usage, err:
        print >>sys.stderr, sys.argv[0].split("/")[-1] + ": " + str(err.msg)
        print >>sys.stderr, "\t for help use --help"
        return 2

    print args

    database_name   = args[0]
    sim_id          = int(args[1])
    new_status      = args[2]

    # load SimulationManager
    
    sm = SimulationManager(database_name, wdir, source_dir) 

    sm.update_status(sim_id, new_status)

    if new_status != "postprocessed":
        sm.set_time(sim_id, sm.time.when[new_status])
    

if __name__ == "__main__":
    sys.exit(main())
    os.system("echo '" + lines + " ' >> " + scriptname)

if __name__ == "__main__":
    sys.exit(main()) 
