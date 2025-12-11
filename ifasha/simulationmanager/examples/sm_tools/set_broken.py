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
    Usage: set-broken.py <database_name> <sim_id> 

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
#        if len(args) is not 2:
 #           raise Usage(main.__doc__)
            
        # option processing
        for option, value in opts:
            if option in ("-h", "--help"):
                raise Usage(main.__doc__)

        if len(args)<1:
            raise Usage(main.__doc__)   

    except Usage, err:
        print >>sys.stderr, sys.argv[0].split("/")[-1] + ": " + str(err.msg)
        print >>sys.stderr, "\t for help use --help"
        return 2

    # print args

    database_name    = args[0]
    sim_id          = int(args[1])
    #status = args[2]

    # load SimulationManager
    
    sm = SimulationManager(database_name, wdir, source_dir) 
    #sm.update_status_forced(sim_id, status, print_out)
    #print('{}{:0>4} Status update: {}'.format(sm.database_name, sim_id, status))

    sm.broken(sim_id, True)


if __name__ == "__main__":
    sys.exit(main())
    os.system("echo '" + lines + " ' >> " + scriptname)

if __name__ == "__main__":
    sys.exit(main()) 
