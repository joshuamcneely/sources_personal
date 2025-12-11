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

import my_simulationmanager.sm_info as sm_info

import my_simulationmanager.sm_tools.show_simulations as show

def main(argv=None):
    """ 
    Usage: insert_comment.py <database_name> <sim_id> <comment>

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

        if len(args)<3:
            raise Usage(main.__doc__)   

    except Usage, err:
        print >>sys.stderr, sys.argv[0].split("/")[-1] + ": " + str(err.msg)
        print >>sys.stderr, "\t for help use --help"
        return 2

    # print args

    database_name    = args[0]
    sim_id          = int(args[1])
    comment = args[2]

    # load SimulationManager
    
    sm = SimulationManager(database_name, wdir, source_dir) 

    sm.update_record(sm.table.simulation, sim_id, 'comment', comment)
    print('{}{:0>4} Comment inserted: {} '.format(sm.database_name, sim_id, comment))

    show.main(database_name)


if __name__ == "__main__":
    sys.exit(main())
    os.system("echo '" + lines + " ' >> " + scriptname)

if __name__ == "__main__":
    sys.exit(main()) 

