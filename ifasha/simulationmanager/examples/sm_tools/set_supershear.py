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
from sm_dynamicsimulation import DynamicSimulation
from ifasha.simulationmanager.utilities import Usage
from ifasha.solidmechanics import lefm
import my_simulationmanager.sm_info

def main(argv=None):
    """ 
    Usage: set-supershear.py <database_name> <sim_id> <location where transition occurred>

    """    
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

    wdir = sm_info.wdir
    source_dir = sm_info.source_dir

    database_name    = args[0]
    sim_id          = int(args[1])
    transition      = float(args[2])

    # load DynamicSimulation
    if database_name[-7:] == 'dynamic':
        project_name = database_name[:-8]
    else:
        print 'option only available for DynamicSimulation'
        raise RuntimeError
    
    sm = DynamicSimulation(project_name, wdir, source_dir) 

    # set-supershear simensionless
    sm.is_supershear(sim_id, transition)
    
    fields = sm.get_relevant_parameters()
    param_dict = sm.get_global_parameter_dict(sim_id)
    sm.print_parameter_dict(param_dict, fields)

if __name__ == "__main__":
    sys.exit(main())
    os.system("echo '" + lines + " ' >> " + scriptname)

if __name__ == "__main__":
    sys.exit(main()) 
