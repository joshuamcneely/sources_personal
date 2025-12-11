#!/usr/bin/env python

# example2-new-simulaiton.py
#
# Code to manage simulations input and outputs
#
# There is no warranty for this code
#
# @version 0.1
# @author Gabriele Albertini <galberti819@gmail.com>
# @date     2016/03/04
# @modified 2016/03/04


from my_simulationmanager.sm_dynamicsimulation import DynamicSimulation
from my_simulationmanager.sm_staticsimulation import StaticSimulation
import my_simulationmanager.sm_info


import sys
def usage(msg):
    print msg


wdir = sm_info.wdir
source_dir = sm_info.source_dir
    
print_out = False

#-------------------------------------------------------------------------    
def main(argv=None):
    """Usage:  generate.py <database_name> <sim_id> <-option> 
    -b    # generate related static"""
    
    generate_both = False

    if argv is None:
        args = sys.argv[1:]
    if "-h" in args or "--help" in args:
        usage(main.__doc__)
        sys.exit(2)
    if "-b" in args :
        generate_both = True
    if len(args) < 2:
        usage(main.__doc__)
        sys.exit(2)
    
    database_name = args[0]
    dynamic_id = int(args[1])

    if database_name[-7:] == 'dynamic':
        project_name = database_name[:-8]

        if generate_both == True:
            sm = DynamicSimulation(project_name, wdir, source_dir) 
            static_id = sm.get_id(dynamic_id, 'static_id')
            
            sm = StaticSimulation(project_name, wdir, source_dir)        
            print('Generate static')
            sm.generate(static_id)
            print(sm.get_status(static_id))


        sm = DynamicSimulation(project_name, wdir, source_dir) 
        print('Generate dynamic')
        sm.generate(dynamic_id)

        print(sm.get_status(dynamic_id))



    else:
        project_name = database_name[:-7]
        sm = StaticSimulation(project_name, wdir, source_dir)
        print('Generate static')
        sm.generate(static_id)
        print(sm.get_status(static_id))
        if generate_both == True:
            print 'option only available for DynamicSimulation'
            raise RuntimeError
        

    
if __name__ == "__main__":
    main()
