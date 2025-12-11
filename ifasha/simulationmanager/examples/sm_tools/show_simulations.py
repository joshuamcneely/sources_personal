#!/usr/bin/env python

# sh_show_simulations.py
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
import os
import numpy as np

from ifasha.simulationmanager import SimulationManager
import my_simulationmanager.sm_info as sm_info

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def main(*argv):
    """ 
    Usage: show_simulations.py <option> <database_name> <sim_id=None>
                            # shows database relevant info for each simulation
    <sim_id>                # show short info
    -e                      # show essential database
    -t                      # get sim_id
    -c <str 'condition'>    # 
    -s <status>             # 
    -i <sim_id>             # shows complete info
    -supershear             # shows supershear 
    -p <parameter>          # get all e modulus in db not implemented
    """

    wdir = sm_info.wdir
    source_dir = sm_info.source_dir

    if len(argv)==0:
        argv = sys.argv[1:]      
    try:
        try:
            opts, args = getopt.getopt(argv, "ehi:tc:svp:", ["help", "output=", "supershear"])
        except(getopt.error, msg):
             raise Usage(msg)

        # option processing
        verbose = False
        tail = False
        condition = False
        status = False
        supershear = False
        info = False
        essential = False
        time = False

        default= True
        print(opts)
        if len(opts)!=0:
            default=False
        for option, value in opts:
            print(option, value)
            if option == "-v":
                verbose = True
            if option == "-e":
                default=True
                essential = True            
            if option == "-t":
                tail = True
            if option == "-c":
                default=False
                condition = True
                cond_str = value
            if option == "-s":
                status = True
            if option == "-i":
                info= True
                default=False
                sim_id = value
            if option == '-r':
                relevant = True
            if option == "supershear":
                supershear = True
            if option in ("-h", "--help"):
                raise Usage(main.__doc__)
            # if option in ("-o", "--output"):
            #     output = value
        if len(args)<1:
            raise Usage(main.__doc__)   
                  
    except(Usage, err):
        print(sys.argv[0].split("/")[-1] + ": " + str(err.msg))
        print("\t for help use --help")
        return 2


    database_name = args[0].split('.')[0]

    sm = SimulationManager(database_name, wdir, source_dir)    



    db_split = sm.database_name.split('_')
    if db_split[1] == sm.type.static:
        if db_split[0] in ['b0','test0','gdbmm']:
            flds = ['id',  'status', 'comment', u'dim', 'msh','psss','asym','lgth']
        else:
            flds = ['id', 'status', 'comment', u'ofh', 'msh','lgth']
    else: # dynamic
        if db_split[0] in['b0','test0','gdbmm']:
            flds = ['id', 'status','comment', 'static_id','dim', 'psss','lgth', 'msh','nucspeed','nucwzone','wdth','taui','tauc','Gamma']#,'asym'
        elif db_split[0][0] == 'a':
            flds = ['id', 'status','comment', 'supershear',  'static_id', u'cs_ratio', u'S', u'ofh', u'ofhd_Lc', u'ofht_Lc', u'ofhytrans','Xc0_dx', 'hgth','msh','ofhystart']
        else:
            flds = ['id', 'status','comment', 'supershear',  'static_id', u'Eratio', u'S']
        if essential:
            flds = ['id', 'status','comment','static_id','dim','psss','S','msh','Xc0_dx','nucspeed','taui']
            if db_split[0] == 'a2':
                flds = ['id', 'status','comment', 'supershear', u'cs_ratio', u'S', u'ofh',u'ofhd_Lc', u'ofht_Lc', u'ofhytrans','Xc0_dx']

    my_slim_flds = ['asym','S','dim','hrs','nb_nodes','cs_ratio','ofh']

    if default:
        if len(args) == 1:
            # show all sim_id in database
            # print '\nAll simulations in {}:\n'.format(sm.database_name)
            # print sm.get_all_sim_id()

            sm.print_database(flds, column_width=9, max_lines=50,slim_flds=my_slim_flds)
            print('\n')

        if len(args) == 2:
            sim_id = int(args[1])
            print('\n{}_{:0>4}'.format(sm.database_name, sim_id))
            table = sm.table.simulation
            parameter_dict = sm.get_parameter_dict(sim_id, table)
            fields = sm.get_fields(table)
            sm.print_parameter_dict(parameter_dict, fields)

    if time:
        nb_days = int(args[1])
        print(sm.get_sim_id_last_days(nb_days))

    if tail:
        ids = sm.get_sim_ids()
        ids = ids[-49:]
        sm.print_database(flds, sim_ids=ids, column_width=9, max_lines=50,slim_flds=my_slim_flds)

    if condition:
        if False:
            cond_dict={}
            for cond in cond_str.split('&&'):
                key, value = [ string.strip() for string in  cond.split('==')]
                print(key,'==',value)
                cond_dict[key]=value
            ids=sm.get_sim_ids_condition(cond_dict)
        else:
            ids = sm.get_sim_ids_condition(cond_str)
        print(ids)
        sm.print_database(flds, sim_ids=ids, column_width=9, max_lines=50,slim_flds=my_slim_flds)

    if status:
        st = args[1]
        print(st)
        ids = sm.get_all_sim_id()
        for i in ids:
            i = int(i)
            if sm.get_status(i) == st:
                print(i)

    if info:
        try:
            sim_id
        except:
            sim_id = args[1]
        print(sim_id)
        sm.print_all_parameter_dict(sim_id)



        
if __name__ == "__main__":
    sys.exit(main())
    os.system("echo '" + lines + " ' >> " + scriptname)

if __name__ == "__main__":
    sys.exit(main()) 
