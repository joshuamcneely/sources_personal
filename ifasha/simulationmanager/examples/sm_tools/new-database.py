#!/usr/bin/env python

# example-new-sim-database.py
#
# Code to manage simulations input and outputs
#
# There is no warranty for this code
#
# @version 0.1
# @author Gabriele Albertini <galberti819@gmail.com>
# @date     2016/03/04
# @modified 2016/03/04

from ifasha.simulationmanager import StaticSimulation
from ifasha.simulationmanager import DynamicSimulation
import ifasha.simulationmanager.examples.sm_info as sm_info

import os


project_name = 'test_1'
memory = False
wdir = sm_info.wdir
source_dir = sm_info.source_dir
    


static_sm = StaticSimulation(project_name, wdir, source_dir, memory)
if not memory:
    static_sm.__del__()
    static_sm = StaticSimulation(project_name, wdir, source_dir, memory)

print static_sm.database_name
print static_sm.input_fld
print static_sm.analysis_fld
print static_sm.opt_fld
print static_sm.info_fld

dynamic_sm = DynamicSimulation(project_name, wdir, source_dir, memory)
if not memory:
    dynamic_sm.__del__()
    dynamic_sm = DynamicSimulation(project_name, wdir, source_dir, memory)

print dynamic_sm.database_name
print dynamic_sm.input_fld
print dynamic_sm.analysis_fld
print dynamic_sm.opt_fld
print dynamic_sm.info_fld


