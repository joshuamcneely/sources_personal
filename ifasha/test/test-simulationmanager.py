#!/usr/bin/env python

# test-simulationmanager.py
#
# Code to manage simulations input and outputs
#
# There is no warranty for this code
#
# @version 0.1
# @author Gabriele Albertini <galberti819@gmail.com>
# @date     2016/03/04
# @modified 2016/03/04

from ifasha.simulationmanager.examples.sm_staticsimulation import StaticSimulation
from ifasha.simulationmanager.examples.sm_dynamicsimulation import DynamicSimulation
import numpy as np
from copy import deepcopy
import ifasha.solidmechanics.lefm as lefm
import ifasha.solidmechanics.definitions as smd

import os

project_name = 'test1'
source_dir = './'
wdir = './'
print_out = False
beta = 1.0
alpha =1.0

# insert input
input_parameters = {
# geometry
    'dim':         3,#
    'lgth':        0.2*beta,
    'hgth':        0.1*beta,
    'wdth':        0.0025,
    'msh':         3200*alpha,
    'asym':        'true',#'false',#'true', <-------------------------------------
    'ccrack':      'false',# <-------------------------------------
# off fault heterogeneity geometry
    'ofh':         'false',#,'true',  
    'ofhxstart':   0.0, # 3 float precision 1.3f
    'ofhxtrans':   0.0, 
    'ofhystart':   0.0,
    'ofhytrans':   0.0,
    'Ehet':        5.65e9 * 1.0,
# material properties
    smd.E:         5.65e9,
    smd.nu:        0.33,
    smd.rho:       1160,
    'psss':           1, # 1 plane stress, 0 plane strain # <---------------------------
# loading
    'sigi':        5.0e6,
    'taui':        0.3e6,
# interface properties
    'interface':   'lswcoh',
    'Gamma':       1.12,
    'tauc':        .8e6,
    'taur':        0.,
    #'interface': 'lswfric',
    'mus':         0,
    'muk':         0,
    'dc':          0,
# nucleation
    'nuchalfsize': 0.050133,
    'nuccenter':   0.0,
    'nucspeed':    124.369,#1400,#
    'nucwzone':    0.005591,
    }

# options
static_options = {
# computation time
    'auto': 1    }

dynamic_options = {
# computation time
    'hrs':         24,#9 3
    'nb_nodes':        4,
# paraview dump
    'para_max':        0, #10
    'para_int':    10000,
# output
    'dump_heights':        '0.0,0.0035,0.0075',
    'is_in_contact':       0,
    'contact_pressure':    1,
    'is_sticking':         1, # essential #######################################################################################
    'friction_traction':   1,
    'frictional_strength': 0,
    'slip':                0,
    'cumulative_slip':     0,
    'slip_velocity':       0,
    'full_field':          0, # <--------------
    }


def compute_mus(S, setup_prop):
    muk = float(setup_prop['muk'])
    sigi = float(setup_prop['sigi'])
    taui = float(setup_prop['taui'])

    taur = sigi * muk
    tauc = taui * (1.+S) - S * taur
    mus = tauc/sigi
    return mus

def compute_tauc(S, setup_prop):
    taui = float(setup_prop['taui'])
    taur = float(setup_prop['taur'])
    tauc = taui * (1.+S) - S * taur
    return tauc

def compute_dc(Lc_dx, setup_prop):
    muk = float(setup_prop['muk'])
    mus = float(setup_prop['mus'])
    sigi = float(setup_prop['sigi'])
    taui = float(setup_prop['taui'])
    tauc = mus * sigi
    taur = muk * sigi   
    dx = float(setup_prop['lgth']) / float(setup_prop['msh'])
    E = float(setup_prop[smd.E])
    nu = float(setup_prop[smd.nu])

    # for central crack
    if setup_prop['ccrack'] == 'true' and setup_prop['psss'] == 0: #plane strain
        dc = Lc_dx * dx / ( E / ((1. + nu)*(1. - nu)) * 0.5 * (tauc - taur) / (np.pi * (taui - taur)**2 )) 
        setup_prop['dc']=dc
        return dc
    elif setup_prop['ccrack'] == 'true' and setup_prop['psss'] == 1: #plane stress
        dc = Lc_dx * dx / ( E * 0.5 * (tauc - taur) / (np.pi * (taui - taur)**2 ))
        setup_prop['dc']=dc
        return dc 
    else:
        raise RuntimeError

def compute_Gamma(Lc_dx, setup_prop):
    taui = float(setup_prop['taui'])
    tauc = float(setup_prop['tauc'])
    taur = float(setup_prop['taur']) 
    dx = float(setup_prop['lgth']) / float(setup_prop['msh'])
    E = float(setup_prop[smd.E])
    nu = float(setup_prop[smd.nu])
    
    # for central crack
    
    if setup_prop['psss'] == 0: #plane strain
        Gamma = Lc_dx * dx / ( E / ((1. + nu)*(1. - nu)) / (np.pi * (taui - taur)**2 )) 
    else:#plane stress
        Gamma = Lc_dx * dx / ( E  / (np.pi * (taui - taur)**2 )) 
            
    if setup_prop['ccrack'] == 'false':
        Gamma /= 1.12
        
    setup_prop['Gamma'] = Gamma
    return Gamma

def compute_Ehet(Eratio, setup_prop):
    Ehet = float(setup_prop[smd.E]) * Eratio
    return Ehet

def compute_Lc(setup_prop):
    taui = float(setup_prop[smd.taui])
    if setup_prop['interface']=='lswfric':
        taur = float(setup_prop[smd.muk])*float(setup_prop[smd.sigi])
        Gamma = lefm.compute_Gamma(setup_prop)
    else:
        taur = float(setup_prop[smd.taur])
        Gamma = float(setup_prop[smd.Gamma])
    delta_tau = taui - taur
    Lc = lefm.compute_critHalfLength(Gamma, delta_tau, setup_prop)
    if setup_prop['ccrack'] == 'true':
        return Lc
    else:
        return Lc*1.12

def compute_heterogeneity_size(ofhd_Lc, ofht_Lc, ofhs_Lc, ofhl_Lc, setup_prop):
    Lc = compute_Lc(setup_prop)

    setup_prop['ofhystart'] = ofhd_Lc * Lc
    setup_prop['ofhytrans'] = (ofhd_Lc + ofht_Lc) * Lc
    setup_prop['ofhxstart'] = ofhs_Lc * Lc
    setup_prop['ofhxtrans'] = (ofhs_Lc + ofhl_Lc) * Lc
    
def check_dict(dict1, dict2):
    """check if 2 dictionary are identical and prints differences
    """
    fields1 = dict1.keys()
    fields2 = dict2.keys()
    for f in fields2:
        if f not in fields1:
            dict1[f] = '-'
    for f in fields1:
        if f not in fields2:
            dict2[f] = '-'
    for fld in dict1.keys():
        if fld not in ['Gamma', 'static_id']:
            if dict1[fld] != dict2[fld] and dict1[fld] != '-' and dict2[fld] != '-':
                print('\nERROR: difference in field: {0:<24}{1:<20}{2:<20}\n'.format(fld, dict1[fld], dict2[fld]))
                raise RuntimeError

#------------------------------------------------------------------------------------

def main():
    # begin with a virgin database
    try:
        #os.remove('{}{}*'.format(source_dir, project_name))    
        os.remove('{}{}_sts.sqlite'.format(source_dir, project_name))    
        os.remove('{}{}_dyn.sqlite'.format(source_dir, project_name))
        os.remove('{}{}_sts*'.format(source_dir, project_name))    
        os.remove('{}{}_dyn*'.format(source_dir, project_name))
    except:
        print('Could not remove database')
        pass
        
    static_sm = StaticSimulation(project_name, wdir, source_dir)
    static_sm.__del__()

    dynamic_sm = DynamicSimulation(project_name, wdir, source_dir)
    dynamic_sm.__del__()
    #-------------------------------------------------------------------------
   
    # load database files
    static_sm = StaticSimulation(project_name, wdir, source_dir)
    dynamic_sm = DynamicSimulation(project_name, wdir, source_dir)


    # 0
    print('\nTEST 0 -----------------------------------------------------------------------------------------------------')
    print('\nCreates first simulation, generated dynamic without static')

    input_parameters_init = deepcopy(input_parameters)

    static_id = static_sm.new_simulation(input_parameters, static_options)
    dynamic_id = dynamic_sm.new_simulation(input_parameters, dynamic_options)

    check_dict(input_parameters_init, input_parameters)

    if static_sm.test == [True,True,True] and dynamic_sm.test == [True,True,True]: 
        print('\nTEST 0 insert OK\n')
    else:
        print(static_sm.test)
        print(dynamic_sm.test)
        raise RuntimeError

    # generate dynamic and test for missing generate static

    try:
        # print '\nGenerate static'
        if static_sm.check_generate(static_id):
            static_sm.generate(static_id)
            print 'static status = ', static_sm.get_status(static_id)    
    except:
        print '\nERROR: could not generate static '
        print 'static_id ', static_id
        raise RuntimeError

    try:
        # print '\nGenerate dynamic'    
        # print dynamic_sm.get_status(dynamic_id)
        if dynamic_sm.check_generate(dynamic_id):
            dynamic_sm.generate(dynamic_id)
            print 'dynamic status = ' ,  dynamic_sm.get_status(dynamic_id)
    except:
        print '\nERROR: could not generate dynamic'
        print 'dynamic_id', dynamic_id
        raise RuntimeError

    # 1
    print('\nTEST 1 -----------------------------------------------------------------------------------------------------')
    print('\nCreates simulation for given dimensionless parameters')

    # dimensionless parameter describing the problem
    Eratio = 0.94
    S = 1.0
    Lc_dx = 100. 

    # off fault heterogeneity geometry dimensionless
    ofhd_Lc = 10. # distance / centre of gravity of heterogeneity
    ofht_Lc = 4.  # thickness
    # entire domain here
    ofhs_Lc = 0.  # start from left edge
    ofhl_Lc = input_parameters['msh'] / (Lc_dx) # length

    input_parameters['Lc'] = compute_Lc(input_parameters)
    print '\nbefore modification '
    print 'Lc    ', input_parameters['Lc']
    print 'Lc_dx ', (float(input_parameters['Lc'])/float(input_parameters['lgth']) * float(input_parameters['msh'])) 
    # print 'mus   ', input_parameters['mus']
    print 'Ehet  ', input_parameters['Ehet']
    # print 'dc    ', input_parameters['dc']
    
    # overwrite parameter in order to match dimensionless constrain
    input_parameters['Ehet'] = compute_Ehet(Eratio, input_parameters)
    # input_parameters['mus']  = compute_mus(S, input_parameters)
    input_parameters['tauc'] = compute_tauc(S, input_parameters)
    # input_parameters['dc']   = compute_dc(Lc_dx, input_parameters)
    input_parameters['Gamma']= compute_Gamma(Lc_dx, input_parameters)
  
    input_parameters['Lc']   = compute_Lc(input_parameters)

    print '\n after modification '
    print 'Lc  ', input_parameters['Lc']
    print 'Lc_dx ', (float(input_parameters['Lc'])/float(input_parameters['lgth']) * float(input_parameters['msh'])) 
    # print 'mus ', input_parameters['mus']
    print 'Ehet', input_parameters['Ehet']
    # print 'dc  ', input_parameters['dc']
    
    # check 
    leratio = float(input_parameters['Ehet'])/float(input_parameters['E'])
    if abs(leratio - Eratio) > 1e6:
        raise RuntimeError

    # check
    lefmS = lefm.compute_S(input_parameters)
    if abs(lefmS - S) > 1e-6:
        print('Error in S computation {} != {}'.format(lefmS,S))
        raise RuntimeError

    # check
    # muk = float(input_parameters['muk'])
    # mus = float(input_parameters['mus'])
    sigi = float(input_parameters['sigi'])
    taui = float(input_parameters['taui'])
    # tauc = mus * sigi
    # taur = muk * sigi   
    tauc = float(input_parameters[smd.tauc])
    taur = float(input_parameters[smd.taur])
    dx = float(input_parameters['lgth']) / float(input_parameters['msh'])
    E = float(input_parameters[smd.E])
    nu = float(input_parameters[smd.nu])
    
    
    Gamma = float(input_parameters['Gamma'])
    
    
    dx = float(input_parameters['lgth']) / float(input_parameters['msh'])
    E = float(input_parameters['E'])
    nu = float(input_parameters['nu'])
    

    myLc = (E / ((1. + nu)*(1. - nu)) * Gamma / (np.pi * (taui - taur)**2 )) 
    if input_parameters['ccrack']=='false':
        myLc *= 1.12
    lefmLc = compute_Lc(input_parameters)
    if abs((lefmLc / dx ) - Lc_dx) > 1e-6:
        print('myLc  ', myLc)
        print('lefmLc', lefmLc)
        print('Error in Lc computation {} != {}'.format(lefmLc/dx,Lc_dx))
        raise RuntimeError
    

    compute_heterogeneity_size(ofhd_Lc, ofht_Lc, ofhs_Lc, ofhl_Lc, input_parameters) 
    
    static_id = static_sm.new_simulation(input_parameters, static_options, print_out)

    dynamic_id = dynamic_sm.new_simulation(input_parameters, dynamic_options)
    dynamic_sm.print_table(dynamic_sm.table.analysis, dynamic_id)

    # integrity_test(dynamic_sm, dynamic_sm.table.analysis)
    test_parameters = dynamic_sm.get_parameter_dict(dynamic_id, dynamic_sm.table.analysis)
    test = [
        test_parameters['Eratio'] - Eratio, 
        test_parameters['S'] - S,
        test_parameters['Lc_dx'] - Lc_dx,
        test_parameters['ofhd_Lc'] - ofhd_Lc,
        test_parameters['ofht_Lc'] - ofht_Lc,
        test_parameters['ofhs_Lc'] - ofhs_Lc,
        test_parameters['ofhl_Lc'] - ofhl_Lc,
        ]

    for t in test:
        if abs(t) > 1e-16:
            print(t)
            raise RuntimeError


    # 2
    print('\nTEST 2 -----------------------------------------------------------------------------------------------------')
    print('\nCreates new simulation with same static input')

    input_parameters['mus'] = 0.52
    input_parameters['Gamma'] = 2.
    dynamic_options['is_in_contact'] = 1

    input_parameters_init = deepcopy(input_parameters)

    static_id = static_sm.new_simulation(input_parameters, static_options)
    dynamic_id = dynamic_sm.new_simulation(input_parameters, dynamic_options)

    check_dict(input_parameters_init, input_parameters)


    if static_sm.test == [False,False,False] and dynamic_sm.test == [True,True,True]: 
        print('\nTEST 2 insert OK\n')
    else:
        print(static_sm.test)
        print(dynamic_sm.test)
        raise RuntimeError

    try:
        # print '\nGenerate static'
        if static_sm.check_generate(static_id):
            static_sm.generate(static_id)
            print 'static status = ', static_sm.get_status(static_id)    
    except:
        print '\nERROR: could not generate static '
        print 'static_id ', static_id
        raise RuntimeError

    try:
        # print '\nGenerate dynamic'    
        # print dynamic_sm.get_status(dynamic_id)
        if dynamic_sm.check_generate(dynamic_id):
            dynamic_sm.generate(dynamic_id)
            print 'dynamic status = ' ,  dynamic_sm.get_status(dynamic_id)
    except:
        print '\nERROR: could not generate dynamic'
        print 'dynamic_id', dynamic_id
        raise RuntimeError

    # 3
    print('\nTEST 3 -----------------------------------------------------------------------------------------------------')
    print('\nCreates new simulation with different static and different dynamic input and same options as 2')
    input_parameters['taui'] = 1.1e6

    input_parameters_init = deepcopy(input_parameters)

    static_id = static_sm.new_simulation(input_parameters, static_options)
    dynamic_id = dynamic_sm.new_simulation(input_parameters, dynamic_options)

    check_dict(input_parameters_init, input_parameters)

    if static_sm.test == [True,False,True] and dynamic_sm.test == [True,False,True]: 
        print('\nTEST 3 insert OK\n')
    else:
        print(static_sm.test)
        print(dynamic_sm.test)
        raise RuntimeError


    # 4
    print('\nTEST 4 -----------------------------------------------------------------------------------------------------')
    print('\nCreates new simulation with same static and dynamic input and options')

    input_parameters_init = deepcopy(input_parameters)

    static_id = static_sm.new_simulation(input_parameters, static_options)
    dynamic_id = dynamic_sm.new_simulation(input_parameters, dynamic_options)

    check_dict(input_parameters_init, input_parameters)

    if static_sm.test == [False,False,False] and dynamic_sm.test == [False,False,False]: 
        print('\nTEST 4 insert OK\n')
    else:
        print(static_sm.test)
        print(dynamic_sm.test)
        raise RuntimeError

    # 5
    print('\nTEST 5 -----------------------------------------------------------------------------------------------------')
    print('\nCreates new simulation with same static and dynamic input and different options')

    dynamic_options["slip_velocity"] = 1
    static_options["auto"] = 0

    input_parameters_init = deepcopy(input_parameters)

    static_id = static_sm.new_simulation(input_parameters, static_options)
    dynamic_id = dynamic_sm.new_simulation(input_parameters, dynamic_options)

    check_dict(input_parameters_init, input_parameters)

    if static_sm.test == [False,True,False] and dynamic_sm.test == [False,True,True]: 
        print('\nTEST 5 insert OK\n')
    else:
        print(static_sm.test)
        print(dynamic_sm.test)
        raise RuntimeError

    # delete database file
    try:
        os.remove('{}{}_sts.sqlite'.format(source_dir, project_name))    
        os.remove('{}{}_dyn.sqlite'.format(source_dir, project_name))
        os.remove('{}{}_sts*'.format(source_dir, project_name))    
        os.remove('{}{}_dyn*'.format(source_dir, project_name))
    except:
        pass
        
if __name__ == "__main__":
    main()
