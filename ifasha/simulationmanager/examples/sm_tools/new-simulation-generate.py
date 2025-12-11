f#!/usr/bin/env python

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

import numpy as np
from copy import deepcopy

from ifasha.solidmechanics import lefm
import ifasha.solidmechanics.definitions as smd

from my_simulationmanager.sm_dynamicsimulation import DynamicSimulation
from my_simulationmanager.sm_staticsimulation import StaticSimulation
import my_simulationmanager.sm_info

project_name = 'master1'

wdir = sm_info.wdir
source_dir = sm_info.source_dir
    
print_out = False


# insert input
input_parameters = {
# geometry
    "lgth":        0.2,
    "hgth":        0.1,
    "msh":         3200,
    "half":        "top",
    "halfb":       "true",
    "ccrack":      "true",#######################################################################################################
# off fault heterogeneity geometry
    "ofh":         "true", 
    "ofhxstart":   0.0, # 3 float precision 1.3f
    "ofhxtrans":   0.2, 
    "ofhystart":   0.009,
    "ofhytrans":   0.015,
# material properties
    "e":           5.65e9,
    "nu":          0.33,
    "rho":         1180,
    "psss":        0, # 1 plane stress, 0 plane strain ############################################################################
    "ehet":        5.65e9 * 0.6,
# loading
    "sigi":         5e6,
    "taui":         1e6,
# interface properties
    "mus":         0.6,
    "muk":         0.0,
    "dc":          0.,
# nucleation
    "nucd":        2e-4,
    "nucwzr":      0.1,
    "nuccenter":   0.0,
    "notch":       0.
    }

# options
static_options = {
# computation time
    "taskpn":      2,
    "hrs":         0,
    "min":         30,
    "minnodes":    1,
    "maxnodes":    1
    }

dynamic_options = {
# computation time
    "hrs":         3,
    "minnodes":    5,
    "maxnodes":    5, 
# paraview dump
    "para_max":    0, #10
    "para_int":    1000,
# output
    "is_in_contact":       0,
    "contact_pressure":    0,
    "is_sticking":         1, # essential #######################################################################################
    "frictional_strength": 0,
    "friction_traction":   0,
    "slip":                0,
    "cumulative_slip":     0,
    "slip_velocity":       0
    }

def compute_mus(S, setup_prop):
    muk = float(setup_prop['muk'])
    sigi = float(setup_prop['sigi'])
    taui = float(setup_prop['taui'])

    tauk = sigi * muk
    taus = taui * (1.+S) - S * tauk
    mus = taus/sigi
    return mus

def compute_dc(Lc_dx, setup_prop):
    muk = float(setup_prop['muk'])
    mus = float(setup_prop['mus'])
    sigi = float(setup_prop['sigi'])
    taui = float(setup_prop['taui'])
    taus = mus * sigi
    tauk = muk * sigi   
    dx = float(setup_prop['lgth']) / float(setup_prop['msh'])
    E = float(setup_prop['e'])
    nu = float(setup_prop['nu'])

    # for central crack
    if setup_prop['ccrack'] == 'true' and setup_prop['psss'] == 0: #plane strain
        dc = Lc_dx * dx / ( E / ((1. + nu)*(1. - nu)) * 0.5 * (taus - tauk) / (np.pi * (taui - tauk)**2 )) 
        setup_prop['dc']=dc
        return dc
    elif setup_prop['ccrack'] == 'true' and setup_prop['psss'] == 1: #plane stress
        dc = Lc_dx * dx / ( E * 0.5 * (taus - tauk) / (np.pi * (taui - tauk)**2 )) 
        setup_prop['dc']=dc
        return dc 
    else:
        raise RuntimeError

def compute_ehet(Eratio, setup_prop):
    ehet = float(setup_prop['e']) * Eratio
    return ehet

def compute_Lc(setup_prop):
    if setup_prop['ccrack'] == 'true':
        taui = float(setup_prop[smd.taui])
        tauk = float(setup_prop[smd.muk])*float(setup_prop[smd.sigi])
        delta_tau = taui - tauk
        Gamma = lefm.compute_Gamma(setup_prop)
        Lc = lefm.compute_critHalfLength(Gamma, delta_tau, setup_prop)
    else:
        raise RuntimeError
    return Lc

def compute_heterogeneity_size(ofhd_Lc, ofht_Lc, ofhs_Lc, ofhl_Lc, setup_prop):
    Lc = compute_Lc(setup_prop)

    setup_prop['ofhystart'] = ofhd_Lc * Lc
    setup_prop['ofhytrans'] = (ofhd_Lc + ofht_Lc) * Lc
    setup_prop['ofhxstart'] = ofhs_Lc * Lc
    setup_prop['ofhxtrans'] = (ofhs_Lc + ofhl_Lc) * Lc


def main():    
#-------------------------------------------------------------------------    
    static_sm = StaticSimulation(project_name, wdir, source_dir)
    dynamic_sm = DynamicSimulation(project_name, wdir, source_dir)
    

    # input_parameters = deepcopy(init_input_parameters)

    # alpha is mesh definition
    alpha = 1.
    input_parameters['msh'] = 3200 * alpha

    # dimensionless parameter describing the problem
    Eratio = 0.6
    S = 2.0
    Lc_dx = 100. * alpha

    # off fault heterogeneity geometry dimensionless
    input_parameters['ofh'] = 'true'
    ofhd_Lc_s = [1.]#2., 3.,4.,5.]#,2.,3.,4.,5.,6.,7.,8] # distance -begin of heterogeneity
    ofht_Lc_s = [0.1]#,2.,3.,4.,5.,6.]#5.] # thickness
    # entire domain here
    ofhs_Lc = 0.  # start from left edge
    ofhl_Lc = float(input_parameters['msh']) / (Lc_dx) # length


    for ofhd_Lc in ofhd_Lc_s:
        for ofht_Lc in ofht_Lc_s:  
            # overwrite parameter in order to match dimensionless constrain
            input_parameters['ehet'] = compute_ehet(Eratio, input_parameters)
            input_parameters['mus']  = compute_mus(S, input_parameters)
            input_parameters['dc']   = compute_dc(Lc_dx, input_parameters)
            compute_heterogeneity_size(ofhd_Lc, ofht_Lc, ofhs_Lc, ofhl_Lc, input_parameters) 

            #---------------------------------------------------------------------------------------
            # check 
            leratio = float(input_parameters['ehet'])/float(input_parameters['e'])
            if abs(leratio - Eratio) > 1e6:
                raise RuntimeError

            # check
            lefmS = lefm.compute_S(input_parameters)
            if abs(lefmS - S) > 1e-6:
                print('Error in S computation {} != {}'.format(lefmS,S))
                raise RuntimeError
    
            # check
            muk = float(input_parameters['muk'])
            mus = float(input_parameters['mus'])
            sigi = float(input_parameters['sigi'])
            taui = float(input_parameters['taui'])
            taus = mus * sigi
            tauk = muk * sigi   
            dx = float(input_parameters['lgth']) / float(input_parameters['msh'])
            E = float(input_parameters['e'])
            nu = float(input_parameters['nu'])
            dc = float(input_parameters['dc'])

            Gamma =  0.5 * (taus - tauk) * dc
            if input_parameters['psss'] == 0: #plain strain
                print('plain strain!')
                myLc = (E / ((1. + nu)*(1. - nu)) * Gamma / (np.pi * (taui - tauk)**2 )) 
            else: # plain stress
                print('plain stress!')
                myLc = E  * Gamma / (np.pi * (taui - tauk)**2 ) 
            lefmLc = compute_Lc(input_parameters)
            if abs((lefmLc / dx ) - Lc_dx) > 1e-6:
                print 'myLc  ', myLc
                print 'lefmLc', lefmLc
                print('Error in Lc computation {} != {}'.format(lefmLc/dx,Lc_dx))
                raise RuntimeError
    
            # -----------------------------------------------------------------------------------
            # check finished
          
            # inset simulaion in database
            # -----------------------------------------------------------------------------------
            print('INSERT STATIC:')
            static_id = static_sm.new_simulation(input_parameters, static_options, print_out)
    
            print('INSERT DYNAMIC:')
            dynamic_id = dynamic_sm.new_simulation(input_parameters, dynamic_options)
            if dynamic_id != -1: 
                dynamic_sm.print_table(dynamic_sm.table.simulation, dynamic_id)
                print('\n')
                dynamic_sm.print_table(dynamic_sm.table.analysis, dynamic_id)

                # integrity_test(dynamic_sm, dynamic_sm.table.analysis)
                test_parameters = dynamic_sm.get_parameter_dict(dynamic_id, dynamic_sm.table.analysis)
                test = [
                    float(test_parameters['Eratio'])     - Eratio, 
                    float(test_parameters['S'])          - S,
                    float(test_parameters['Lc_dx'])      - Lc_dx,
                    float(test_parameters['ofhd_Lc'])    - ofhd_Lc,
                    float(test_parameters['ofht_Lc'])    - ofht_Lc,
                    float(test_parameters['ofhs_Lc'])    - ofhs_Lc,
                    float(test_parameters['ofhl_Lc'])    - ofhl_Lc,
                    ]
                
                for t in test:
                    if abs(t) > 1e-16:
                        print(t)
                        raise RuntimeError
            
            try:
                if static_sm.check_generate(static_id):
                    static_sm.generate(static_id)
            except:
                print '\nERROR: could not generate static id = ', static_id
                raise RuntimeError

            try:
                if dynamic_sm.check_generate(dynamic_id):
                    dynamic_sm.generate(dynamic_id)
            except:
                print '\nERROR: could not generate dynamic id = ', dynamic_id
                #raise RuntimeError
                
if __name__ == "__main__":
    main()
