#!/usr/bin/env python
#
# nucleation_hunter.py
#
# Code to find rupture nucleation
# There is no warranty for this code
#
# @version 1.0
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     2016/05/19
# @modified 2016/05/19

from __future__ import print_function, division, absolute_import

import sys
import numpy as np
from timeit import default_timer as timer

import solidmechanics as sm


class NucleationHunter():
    """Class used to identify the nucleation of daughter crack

    """
    def __init__(self):
        """Initiates an object

        """
        self.evolution = {
            0: 'main_crack',
            1: 'front_0',
            2: 'daughter_crack',
            3: 'front_1',
            'main_crack'      :0,
            'front_0'      :1,
            'daughter_crack'  :2,
            'front_1'      :3,}

    def load(self, position, time, is_sticking):
        """Function used to load input from the simulation data

        Args:
            position (numpy.memmap):   get data using get_field_at_t_index at idxs = 0
            time (numpy.memmap):       get data using get_full_field and the time FieldId
            is_sticking(numpy.memmap): get data using get_full_field and the is_sticking FieldId

        """
        self.time = time
        soi = np.argsort(position, axis=1)
        sti = np.indices(position.shape)
        is_sticking = is_sticking[sti[0],soi]
        position = position[sti[0],soi]

        self.is_sticking = is_sticking
        self.position = position
        
        self.is_supershear = False
        self.estimate_is_supershear = False
        self.nuc_t_idx = 0
        self.nuc_p_idx = 0

    def estimate_transition(self, prop_speed, input_param):
        """Function used to compute a row estimate of the transition position, based of the velocity

        Args:
            prop_speed (numpy.array): the array of propagation speed
            input_param (dict): dictionary for solid mechanics definitions

        """    
        print(' - ESTIMATE TRANSITION POSITION')
        print(prop_speed.shape)
        for t_idx in range(len(prop_speed[:, 2])):
            if prop_speed[t_idx, 2] > input_param[sm.smd.cR]:
                transition_pos = prop_speed[t_idx,0]
                self.nuc_t_idx = int(t_idx)
                self.nuc_p_idx = int(transition_pos / input_param['lgth'] * input_param['msh'])
                self.estimate_is_supershear = True
                print('   transition position estimate:')
                print('   position   =', transition_pos)
                print('   time       =', self.time[t_idx])
                break
            else:
                self.nuc_t_idx = 0
                self.nuc_p_idx = 0
                self.estimate_is_supershear = False


    def find_p_idx_main_crack(self, t_idx):
        """Function used to find position index at a certain time to be sure to begin search in open crack region

        Args:
            t_idx (int): time step

        Returns:
            int: position index of the main crack at a given time

        """
        for p_idx in range(0, len(self.position[0])-1):
            if self.is_sticking[t_idx, p_idx] == True:
                print(self.evolution[1],' @ p_idx = ', p_idx)
                p_idx_main_crack = p_idx - 1
                break
        return p_idx_main_crack


    def find_daughter_crack_nuc(self, optimise=False):
        """Function used to find the exact transition position, defined as the point where the daughter_crack nucleates

        Args:
            optimise (bool): whether or not to use an optimise algorithm with time step = 10

        """
        print(' - FIND DAUGHTER CRACK')
        crack_space_evolution = 0
        
        start = timer()

        # start at some time previous the estimate transition
        t_idx_start = int(max(0, self.nuc_t_idx * 0.9))
        t_idx_end = len(self.time)
            
        p_idx_start = 0
        p_idx_end = len(self.position[0])

        print('   start time     =', self.time[t_idx_start])
        print('   end time       =', self.time[t_idx_end-1])
        print('   start position =', self.position[0, p_idx_start])
        print('   end position   =', self.position[0, p_idx_end-1])

        t_idx = t_idx_start
        
        nbp = len(str(t_idx_end))

        if optimise:
            t_step = 10
        else:
            t_step = 1   

        for t_idx in range (t_idx_start, t_idx_end, t_step):
            print('     * time {1:{0}d}/{2:{0}d}'.format(nbp, t_idx+1, t_idx_end), end='\r')
            sys.stdout.flush()

            if self.is_sticking[t_idx, p_idx_start] == False:
                crack_space_evolution = 0
            else:
                print('\t\t\tWARNING: start time very low', end='\r')
                sys.stdout.flush()
                continue

            for p_idx in range(p_idx_start, p_idx_end-1):
                # print(p_idx,self.evolution[crack_space_evolution])
                if self.is_sticking[t_idx, p_idx] != self.is_sticking[t_idx, p_idx +1]:
                    crack_space_evolution += 1
                    if crack_space_evolution == self.evolution['front_0']:
                        new_p_idx_start = p_idx
                    if crack_space_evolution == self.evolution['daughter_crack']:
                        print('\n\t{1:<12} @ position {2:{0}d}/{3:{0}d}'.format(nbp, 
                                                                self.evolution[crack_space_evolution],
                                                                p_idx,
                                                                p_idx_end))
                        sys.stdout.flush()
                        p_idx_daughter_crack_start = p_idx        
                    if crack_space_evolution == self.evolution['front_1']:    
                        p_idx_daughter_crack_end = p_idx
            
            p_idx_start = min(new_p_idx_start, p_idx_end-2)

            if crack_space_evolution == self.evolution['front_1']:
                if not optimise:
                    if p_idx_daughter_crack_start > 0.9 * p_idx_end:
                        print('WARNING: result might be affected by boundary condition')
                        self.is_supershear = False
                        break
                    else:
                        self.is_supershear = True
                        self.nuc_p_idx = p_idx_daughter_crack_start
                        print('\tSUPERSHEAR TRANSITION!')
                        print('\tdaughter crack length [elements] =',(p_idx_daughter_crack_end - p_idx_daughter_crack_start)) 
                        print('\t                      [mm]       =',self.position[0,p_idx_daughter_crack_end - p_idx_daughter_crack_start])
                        print('\tposition   =', self.position[0,self.nuc_p_idx])
                        print('\ttime       =', self.time[t_idx])
                        break   
                else:
                    self.nuc_p_idx = (p_idx_daughter_crack_end + p_idx_daughter_crack_start)/2
                    self.nuc_t_idx = t_idx
                    self.find_daughter_crack_nuc(False)
                    break


        print('\n\tit took:', timer()-start)
        if crack_space_evolution == self.evolution['front_0']:
            print('\tno supershear transition')
            self.is_supershear = False
            self.nuc_t_idx = 0
            self.nuc_p_idx = 0

    def get_transition_position(self):
        """Function used to get the transition position where the daughter crack nucleates

        Returns:
            int: position index (if supershear exists, OR)
            bool: False (no supershear at all)
        """
        if self.is_supershear:
            return self.position[0, self.nuc_p_idx]
        else:
            return False
