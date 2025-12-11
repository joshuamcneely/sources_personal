#!/usr/bin/env python
#
# bump_hunter.py
#
# Code to find rupture nucleation
# There is no warranty for this code
#
# @version 2.0
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     2016/05/19
# @modified 2020/12/22

from __future__ import print_function, division, absolute_import

import sys
import numpy as np
from scipy.signal import argrelextrema


class BumpHunter():
    """Class used to identify the bump ahead of the crack tip

    """
    def __init__(self):
        """Initiates an object

        """
        pass

    def load(self, is_sticking, friction_traction, position, time):
        """Function used to load input from the simulation data

        Args:
            is_sticking(numpy.memmap):        get data using get_full_field and the is_sticking FieldId
            friction_traction (numpy.memmap): get data using get_full_field and the friction_traction FieldId
            position (numpy.memmap):          get data using get_field_at_t_index at idxs = 0
            time (numpy.memmap):              get data using get_full_field and the time FieldId
            is_sticking(numpy.memmap):        get data using get_full_field and the is_sticking FieldId

        """
        if position.shape!=friction_traction.shape:
            position= np.tile(position,(time.shape[0],1))

        soi = np.argsort(position, axis=1)
        sti = np.indices(position.shape)
        is_sticking = is_sticking[sti[0],soi]
        position = position[sti[0],soi]
        friction_traction = friction_traction[sti[0],soi]
            

        self.is_sticking = is_sticking
        self.position = position
        self.friction_traction = friction_traction
        self.time = time

        self.position = position[0]

        self.tip_position  = []
        self.tip_position_idx     = []
        self.bump_position = []
        self.bump_position_idx = []
        self.bump_size     = []
        self.bump_time = []        
        self.bump_time_idx= []

        self.bump = []
        self.bump_idx = []
        self.fields={'front_position':0, 'bump_position':1, 'bump_time':2, 'bump_size':3}

    def hunt(self):
        """Function used to hunt the bump
        """
        print(' - HUNT THE BUMP')
        nbp = len(str(self.time.shape[0]))
        for t_idx in range(self.time.shape[0]):
            
            if all(self.is_sticking[t_idx,:]):
                continue
            else:
                front_idx = np.argmax(np.diff(np.array(self.is_sticking[t_idx,:])*1.0))
            
            # front is at the end of interface
            if front_idx in [self.position.shape[0],0]:
                continue

            # Identify bump positions

            # Identify part of interface right of the crack tip: where crack did not propagate yet
            field_no_slip = self.friction_traction[t_idx,front_idx:]

            # Find local minimum: negative bump before the bump
            neg_bump_idx = argrelextrema(field_no_slip, np.less)[0]
            if not len(neg_bump_idx):
                continue
            neg_bump_idx = neg_bump_idx[0]
            neg_bump_idx += front_idx
            field_after_neg_bump = self.friction_traction[t_idx,neg_bump_idx:]

            # Find the bump position as absolute maximum in the domain after the negative bump
            bump_idx = np.argmax(field_after_neg_bump)
            bump_idx += neg_bump_idx
           
            self.bump_position.append(self.position[bump_idx])
            self.bump_position_idx.append(bump_idx)
            self.bump_size.append(self.friction_traction[t_idx, bump_idx])
            self.tip_position.append(self.position[front_idx])
            self.tip_position_idx.append(front_idx)
            self.bump_time.append(self.time[t_idx])
            self.bump_time_idx.append(t_idx)
            # print('     * time {1:{0}d}/{2:{0}d}   * bump {3:1.3e}'\
            #     .format(nbp, 
            #             t_idx, 
            #             self.time.shape[0],
            #             self.bump_size[-1]), end='\r')
            # sys.stdout.flush()
        print('')

        
    def get_bump(self):
        """Function used to provide the general information of bump

        Returns:
            numpy.array: bump[l,x,t,f], where
            l is the main rupture front position,
            x is the bump position,
            t is the time step,
            f is the bump maximum stress

        """
        self.bump = [self.tip_position, self.bump_position, self.bump_time, self.bump_size]
        return np.array(self.bump)

    def get_bump_idx(self):
        """Function used to provide the index information of bump

        Returns:
            numpy.array: bump_idx[l_idx,x_idx,t_idx], where
            l_idx is the main rupture front position index,
            x_idx is the bump position index,
            t_idx is the time step index

        """
        self.bump_idx = [self.tip_position_idx, self.bump_position_idx, self.bump_time_idx]
        return np.array(self.bump_idx)


    def crop_bump(self, bump, taui,tauc,taur, xmax=np.inf):
        """Function used to cut away (1) initail nonsencse (2) plateau at S (3) beyond valid domain

        Args:
            bump (numpy.array): general information of bump got from get_bump
            taui (float): the initial shear strength
            tauc (float): peak shear strength
            taur (float): residual shear strength
            xmax (float): valid domain limit

        Returns:
            numpy.array: cropped version of bump[l,x,t,f], where
            l_idx is the main rupture front position index,
            x_idx is the bump position index,
            t_idx is the time step index

        """
        cropped_bump = [] 
        for b in range(len(bump[0])):
            if (bump[3,b]-taui)/(taui-taur)<0.5:
                continue # initial nonsense
            if np.abs((tauc-bump[3,b])/tauc)<1e-4:
                break # plateau at S
            if bump[1,b]>xmax:
                break # stuff after valid domain
            cropped_bump.append(bump[:,b])
        return np.transpose(cropped_bump)
            

    def compute_dif_detect_PS(self, norm_bump, norm_ref_bump, dx, *args,**kwargs):
        """Function used to get normalized bump maximum stress differences at namely same front_position comparing to the reference,
        and the normalized bump when P-wave arrives

        Args:
            norm_bump (numpy.array): general information of normalized bump
            norm_ref_bump (numpy.array): general information of normalized reference bump
            dx (float): mesh definition
            **kwargs (dict):    - fix_field='front_position', 'bump_position' default: 'front_position'
                                - PS_arrival=[True,False]
                                - cs_ratio=float
                                - tracking_threshold=1e-3
                                - PS_min=0

        Returns:
            numpy.array: normalized bump maximum stress differences, norm_dif_bump[l,x,t,df], where
                        l is the main rupture front position (normalized),
                        x is the bump position (normalized),
                        t is the time step (normalized),
                        df is the bump maximum stress differences (normalized)
            numpy.array: normalized bump when P-wave arrives, PS_arrival[l,x,t,f], where
                        l is the main rupture front position (normalized),
                        x is the bump position (normalized),
                        t is the time step (normalized),
                        f is the bump maximum stress (normalized)

        Note:
            normalized in this way: [l/lc,x/lc,t*cs/lc,(f-taui)/(taui-taur)]
        """
        fields={'front_position':0, 'bump_position':1, 'bump_time':2, 'bump_size':3}
        fix_field = fields[kwargs.get('fix_field',"front_position")]

        detect_PS = kwargs.get('PS_arrival',False)
        if detect_PS:
            print('DETECT PS ARRIVAL')
            try:
                cs_ratio = kwargs.get('cs_ratio')
            except:
                print('ERROR: specify cs_ratio of simulation')
                raise RuntimeError
            tracking_threshold = kwargs.get('tracking_threshold',1e-3)
            PS_min = kwargs.get('PS_min',0.)
            print('tracking_threshold',tracking_threshold)
            print('PS_min',PS_min)

        PS_arrival = []    
        norm_dif_bump = []
        
        for b in range(len(norm_bump[0])):
            for r in range(len(norm_ref_bump[0])):
                if abs(norm_bump[fix_field,b] - norm_ref_bump[fix_field,r])<2*dx \
                and norm_bump[3,b] > 0.5 \
                and norm_ref_bump[3,r] > 0.5:
                    norm_dif_bump.append([norm_bump[0,b],
                                          norm_bump[1,b],
                                          norm_bump[2,b],
                                          norm_bump[3,b]-norm_ref_bump[3,r],
                                          norm_bump[3,b]])
                                          
                    if detect_PS:
                        if cs_ratio<1:
                            low_high_factor = 1. # Low velocity heterogeneity
                        else:
                            low_high_factor = -1. # High velocity heterogeneity
                        if low_high_factor*((norm_bump[3,b]-norm_ref_bump[3,r])/norm_ref_bump[3,r])>tracking_threshold\
                           and norm_bump[1,b]>PS_min:
                            PS_arrival.append([norm_bump[0,b],
                                               norm_bump[1,b],
                                               norm_bump[2,b],
                                               norm_bump[3,b],
                                               norm_bump[3,b]-norm_ref_bump[3,r]]) # [tip_position, bump_position, bump_time,bump_size,dif_bump_size]
                                                  

        norm_dif_bump = np.transpose(norm_dif_bump)
        if detect_PS: 
            PS_arrival = np.transpose(PS_arrival)
        return norm_dif_bump,PS_arrival
        
        
        
    def detect_PS(self, norm_bump, norm_ref_bump, dx, *args, **kwargs):
        """Function used to detect PS arrival

        Args:
            norm_bump (numpy.array): general information of normalized bump
            norm_ref_bump (numpy.array): general information of normalized reference bump
            dx (float): mesh definition
            **kwargs (dict):    - fix_field='front_position', 'bump_position' default: 'front_position'
                                - PS_arrival=[True,False]
                                - cs_ratio=float
                                - tracking_threshold=1e-3
                                - PS_min=0

        Returns:
            numpy.array: normalized bump when P-wave arrives, PS_arrival[l,x,t,f], where
                        l is the main rupture front position (normalized),
                        x is the bump position (normalized),
                        t is the time step (normalized),
                        f is the bump maximum stress (normalized)
        """
        kwargs['PS_arrival']=True
        norm_dif_bump, PS_arrival = self.compute_dif_detect_PS(norm_bump, norm_ref_bump, dx, *args,**kwargs)
        return PS_arrival

    def detect_SS(self, bump,start_id):
        """Function used to detect SS arrival based on discontinuity on bump position

        Args:
            norm_bump (numpy.array): general information of bump
            start_id (int): time step start index

        Returns:
            numpy.array: normalized bump when S-wave arrives, PS_arrival[l,x,t,f], where
                        l is the main rupture front position
                        x is the bump position
                        t is the time
                        f is the bump maximum stress
        """
        SS_arrival = []
        dx = bump[1,start_id + 1:-1]- bump[1,start_id:-2]
        avdx = np.average(np.abs(dx))
        for t_id in range(start_id,len(bump[2])-1):
            dxb = bump[1,t_id+1]-bump[1,t_id]
            if dxb < - avdx*jump_factor: # detect discontinuity in bump eqmot - corresponds to SS arrival
                SS_arrival.append(bump[:,t_id+1]) # [tip_position, bump_position, bump_time, bump_size]
        
        return np.transpose(SS_arrival)

    def split_PS_SS(self,bump, start_id=0,jump_factor=1.0):
        """
        splits bump before and after SS arrival based on first detect_SS(bump,start_id)
        use cropped bump! 
        """
        SS_arrival = self.detect_SS(bump, start_id,jump_factor)#[0]
        fld = self.fields['front_position']
        if len(SS_arrival):
            ss_id = np.argmin(np.abs(bump[fld]-SS_arrival[fld][0]))
        else:
            ss_id = -1
        before_SS_arrival = bump[:,0:ss_id]
        after_SS_arrival = bump[:,ss_id:-1]
        return [before_SS_arrival, after_SS_arrival]

    def compute_dif_bump(self,norm_bump,norm_ref_bump,dx,*args,**kwargs):
        """Function used to compute dif_bump (normalzed input)

        Args:
            norm_bump (numpy.array): general information of normalized bump
            norm_ref_bump (numpy.array): general information of normalized reference bump
            dx (float): mesh definition
            **kwargs (dict):    - fix_field='front_position', 'bump_position' default: 'front_position'
                                - PS_arrival=[True,False]
                                - cs_ratio=float
                                - tracking_threshold=1e-3
                                - PS_min=0

        Returns:
            numpy.array: normalized bump maximum stress differences, norm_dif_bump[l,x,t,df], where
                        l is the main rupture front position (normalized),
                        x is the bump position (normalized),
                        t is the time step (normalized),
                        df is the bump maximum stress differences (normalized)
        """
        norm_dif_bump, PS_arrival = self.compute_dif_detect_PS(norm_bump, norm_ref_bump, dx, *args,**kwargs)
        return norm_dif_bump


