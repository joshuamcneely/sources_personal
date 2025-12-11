#!/usr/bin/env python
#
# cohesive_zone_hunter.py
#
# Code to find the cohesive zone
# There is no warranty for this code
#
# @version 1.0
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     2016/05/19
# @modified 2020/12/22

from __future__ import print_function, division, absolute_import

import numpy as np

def hunt_cohesive_zone(positions,times,friction_traction,is_sticking):
    #times = time[:,0]
    position = positions[0]
    tau_r = np.min(friction_traction)
    tau_c = np.max(friction_traction)

    tau_0 = friction_traction[0,0]
    epsilon=abs(1e-3*(tau_c-tau_r))

    if tau_r<0:
        print('Warning: tau_r<0')
        print('tau_r,tau_c,epsilon',tau_r,tau_c,epsilon)
        print('set tau_r=0')
        tau_r=0;

    tau_c_global = tau_c

    is_residual = friction_traction<tau_r+epsilon
    
    is_not_coh_zone = is_residual + is_sticking
    is_coh_zone = is_not_coh_zone==False

    return is_coh_zone


def hunt_cohesive_zone_boundary(position,time,friction_traction,is_sticking,direction='right'):

    if direction=='right':
        sign=1.0;
    else:
        sign=-1.0;

    is_coh_zone = hunt_cohesive_zone(position,time,friction_traction,is_sticking)
    
    is_stick_idx = 3.0
    is_coh_idx   = 2.0
    is_res_idx   = 0.0
    
    is_stick_to_is_coh = is_stick_idx-is_coh_idx
    is_coh_to_is_res = is_coh_idx-is_res_idx

    all_info = np.array(is_sticking)*is_stick_idx+np.array(is_coh_zone)*is_coh_idx

    Xc_c_pos = np.array([position[0,np.argmin(np.abs(np.diff(all_info[idx])-is_stick_to_is_coh*sign))] for idx in range(is_coh_zone.shape[0])])
    Xc_c_time = time[-Xc_c_pos.shape[0]:]
    
    Xc_r_pos = np.array([position[0,np.argmin(np.abs(np.diff(all_info[idx])-is_coh_to_is_res*sign))] for idx in range(is_coh_zone.shape[0])])
    Xc_r_time = time[-Xc_c_pos.shape[0]:]

    flt = [any(is_sticking[idx]==False) for idx in range(is_sticking.shape[0])]

    return np.array([Xc_c_pos[flt],Xc_c_time[flt]]), np.array([Xc_r_pos[flt],Xc_r_time[flt]]), 
