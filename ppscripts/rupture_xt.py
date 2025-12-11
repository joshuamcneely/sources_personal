#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt

import solidmechanics as sm
import solidmechanics.lefm as lefm
from ifasha.rupturehunter import RuptureHunter

import ppscripts.postprocess as pp
from ppscripts.input_parameters import get_material_properties
from ppscripts.save_ruptures import load_ruptures

# -----------------------------------------------------------------------------
def rupture_xt(snames, **kwargs):

    wdir = kwargs.get('wdir',"./data")
    fname = kwargs.get('rpt_fname','ruptures.txt')
    #sst_fname = kwargs.get('sst_fname','supershear_theory.txt')

    # options are: 'a0' (supershear) or 'lc' (Griffith's length)
    norm_length = kwargs.get('norm_length','one')

    # options are: none
    norm_time = kwargs.get('norm_time','one')

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for sname in snames:

        sname = pp.sname_to_sname(sname)

        # --------------
        # WARNING
        # getting information from input file needs improvement to work with
        # both the weak interface code (which it does already)
        # but also with akantu.

        t0 = 1.
        l0 = 1.
        if not norm_time == 'one' or not norm_length == 'one':
            # get information about simulation
            input_data = pp.get_input_data(sname,**kwargs)
            code = pp.get_code_name(sname, **kwargs)
            if code == 'weak-interface':
                Gamma = float(input_data['Gc'])
                tau_c = float(input_data['tau_c'])-float(input_data['tau_c_res'])
                tau_0 = float(input_data['shear_load']) - float(input_data['tau_c_res'])

            elif code == 'akantu':
                int_dict = input_data['friction-0']
                Gamma = float(int_dict['G_c'])
                tau_c = float(int_dict['tau_c']) - float(int_dict['tau_r'])
                tau_0 = 0 # <---------------------------------------------------

            if not norm_length == 'one':
                materials = get_material_properties(sname, **kwargs)
                if len(materials) == 1:
                    material = materials.items()[0]
                else:
                    raise RuntimeError("Can't choose between materials!")
                
            if norm_length == 'lc':
                l0 = lefm.compute_critHalfLength(Gamma, tau_0, material)
            elif norm_length == 'a0':
                l0 = Gamma * material[sm.smd.mu] / tau_c**2

        # -------------------------
        rpts = load_ruptures(sname,**kwargs)
        #rpts[0].get_sorted_front()

        only_first = kwargs.get('only_first',False)

        for rpt in rpts:
            if only_first:
                rpt = rpt.first()
            front = rpt.get_sorted_front()
            if len(front):
                ax.plot(front[:,0]/l0,front[:,2]/t0,'.-')
            #prop_speed = rpt.get_propagation_speed(0,avg_dist)
            #if len(prop_speed):
            #    ax.plot(prop_speed[:,0]/l0, prop_speed[:,2]/c0, '.-')

        # combine supershear and subRayleigh theory?
        # rupture propagation
        ## try:
        ##     if 'sst_fname' in kwargs:
        ##         eom = load_supershear_theory(sname,sst_fname)
        ##     else:
        ##         eom = load_supershear_theory(sname)
        ## except:
        ##     print('no theory')
        ##     pass
        ## else:
        ##     ax.plot(eom[:,0]/a0,
        ##             eom[:,1]/material[sm.smd.cp],'o')

    if new_figure:
        ax.set_xlabel(r'position')
        ax.set_ylabel(r'time')

        ax.legend(loc='best')



# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.exit('Missing argument! usage: ./rupture_xt.py simids')

    # simid can be 22,24,30-33 -> [22,24,30,31,32,33]
    simids = pp.string_to_simidlist(str(sys.argv[1]))

    rupture_xt(simids)

    plt.show()
