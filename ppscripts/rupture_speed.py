#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt

from ifasha.rupturehunter import RuptureHunter
import solidmechanics as sm
import solidmechanics.lefm as lefm

import ppscripts.postprocess as pp
from ppscripts.input_parameters import get_wave_speed
from ppscripts.save_ruptures import load_ruptures
#from save_supershear_theory import load_supershear_theory

# ------------------------------------------------------------------------------
def rupture_speed(snames, **kwargs):

    wdir = kwargs.get('wdir',"./data")
    group = kwargs.get('group','interface')
    fname = kwargs.get('rpt_fname','ruptures.txt')
    #sst_fname = kwargs.get('sst_fname','supershear_theory.txt')

    if 'avg_dist' not in kwargs:
        kwargs['avg_dist'] = 0.005

    # options are: 'a0' (supershear) or 'lc' (Griffith's length)
    norm_length = kwargs.get('norm_length','one')

    # options are: 'cp', 'cs', 'cR'
    norm_speed = kwargs.get('norm_speed','one')

    # plot theory
    plot_theory = kwargs.get('plot_theory',False)
        
    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    # x-axis: default: crack length
    plot_vs = kwargs.get('plot_vs','l')
    
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

        c0 = 1.
        l0 = 1.
        if not norm_speed == 'one' or not norm_length == 'one' or plot_theory:
            # get information about simulation
            code = pp.get_code_name(sname)
            input_data = pp.get_input_data(sname,**kwargs)

            # value is given
            if type(norm_speed) == float or type(norm_speed) == int:
                c0 = norm_speed
            # value is a wave speed
            else:
                c0 = get_wave_speed(sname,norm_speed,**kwargs)

            materials = get_material_properties(sname, **kwargs)
            # find the material of interest
            for matname in materials:
                if matname in norm_speed:
                    mat = materials[matname]

            ax.axhline(y=1., ls='--', color='0.75')

                
            if not norm_length == 'one' or plot_theory:
                if code == 'weak-interface':
                    Gamma = float(input_data['Gc'])
                    tau_c = float(input_data['tau_c']) - float(input_data['tau_c_res'])
                    tau_0 = float(input_data['shear_load']) - float(input_data['tau_c_res'])

                elif code == 'akantu':
                    int_dict = input_data['friction-0']
                    Gamma = float(int_dict['G_c'])
                    tau_c = float(int_dict['tau_c']) - float(int_dict['tau_r'])
                    tau_0 = 0 # <---------------------------------------------------

                if norm_length == 'lc':
                    l0 = lefm.compute_critHalfLength(Gamma, tau_0, material)
                elif norm_length == 'a0':
                    l0 = Gamma * material[sm.smd.mu] / tau_c**2

        # -------------------------
        rpts = load_ruptures(sname,**kwargs)
        for rpt in rpts:
            if 'prop_dir' in kwargs:
                rpt = rpt.first()
                front = rpt.get_sorted_front()
                (xnuc,znuc), tnuc = rpt.get_nucleation()
            else:
                xnuc = 0.

            prop_speed = rpt.get_propagation_speed(0,**kwargs)
            if len(prop_speed):
                if plot_vs == 'l':
                    ax.plot((prop_speed[:,0] - xnuc)/l0, prop_speed[:,2]/c0,
                            '.-',label=sname)
                    ax.set_xlabel(r'crack length l')
                elif plot_vs == 't':
                    ax.plot(prop_speed[:,1], prop_speed[:,2]/c0,
                            '.-',label=sname)
                    ax.set_xlabel(r'time t')
                else:
                    raise RuntimeError("Don't know 'plot_vs' = "+plot_vs)


            if plot_theory and plot_vs == 'l':
                print(material)
                kwargs['uni_tau0']=tau_0
                kwargs['uni_Gamma']=Gamma
                local_l = (prop_speed[:,0]-xnuc)/l0
                eom = lefm.subRayleigh_equation_of_motion(local_l,
                                                          material,
                                                          **kwargs)
                ax.plot(eom[:,0], eom[:,1]/c0, 'k-')


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

        
        ax.set_ylabel(r'propagation speed')

    try:
        ax.legend(loc='best')
    except:
        pass



# ------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.exit('Missing argument! usage: ./rupture_speed.py simid[s]')

    # simid can be 22,24,30-33 -> [22,24,30,31,32,33]
    simids = pp.string_to_simidlist(str(sys.argv[1]))

    if (len(sys.argv) ==3):
    	d_slip = sys.argv[2]
    	rupture_speed(simids,d_slip=d_slip)
    else:
        rupture_speed(simids)

    plt.show()
