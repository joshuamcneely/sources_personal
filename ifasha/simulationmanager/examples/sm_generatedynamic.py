#!/usr/bin/env python

from __future__ import print_function
import sm_info as info

#-----------------------------------------------------------------------------------------------------------

file_str = dict()
file_str["input"] = """user parameters [
     # NAMES 
     simulation_name = {database_name}_{id:0>4}
     load_simulation_name = {project_name}_{static_type_name}_{static_id:0>4}        
  
     # FOLDERS
     output_folder       = {out_dir}
     paraview_folder     = {out_dir}
     restart_load_folder = {restart_dir}static_equilibrium/
     restart_dump_folder = {restart_dir}

     # BOUNDARY CONDITIONS
     {pbc}{blktop}top_traction   = [ {taui:1.2e} , -{sigi:1.2e} ]
     {pbc}bot_traction   = [-{taui:1.2e} ,  {sigi:1.2e} ]
     {pbc}left_traction  = [     0 , -{taui:1.2e} ]
     {pbc}right_traction = [     0 ,  {taui:1.2e} ]

     {notpbc}block_x_bcs = {blkx}
     {notpbc}block_y_bcs = {blky}
     {zsymc}block_z_bcs = {zsymb}

     # notch_size     = 
     
     # INTERFACE
     vertical_normals = false 
     pretend_is_in_contact = true
     
     # NUCLEATION
     nuc_half_size = {nuchalfsize:1.6f}
     nuc_speed = {nucspeed:1.3f}
     nuc_center = {nuccenter:1.3f}
     nuc_w_zone = {nucwzone:1.6f}
     # nuc_is_circular = true

     # CENTRAL CRACK SETUP
     central_crack = {ccrack}

     # MESH
     spatial_dimension = {dim}
     mesh              = {mesh_dir}{mesh}
     antisym_setup = {asym}
     
     # TIME
     simulation_time  = 5.5e-4
     time_step_factor = 0.1

     # PARTITION METHOD
     # partition_method = tagging
     # nb_procs = [2,1] 
"""
file_str['options']=""" 
     # DUMP
     {dump_y}

     # DUMP AT A PRECISE TIME
     # precise_dump_time = 6.51586e-05

     # FULL DUMP
     {full}
     full_dump_strain = false
     full_dump_gradu = true

     # GROUP DUMP                           
     model_dump_groups = {dump_groups}

     # FREQUENCY THRESHOLD
     rel_slip_for_highf = 0.1
          
     # INTERFACE TEXT DUMPER 
     interface_dump_highf = 1e-7
     interface_dump_lowf  = 3e-7
     interface_dump_fields = {dumpfields}

     # AT DISTANCE TEXT DUMPER
     at_distance_dump_highf = 1e-7
     at_distance_dump_lowf  = 3e-7

     # GLOBAL TEXT DUMPER
     global_dump_highf = 1e-7
     global_dump_lowf  = 1e-7
     
     # FULL PARAVIEW DUMPER  # value cannot be given in 1e6 format
     nb_max_paraview_dumps = {para_max}
     int_paraview_dumps    = {para_int}

]
"""

file_str['ofhgeom']="""
     # OFF FAULT HETEROGENEITY
     off_fault_het = true
     off_fault_het_x_start_position = {ofhxstart:1.6f}
     off_fault_het_x_trans_position = {ofhxtrans:1.6f}
     off_fault_het_y_start_position = {ofhystart:1.6f}
     off_fault_het_y_trans_position = {ofhytrans:1.6f}

"""

file_str['interfhet']="""
     # INTERFACE HETEROGENEITY
     interface_heterogeneity = false
     heterogeneity_start_position = {hetstart:1.3f}
     heterogeneity_trans_position = {hettrans:1.3f} 
     heterogeneity_mu_s = {hetmus:1.2f}
     heterogeneity_mu_k = {hetmuk:1.2f}
     heterogeneity_d_c  = {hetdc:1.2e}

"""

file_str['materials']="""
material elastic [
     name = slider
     rho  = {rho}
     nu   = {nu:1.2f}
     E    = {E:1.2e}
     {psss_state}
]

material elastic [
     name = base
     rho  = {rho}
     nu   = {nu:1.2f}
     E    = {E:1.2e}
     {psss_state}
     ]

material elastic [
     name = heterogeneity
     rho  = {rho}
     nu   = {nu:1.2f}
     E    = {Ehet:1.2e}
     {psss_state}
]


"""

file_str['lswcoh_new']="""
friction linear_shear_cohesive [
     G_c   = {Gamma:1.6f}
     tau_c = {tauc:1.2f}
     tau_r = {taur:1.2f}
]
"""

file_str['lswcoh']="""
friction linear_cohesive no_regularisation [
     G_c   = {Gamma:1.6f}
     tau_c = {tauc:1.2f}
     tau_r = {taur:1.2f}
]
"""

file_str['lswfric']="""
friction linear_slip_weakening_no_healing [
     mu_s = {MUS:1.2f}
     mu_k = {MUK:1.2f}
     d_c  = {DC:1.2e}
]
"""

file_str["sub"] = """#!/bin/sh
#PBS -N {database_name}_{id:0>4}
#PBS -S /bin/sh
#PBS -V
#PBS -j oe
#PBS -m bea
#PBS -M {email}
#PBS -l select={nb_nodes}:ncpus=48:mpiprocs={mpippn}
#PBS -l walltime={hrs}:00:00

# Load modules for Ada HPC
source /gpfs01/home/pmyjm22/uguca_project/source/uguca/module-list.sh

cd $PBS_O_WORKDIR

./status_time_update.py {database_name} {id} {run} 

mpirun -np {mpip} ./{executable} {input_file} 2>&1 | tee {database_name}_{id:0>4}.o$PBS_JOBID.progress

./status_time_update.py {database_name} {id} {fin} 


"""

file_str["add_dynamic_launch"] = """
./launch.py {database_name} {id}
"""
# ---------------------------------------------------------------------

def generate_dynamic(dynamic_sm, sim_id):
    
    # fetch parameter dictionary
    pdict = dynamic_sm.get_global_parameter_dict(sim_id)

    # complete parameter not in the database
    pdict['database_name'] = dynamic_sm.database_name
    pdict['project_name'] = dynamic_sm.project_name

    pdict['out_dir']     = info.output_dir
    pdict['restart_dir'] = info.restart_dir
    pdict['mesh_dir']    = info.mesh_dir 
    pdict['source_dir']  = dynamic_sm.source_dir # where database is located
    pdict['wdir']        = dynamic_sm.wdir # program is executed

    if pdict['dump_heights'] not in ['none',None,False,'false','']:
        pdict['dump_y']="""
     dump_heights          = [{dump_heights}]
     dump_height_tolerance = 5e-8  """.format(**pdict)
    else:
        pdict['dump_y']="""
     # dump_heights          = []
     # dump_height_tolerance = 5e-8  """.format(**pdict)

    pbc = False # periodic BC
    zsym = True # symmetry in z axis
    

    # MESH
    if pdict['dim']==2:
        pdict['blkx'] = "slider_top,slider_left,slider_right"
        pdict['blky'] = "slider_top,slider_left,slider_right"
        if pdict['asym']=='true':
            pdict['mesh'] = 'block_2d_L{lgth}_Ht{hgth}_msh{msh:1.0f}_quad4.msh'.format(**pdict)
        elif pdict['asym']=='false':
            pdict['mesh'] = 'block_2d_L{lgth}_Ht{hgth}_Hb{hgth}_msh{msh:1.0f}_quad4.msh'.format(**pdict)
      
    elif pdict['dim']==3:
        pdict['zsymb']='slider_back'
        pdict['blkx'] = "slider_top,slider_left,slider_right"
        pdict['blky'] = "slider_top,slider_left,slider_right"
        if pdict['asym']=='true':
            pdict['mesh'] = 'block_3d_L{lgth}_Ht{hgth}_Wt{wdth}_msh{msh:1.0f}_hexa8.msh'.format(**pdict)
        elif pdict['asym']=='false':
            pdict['mesh'] = 'block_3d_L{lgth}_Ht{hgth}_Wt{wdth}_Hb{hgth}_Wb{wdth}_msh{msh:1.0f}_hexa8.msh'.format(**pdict)
            pdict['zsymb'] = 'slider_back,base_back'

    # Material PSSS
    if pdict['dim']==2:
        pdict['psss_state'] = 'Plane_Stress = {psss}'.format(**pdict)
    elif pdict['dim']==3:
        pdict['psss_state'] = ''

    if pbc: # periodic boundary conditions in x direction
        pdict['lpbc'] = 'pbc'
        pdict['pbc'] = '#'
        pdict['notpbc'] = ''
    else:
        pdict['lpbc'] = ''
        pdict['pbc'] = ''
        pdict['notpbc'] = '#'

    if pdict['blocktop']=='true':
        pdict['blktop'] = '#'
        pdict['notpbc']=''
        pdict['blkx']='slider_top'
        pdict['blky']='slider_top'
    else:
        pdict['blktop'] = ''

    if zsym and pdict['dim']==3: # symmetric w.r.t. to z
        pdict['zsym'] = 'zsym'
        pdict['zsymc'] = ''
    else:
        pdict['zsym'] = ''
        pdict['zsymc'] = '#'
        pdict['zsymb'] = ''

    pdict['mpippn'] = 48
    pdict['mpip'] = pdict['nb_nodes'] * pdict['mpippn']

    #--------------------------------------

    pdict['run']       = dynamic_sm.status.running
    pdict['fin']       = dynamic_sm.status.finished
    pdict['postp']     = dynamic_sm.status.postprocessed 
    pdict['static_type_name'] = dynamic_sm.type.static
    
    if "binary_shear_info" in pdict.keys():
        new_ntn_package = True
    else:
        new_ntn_package = False

    dump_options_new = ["binary_shear_info", "binary_normal_info", "cohesive_normal", "cohesive_shear", "strength", "shear_gap", "absolute_shear_gap", "normal_gap", "shear_gap_rate", "normal_gap_rate",'velocity']

    dump_options = ["is_in_contact", "contact_pressure", "is_sticking", "frictional_strength", "friction_traction", "slip", "cumulative_slip", "slip_velocity","velocity"]

    
    pdict['dump_groups']='slider'
    if pdict['asym']=='false':
        pdict['dump_groups']='slider,base'
    elif pdict['ofh']=='true':
        pdict['dump_groups']='slider,heterogeneity'

    if new_ntn_package:
        pdict['executable']="ffc_mixed_mode_fracture"
        dumpfields = [o for o in dump_options_new if o in pdict and int(pdict[o])==1]
    else:
        pdict['executable']="ffc_mode_ii_fracture"
        dumpfields = [o for o in dump_options     if o in pdict and int(pdict[o])==1]

    dpf = ','.join(dumpfields)
    pdict['dumpfields'] = dpf

    if pdict['full_field']==1:
        pdict['full'] = "full_dump_positions = [ 0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08, 0.09,  0.1 ,  0.11,  0.12,  0.13,  0.14,  0.15,  0.16,  0.17, 0.18,  0.19,  0.2 ,  0.21,  0.22,  0.23,  0.24,  0.25,  0.26, 0.27,  0.28,  0.29,  0.3 ,  0.31,  0.32,  0.33,  0.34,  0.35, 0.36,  0.37,  0.38,  0.39, 0.40]"
    else:
        pdict['full'] = "# full_dump_positions = [ 0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08, 0.09,  0.1 ,  0.11,  0.12,  0.13,  0.14,  0.15,  0.16,  0.17, 0.18,  0.19,  0.2 ,  0.21,  0.22,  0.23,  0.24,  0.25,  0.26, 0.27,  0.28,  0.29,  0.3 ,  0.31,  0.32,  0.33,  0.34,  0.35, 0.36,  0.37,  0.38,  0.39, 0.40]"

    sub_file    = "{0}{1}_{2:0>4}.sub".format(dynamic_sm.wdir, dynamic_sm.database_name, sim_id)
    input_file  = "{0}{1}_{2:0>4}.in".format(dynamic_sm.wdir, dynamic_sm.database_name, sim_id)
    pdict['input_file'] = "{0}_{1:0>4}.in".format(dynamic_sm.database_name, sim_id)
    static_sub_file = "{0}{project_name}_sts_{static_id:0>4}.sub".format(dynamic_sm.wdir, **pdict)

    # write files
    with open(input_file, "w") as inp:
        print(file_str["input"].format(**pdict),file=inp)
    with open(input_file, "a") as inp:
        if pdict['ofh']=='true':
            print(file_str['ofhgeom'].format(**pdict),file=inp)
        #if pdict['intfh']=='true':
        #   print(file_str['intfhgeom'].format(**pdict),file=inp)
        print(file_str["options"].format(**pdict),file=inp)
        if pdict['interface']=='lswfric':
            print(file_str['lswfric'].format(**pdict),file=inp)
        elif pdict['interface']=='lswcoh':
            if new_ntn_package:
                print(file_str['lswcoh_new'].format(**pdict),file=inp)
            else:
                print(file_str['lswcoh'].format(**pdict),file=inp)
        print(file_str["materials"].format(**pdict),file=inp)
    with open(sub_file, "w") as sub:
        print(file_str["sub"].format(**pdict),file=sub)
    # with open(static_sub_file, "a") as static_sub:
    #     print(file_str["add_dynamic_launch"].format(**pdict), file=static_sub)

    dynamic_sm.update_status(sim_id, dynamic_sm.status.generated)
    
    print('Dynamic input generated sim_id {0:0>4}'.format(sim_id))