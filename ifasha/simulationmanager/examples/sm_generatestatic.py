#!/usr/bin/env python

from __future__ import print_function
import sm_info as info

#-----------------------------------------------------------------------------------------

file_str = dict()
file_str["input"] = """user parameters [

     simulation_name = {database_name}_{id:0>4}        
     
     # FOLDERS
     output_folder       = {out_dir}
     paraview_folder     = {out_dir}
     restart_load_folder = {restart_dir}
     restart_dump_folder = {restart_dir}static_equilibrium/

     spatial_dimension = {dim}
     mesh              = {mesh_dir}{mesh}
     antisym_setup     = {asym}

     top_traction   = [ {taui:1.2e} , -{sigi:1.2e} ]
     bot_traction   = [-{taui:1.2e} ,  {sigi:1.2e} ]
     left_traction  = [     0 , -{taui:1.2e} ]
     right_traction = [     0 ,  {taui:1.2e} ]
{dbc}"""

file_str["ofhgeom"]="""
     # OFF FAULT HETEROGENEITY
     off_fault_het = true
     off_fault_het_x_start_position = {ofhxstart:1.6f}
     off_fault_het_x_trans_position = {ofhxtrans:1.6f}
     off_fault_het_y_start_position = {ofhystart:1.6f}
     off_fault_het_y_trans_position = {ofhytrans:1.6f}
"""

file_str["options"]="""
     maximal_iteration = 2
     precision         = 5e-6
]"""

file_str["materials"]="""
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

mpirun -np {mpip} ./{method} {input_file} 2>&1 | tee {database_name}_{id:0>4}.o$PBS_JOBID.progress

./status_time_update.py {database_name} {id} {fin} 


"""

# ---------------------------------------------------------------------

def generate_static(static_sm, sim_id,options=None):
  
  # fetch static parameter dictionary
  # note: complete dict may not be filled yet
  pdict = static_sm.get_global_parameter_dict(sim_id)

  pdict["database_name"] = static_sm.database_name

  pdict["out_dir"]     = info.output_dir
  pdict["restart_dir"] = info.restart_dir
  pdict["mesh_dir"]    = info.mesh_dir 
  pdict["source_dir"]  = static_sm.source_dir # where database is located
  pdict["wdir"]        = static_sm.wdir # program is executed

  pdict["run"]       = static_sm.status.running
  pdict["fin"]       = static_sm.status.finished
  pdict["postp"]     = static_sm.status.postprocessed 
  
  # MESH
  if pdict["dim"]==2:
    pdict["dbc"]="""
     bot_block_y = true 
     block_base_bot_rgh_x = true """
    if pdict["asym"]=="true":
      pdict["mesh"] = "block_2d_L{lgth}_Ht{hgth}_msh{msh:1.0f}_quad4.msh".format(**pdict)
    elif pdict["asym"]=="false":
      pdict["mesh"] = "block_2d_L{lgth}_Ht{hgth}_Hb{hgth}_msh{msh:1.0f}_quad4.msh".format(**pdict)
  elif pdict["dim"]==3:
    pdict["dbc"]="""
     block_x_bcs = slider_bottom_left # ,-separated
     #block_y_bcs = # ,-separated
     block_z_bcs = slider_bottom_back # ,-separated"""
    pdict["zsymb"]="slider_back"
    pdict["blkx"] = "slider_top,slider_left,slider_right"
    pdict["blky"] = "slider_top,slider_left,slider_right"
    if pdict["asym"]=="true":
      pdict["mesh"] = "block_3d_L{lgth}_Ht{hgth}_Wt{wdth}_msh{msh:1.0f}_hexa8.msh".format(**pdict)
    elif pdict["asym"]=="false":
      pdict["mesh"] = "block_3d_L{lgth}_Ht{hgth}_Wt{wdth}_Hb{hgth}_Wb{wdth}_msh{msh:1.0f}_hexa8.msh".format(**pdict)

  # Material PSSS
  if pdict["dim"]==2:
    pdict["psss_state"] = "Plane_Stress = {psss}".format(**pdict)
  elif pdict["dim"]==3:
    pdict["psss_state"] = ""


  #--------------------------------------

  sub_file   = "{0}{1}_{2:0>4}.sub".format(static_sm.wdir, static_sm.database_name, sim_id)
  input_file = "{0}{1}_{2:0>4}.in".format(static_sm.wdir, static_sm.database_name, sim_id)
  pdict["input_file"] = "{}_{:0>4}.in".format(static_sm.database_name, sim_id)

  if pdict["ofh"]=="true":
    pdict["method"]   = "ffc_full_mpi_static_solution" 
    if pdict["dim"] ==3:
      print("WARNING: it might not work in 3D")
  else:
    pdict["method"]   = "ffc_analytic_static_solution" # 3d works for now only with analytic solution
    
  # "ffc_full_mpi_static_solution" , "ffc_analytic_static_solution"
  # tagging or scotch
  pdict["part_met"] = "scotch" # "scotch" , "tagging"

  if pdict["method"] == "ffc_full_mpi_static_solution":
    if pdict["part_met"] == "tagging": # tagging
      pdict["mpippn"]   = 10 # mpi_processes_per_node
      pdict["xproc"]    = 1  # has to be =1 if pbc aside from that it can be anything
      pdict["yproc"]    = 3  # should be an odd number
      # -------------------------------------------------------
      if pbc and not pdict["xproc"] == 1:
        print("xproc has to be zero")
        pdict["xproc"] = 1
      pdict["mpip"] = int( pdict["XPROC"] * pdict["YPROC"])
      pdict["nb_nodes"] = int(math.ceil(pdict["mpip"] / float(pdict["mpippn"])))
    elif pdict["part_met"] == "scotch": # scotch
      pdict["mpippn"]   = 4 #mpi_processes_per_node
      pdict["nb_nodes"] = int(min(16,max(1,round(.75*pdict["dim"]*pdict["msh"]**2/3200**2))))
      pdict["hrs"] = 2*pdict["nb_nodes"]
      # -------------------------------------------------------
      pdict["mpip"] = int(pdict["nb_nodes"] * pdict["mpippn"]) # mpi process
      pdict["xproc"] = 1 # no effect, just to make this script work
      pdict["yproc"] = 1 # no effect, just to make this script work
  elif pdict["method"] == "ffc_analytic_static_solution":
    # -------------------------------------------------------
    # method is not parallel
    pdict["mpippn"]   = 1 # mpi_processes_per_node
    pdict["nb_nodes"] = 1
    pdict["hrs"] = 2
    pdict["mpip"] = int(pdict["nb_nodes"] * pdict["mpippn"]) # mpi process
    pdict["xproc"] = 1 # no effect, just to make this script work
    pdict["yproc"] = 1 # no effect, just to make this script work
  

  # write files
  with open(input_file, "w") as inp:
     print(file_str["input"].format(**pdict),file=inp)
  with open(input_file, "a") as inp:
    if pdict["ofh"]=="true":
      print(file_str["ofhgeom"].format(**pdict),file=inp)
    print(file_str["options"].format(**pdict),file=inp)
    print(file_str["materials"].format(**pdict),file=inp)
 
  with open(sub_file,"w") as sub:
     print(file_str["sub"].format(**pdict),file=sub)

  print("Static input generated  sim_id {0:0>4}".format(sim_id))
  
  # update status
  static_sm.update_status(sim_id, static_sm.status.generated)
'''
import sys
def usage(msg):
    print msg

def main(argv=None):
    """Usage:  generatestatic.py <project name> <sim_id> """
    if argv is None:
        args = sys.argv[1:]
    if "-h" in args or "--help" in args:
        usage(main.__doc__)
        sys.exit(2)
    if len(args) is not 2:
        usage(main.__doc__)
        sys.exit(2)

    sim_id = args[0]
    print("static_id " , sim_id)
    generate_static(


if __name__ == "__main__":
    sys.exit(main())

'''
