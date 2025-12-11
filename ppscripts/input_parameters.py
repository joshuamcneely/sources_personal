#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import solidmechanics as sm

import ppscripts.postprocess as pp

def get_material_properties(sname, **kwargs):

    # get information about simulation
    code = pp.get_code_name(sname, **kwargs)
    input_data = pp.get_input_data(sname,**kwargs)

    materials = dict()
    
    if code == 'weak-interface':
        mats = input_data['mat_names'].split(',')
        for matname in mats:
            mat = sm.LinearElasticMaterial({
                sm.smd.E       : float(input_data['E_'+matname]),
                sm.smd.nu      : float(input_data['nu_'+matname]),
                sm.smd.rho     : float(input_data['rho_'+matname]),
                sm.smd.pstress : bool(input_data['pstress_'+matname])
            })
            materials[matname] = mat
        
    elif code == 'akantu':
        mats = [s for s in input_data.keys() if 'material' in s]
        for mat_name in mats:
            mat_dict = input_data[mat_name]
            matname = mat_dict['name']
            mat = sm.LinearElasticMaterial({
                sm.smd.E       : float(mat_dict['E']),
                sm.smd.nu      : float(mat_dict['nu']),
                sm.smd.rho     : float(mat_dict['rho']),
                sm.smd.pstress : bool(mat_dict['Plane_Stress'])
            })
            materials[matname] = mat
            
    return materials

def get_wave_speed(sname, wavename,**kwargs):

    # get all the material properties
    materials = get_material_properties(sname,**kwargs)

    # find the material of interest
    for matname in materials:
        if matname in wavename:
            mat = materials[matname]

    # find the relevant wave speed
    if wavename.endswith('cp'):
        c0 = mat[sm.smd.cp]
    elif wavename.endswith('cs'):
        c0 = mat[sm.smd.cs]
    elif wavename.endswith('cR'):
        c0 = mat[sm.smd.cR]

    return c0
