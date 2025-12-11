#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import numpy as np
from collections import Iterable

import ifasha.datamanager as idm

wdir = './data'

bname = 'dyn_2d_TE2.90e+09nu0.39rho1200psss1_BE5.65e+09nu0.33rho1180psss1_gc1.10tc4.80e+06tr3.20e+06_taui3.60e+06sigi5.00e+06_nucC0.1HS7.0e-02WZ8.00e-03V93_L0.3pbcHt0.15b0.15msh4800'
bname = 'dyn_2d_TE2.90e+09nu0.39rho1200psss1_BE5.65e+09nu0.33rho1180psss1_gc1.10tc4.40e+06tr3.20e+06_taui3.50e+06sigi5.00e+06_nucC0.1HS2.2e-02WZ8.00e-03V93_L0.3pbcHt0.15b0.15msh4800'
bname = 'dyn_2d_gc1.10tc4.40e+06tr3.20e+06_taui3.45e+06sigi5.00e+06_nucC0.1HS2.4e-02WZ8.00e-03V93_hetXs2.50e-02Xe2.00e-01gc7.53e+00tc4.40e+06tr3.20e+06_L0.3pbcHt0.15b0.15msh4800'

bname = 'dyn_2d_TE2.90e+09nu0.39rho1200psss1_BE5.65e+09nu0.33rho1180psss1_gc1.10tc4.40e+06tr3.20e+06_taui3.50e+06sigi5.00e+06_nucC0.1HS2.2e-02WZ8.00e-03V93_L0.3pbcHt0.15b0.15msh4800'
bname = 'dyn_2d_TE2.90e+09nu0.39rho1200psss1_BE5.65e+09nu0.33rho1180psss1_gc1.10tc4.40e+06tr3.20e+06_taui3.50e+06sigi5.00e+06_nucC0.1HS2.4e-02WZ8.00e-03V93_L0.45pbcHt0.22b0.22msh7200'

bname = 'dyn_2d_TE2.90e+09nu0.39rho1200psss1_BE5.65e+09nu0.33rho1180psss1_gc1.10tc4.40e+06tr3.20e+06_taui3.50e+06sigi3.00e+06_nucC0.1HS2.4e-02WZ8.00e-03V93_L0.3pbcHt0.15b0.15msh4800'

base = "bimat_lower_norm_stress"

# ---------------------------------------------------------------------

# full
flds = [
    [idm.FieldId('strain',0), "eps11"],
    [idm.FieldId('strain',1), "eps22"],
    [idm.FieldId('strain',2), "eps12"]
]
groups = ['full']

# off fault nodal
flds = [
    [idm.FieldId('displacement',0),'disp1'],
    [idm.FieldId('displacement',1),'disp2'],
]
groups = [
    'node-at-distance-6.25e-05',
    'node-at-distance--6.25e-05',
]

# displacements
flds = [
    [idm.FieldId('displacement',0), 'disp1'],
    [idm.FieldId('displacement',1), 'disp2'],
]
groups = [
    'slider_bottom',
    'base_top',
]

# off fault
flds = [
    [idm.FieldId('strain',0), "eps11"],
    [idm.FieldId('strain',1), "eps22"],
    [idm.FieldId('strain',2), "eps12"],
    #[idm.FieldId('gradu',0), "gradu11"],
    #[idm.FieldId('gradu',1), "gradu12"],
    #[idm.FieldId('gradu',2), "gradu21"],
    #[idm.FieldId('gradu',3), "gradu22"],
]
groups = ['at-distance']
groups = [
    'at-distance-3.51e-03',
    'at-distance-7.01e-03',
    'at-distance-1.32e-05',
    'at-distance--3.01e-03',
    'at-distance--3.51e-03',
    'at-distance--7.01e-03',
    'at-distance--1.32e-05',
    'at-distance-3.01e-03',
]

# at interface
flds = [
    [[idm.FieldId('friction_traction',0), idm.FieldId('friction_traction',1)], 'ft'], # inclined interface
    [[idm.FieldId('contact_pressure',0), idm.FieldId('contact_pressure',1)], 'cp'],
    [idm.FieldId('slip',0), 'slip'],
    [[idm.FieldId('slip_velocity',0), idm.FieldId('slip_velocity',1)], 'slip_velocity']
]
#flds = [[idm.FieldId('friction_traction',0), 'ft']] # horizontal interface
groups = ['interface']

tidcs_select = None # [4] = forth dump ; None = you want automatically all
# if tidcs is None, you can set:
tidxinterval=1 # 1 = all
end_time = None # None = until end of simulation

# ---------------------------------------------------------------------
# load the simulation data
dma = idm.DataManagerAnalysis(bname,wdir)

for group in groups:
    data = dma(group)

    simid = data.sim_info.split('=')[1]
    if tidcs_select is None:
        name = base+"_"+simid
    else:
        tt="_".join([str(i) for i in tidcs_select])
        name = base+"_"+simid+"_tidx"+tt

    positions_x = []
    positions_y = []
    fields_data = []
    for fld in flds:
        fields_data.append([])

    if tidcs_select is None:
        if end_time is None:
            lasttidx = data.get_t_index('last')
        else:
            lasttidx = data.get_index_of_closest_time(idm.FieldId('time'),end_time)
        print('last t index =',lasttidx)
        tidcs = range(1,lasttidx+1,tidxinterval)
    else:
        tidcs = [i for i in tidcs_select]

    times = []
    for tidx in tidcs:

        time = data.get_field_at_t_index(idm.FieldId('time'),tidx)[0]
        times.append(time)
        print('time',time)

        for f in range(len(flds)):
            fld = flds[f]
            print(tidx,fld[0])


            if not isinstance(fld[0], Iterable): # only one field

                position_x, position_y, field = data.get_xy_plot(idm.FieldId('position',0),
                                                                 idm.FieldId('position',1),
                                                                 fld[0],
                                                                 tidx)

                position_x = np.reshape(position_x, (1,-1))
                position_y = np.reshape(position_y, (1,-1))
                field      = np.reshape(field, (1,-1))

                fields_data[f].append(field[0,:])


            else:
                field_to_add = 0
                for ff in fld[0]:
                    position_x, position_y, field = data.get_xy_plot(idm.FieldId('position',0),
                                                                     idm.FieldId('position',1),
                                                                     ff,
                                                                     tidx)

                    position_x = np.reshape(position_x, (1,-1))
                    position_y = np.reshape(position_y, (1,-1))
                    field      = np.reshape(field, (1,-1))

                    field_to_add += field * field

                field_to_add = np.sqrt(field_to_add)
                fields_data[f].append(field_to_add[0,:])


            if f == 0:
                positions_x.append(position_x[0,:])
                positions_y.append(position_y[0,:])

    print('start writing files')

    times       = np.array(times)    
    positions_x = np.transpose(np.array(positions_x))
    positions_y = np.transpose(np.array(positions_y))
    for fd in fields_data:
        fd = np.transpose(np.array(fd))

    np.savetxt(name+"_"+group+"_time"+".csv", times, delimiter=",")
    np.savetxt(name+"_"+group+"_x"+".csv", positions_x[:,0], delimiter=",")
    np.savetxt(name+"_"+group+"_y"+".csv", positions_y[:,0], delimiter=",")
    for f in range(len(fields_data)):
        np.savetxt(name+"_"+group+"_"+flds[f][1]+".csv", fields_data[f], delimiter=",")
