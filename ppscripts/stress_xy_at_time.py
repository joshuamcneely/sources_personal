#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d

import ifasha.datamanager as idm
import solidmechanics as sm

# location of simulation data
wdir = './data'

bname = '449208'
bname = 'dyn_2d_TE2.90e+09nu0.39rho1200psss0_BE5.65e+09nu0.33rho1180psss0_gc1.10tc4.80e+06tr3.20e+06_taui3.80e+06sigi5.00e+06_nucC0.1HS5.0e-02WZ8.00e-03V932_L0.3pbcHt0.15b0.15msh4800'
bname = 'dyn_2d_TE2.90e+09nu0.39rho1200psss0_BE5.65e+09nu0.33rho1180psss0_gc1.10tc4.80e+06tr3.20e+06_taui3.70e+06sigi5.00e+06_nucC0.1HS5.0e-02WZ8.00e-03V93_L0.3pbcHt0.15b0.15msh4800'

pmma_bot = sm.LinearElasticMaterial({
    sm.smd.E : 5.65e9,
    sm.smd.nu : 0.33,
    sm.smd.rho : 1180,
    sm.smd.pstrain : True
})

pc_top = sm.LinearElasticMaterial({
    sm.smd.E : 2.9e9,
    sm.smd.nu : 0.39,
    sm.smd.rho : 1200,
    sm.smd.pstrain : True
})

# all
nb_x_elements  = 9600 # nb quad points = 2x nb nodes
st_x_elements  = 3200
int_x_elements = 40

nb_y_elements  = 3200 # only top half
nb_y_elements = 9600 # top and bot
st_y_elements  = 0
int_y_elements = 40


# ---------------------------------------------------------------------
# load data manager
dma = idm.DataManagerAnalysis(bname,wdir)
data = dma('full')

last = data.get_t_index('last')

X,Y,Uxx = data.get_sliced_x_sliced_y_plot(idm.FieldId("position",0),
                                          np.arange(st_x_elements,nb_x_elements,int_x_elements),
                                          idm.FieldId("position",1),
                                          np.arange(st_y_elements,nb_y_elements,int_y_elements),
                                          idm.FieldId('strain',0), last)

X,Y,Uyy = data.get_sliced_x_sliced_y_plot(idm.FieldId("position",0),
                                          np.arange(st_x_elements,nb_x_elements,int_x_elements),
                                          idm.FieldId("position",1),
                                          np.arange(st_y_elements,nb_y_elements,int_y_elements),
                                          idm.FieldId('strain',1), last)

X,Y,Uxy = data.get_sliced_x_sliced_y_plot(idm.FieldId("position",0),
                                          np.arange(st_x_elements,nb_x_elements,int_x_elements),
                                          idm.FieldId("position",1),
                                          np.arange(st_y_elements,nb_y_elements,int_y_elements),
                                          idm.FieldId('strain',2), last)

Sxx = np.zeros_like(Uxx)
Syy = np.zeros_like(Uxx)
Sxy = np.zeros_like(Uxx)

strain = sm.InfinitesimalStrain()
for (i,j),tmp in np.ndenumerate(Uxx):
    strain[0,0] = Uxx[i,j]
    strain[1,1] = Uyy[i,j]
    strain[1,0] = Uxy[i,j]
    strain[0,1] = strain[1,0]

    y = Y[i,j]
    
    if y > 0:
        stress = pc_top.strain_to_stress(strain)
    elif y < 0:
        stress = pmma_bot.strain_to_stress(strain)
    else:
        print('y is zero, why?')
        raise(RuntimeError)

    Sxx[i,j] = stress[0,0]
    Syy[i,j] = stress[1,1]
    Sxy[i,j] = stress[1,0]

fig = plt.figure()
ax = fig.add_subplot(111)
fg1 = ax.pcolor(X,Y,Sxx)#,vmin=-6e6,vmax=-4e6)
plt.colorbar(fg1)

fig = plt.figure()
ax = fig.add_subplot(111)
fg2 = ax.pcolor(X,Y,Syy,vmin=-6e6,vmax=-4e6)
plt.colorbar(fg2)

fig = plt.figure()
ax = fig.add_subplot(111)
fg3 = ax.pcolor(X,Y,Sxy,vmin=3e6,vmax=4.5e6)
plt.colorbar(fg3)

plt.show()
