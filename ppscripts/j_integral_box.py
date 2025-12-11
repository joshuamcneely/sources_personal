#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import ifasha.datamanager as idm
import solidmechanics as sm
import solidmechanics.lefm as lefm

import ppscripts.postprocess as pp
from ppscripts.save_ruptures import load_ruptures

def plot_j_integral(sname, time, Ds, **kwargs):

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    with_legend = kwargs.get('with_legend',False)

    # get values
    Js,Jts,Jbs = j_integral(sname,time,Ds, **kwargs)
    print(Js)

    ax.plot(Ds, Jts, '.-', label='top half')
    ax.plot(Ds, Jbs, '.-', label='bot half')
    ax.plot(Ds, Js, '.-', label='tot')

    if new_figure or with_legend:
        ax.legend(loc='best')

    return ax

def j_integral(sname, time, Ds, **kwargs):
    
    wdir = kwargs.get('wdir',"./data")
    sname = pp.sname_to_sname(sname)

    # strain is dumped from quad points
    # velocity from nodal points
    # thus two different groups
    strain_group = kwargs.get('strain_group','full')
    velo_group = kwargs.get('velo_group','node-full')

    # find material properties in input file
    top_mat_name = kwargs.get('top_mat_name','top')
    bot_mat_name = kwargs.get('bot_mat_name','bot')

    input_data = pp.get_input_data(sname, **kwargs)
    code = pp.get_code_name(sname, group=strain_group, **kwargs)
    if code == 'akantu':
        mats = [s for s in input_data.keys() if 'material' in s]
        # homogeneous set-up with one material
        if len(mats) == 1:
            mat_dict = input_data[mats[0]]
            mat = sm.LinearElasticMaterial({
                sm.smd.E       : float(mat_dict['E']),
                sm.smd.nu      : float(mat_dict['nu']),
                sm.smd.rho     : float(mat_dict['rho']),
                sm.smd.pstress : bool(mat_dict['Plane_Stress'])
            })
            tmat = mat
            bmat = mat
        # bi-material setup
        elif len(mats) == 2:
            for matname in mats:
                mat_dict = input_data[matname]
                mat = sm.LinearElasticMaterial({
                    sm.smd.E       : float(mat_dict['E']),
                    sm.smd.nu      : float(mat_dict['nu']),
                    sm.smd.rho     : float(mat_dict['rho']),
                    sm.smd.pstress : bool(mat_dict['Plane_Stress'])
                })
                mat_name = mat_dict['name']
                if mat_name == top_mat_name:
                    tmat = mat
                elif mat_name == bot_mat_name:
                    bmat = mat
                else:
                    print('do not know material name',mat_name)
                    raise
        else:
            print('too many or not enough materials')
            raise

        # get interface properties
        int_dict = input_data['friction-0']
        Gamma = float(int_dict['G_c'])
        print('Gamma = {}'.format(Gamma))
        #tau_c = float(int_dict['tau_c'])
        tau_r = float(int_dict['tau_r'])

    res_stress = sm.CauchyStress([[0,tau_r],[tau_r,0]])
    print('residual shear stress',res_stress)

    # load the simulation data
    dma = idm.DataManagerAnalysis(sname,wdir)
    strain_data = dma(strain_group)

    # find time index closest to provided time
    tidx = strain_data.get_index_of_closest_time(idm.FieldId('time'),time)
    time = strain_data.get_full_field(idm.FieldId('time'))[tidx]
    print('actual time = {}'.format(time))

    # find crack tip position
    rpt = load_ruptures(sname)[0]
    rpt = rpt.first()
    front = rpt.get_sorted_front()
    nuc_p, nuc_t = rpt.get_nucleation()
    nuc_i = np.argmin(abs(front[:,0]-nuc_p[0]))
    prop_dir = kwargs.get('prop_dir','to_right')
    if prop_dir == 'to_right':
        ridx = nuc_i + np.argmin(abs(front[nuc_i:,2]-time))
        xtip = front[ridx,0]
    elif prop_dir == 'to_left':
        ridx = np.argmin(abs(front[:nuc_i,2]-time))
        xtip = front[ridx,0]
    else:
        print('do not understand prop_dir: {}'.format(prop_dir))
        raise RuntimeError
    print('crack tip position = {}'.format(xtip))

    # tractions
    data = dma('interface')
    # position
    if data.has_field(idm.FieldId('position',0)):
        x_fldid = idm.FieldId('position',0)
    elif data.has_field(idm.FieldId('coord',0)):
        x_fldid = idm.FieldId('coord',0)
    else:
        raise('Does not have any position information')
    x_position = data.get_field_at_t_index(x_fldid,0)
    tractions = data.get_field_at_t_index(idm.FieldId('friction_traction',0), tidx)
    # sort
    soi = np.argsort(x_position, axis=1)
    sti = np.indices(x_position.shape)
    x_position = x_position[sti[0],soi][0,:]
    tractions  = tractions[sti[0],soi][0,:]
    
    tipidx = np.argmin(abs(x_position-xtip))
    if prop_dir == 'to_right':
        tipidx += 1 # because front is positioned where it slides
    elif prop_dir == 'to_left':
        tipidx -= 1 # because front is positioned where it slides
        
    # find position of cohesive zone tail
    if prop_dir == 'to_right':
        tailidx = np.where(tractions[:tipidx]-tau_r < 1e-6*tau_r)[0][-1]
        xtail = x_position[tailidx]
        llim = tailidx
        # because [:rlim] needs plus one to include this position
        rlim = min(tipidx+1,len(x_position)) 
    elif prop_dir == 'to_left':
        tailidx = tipidx+np.where(tractions[tipidx:]-tau_r < 1e-6*tau_r)[0][0]
        xtail = x_position[tailidx]
        llim = tipidx
        # because [:rlim] needs plus one to include this position
        rlim = min(tailidx+1,len(x_position))
    else:
        print('do not understand prop_dir: {}'.format(prop_dir))
        raise RuntimeError
    #print('cohesive zone tail position = {}'.format(xtail))
    
    Xc = abs(xtip - xtail)


    # find rupture speed
    avg_dist = kwargs.get('avg_dist',0.005)
    prop_speed = rpt.get_propagation_speed(0,avg_dist)
    cidx = np.argmin(abs(prop_speed[:,0]-xtip))
    Cf = prop_speed[cidx,2]

    # x position
    if strain_data.has_field(idm.FieldId('position',0)):
        x_fldid = idm.FieldId('position',0)
    elif strain_data.has_field(idm.FieldId('coord',0)):
        x_fldid = idm.FieldId('coord',0)
    else:
        raise('Does not have any position information')
        
    # y position
    if strain_data.has_field(idm.FieldId('position',1)):
        y_fldid = idm.FieldId('position',1)
    elif strain_data.has_field(idm.FieldId('coord',1)):
        y_fldid = idm.FieldId('coord',1)
    else:
        raise('Does not have any position information')


    # find slice to load only part of full field (otherwise too much memory)
    # have to do the same way as get_xy_plot does to get correct slice
    X0 = strain_data.get_field_memmap(x_fldid)[0,:]
    Y0 = strain_data.get_field_memmap(y_fldid)[0,:]

    p = X0.argsort()
    X0 = X0[p]
    Y0 = Y0[p]
    Xr = X0[::-1]
    Xmin = Xr[-1]
    lng = len(Xr) - np.where(Xr==Xmin)[0][0]
    X0.shape = [-1,lng]
    Y0.shape = [-1,lng]

    soi = np.argsort(Y0, axis=1)
    sti = np.indices(Y0.shape)
    X0 = X0[sti[0], soi]
    Y0 = Y0[sti[0], soi]

    X0 = X0[:,0] - xtip
    Y0 = Y0[0]

    # approximate element size
    dx = (np.max(X0) - np.min(X0)) / len(X0)
    dy = (np.max(Y0) - np.min(Y0)) / len(Y0)
    print('approx elem size: {},{}'.format(dx,dy))

    # make slices to load only part of full field
    Dmax = max(Ds)
    xE=Dmax+5*dx # exterior limit
    boxshape = kwargs.get('boxshape','cohzoneplus')#'square')
    if boxshape == 'square':
        Xslice = np.array([i for i,j in enumerate(X0) if -xE<j<xE])
    elif boxshape == 'cohzoneplus':
        Xslice = np.array([i for i,j in enumerate(X0) if -(Xc+2*dx)<j<xE])

    yE=Dmax+5*dy # exterior limit
    Yslice = np.array([i for i,j in enumerate(Y0) if -yE<j<yE])

    
    # get reference strain when residual stress is non-zero
    print('Load strain')
    X,Y,E110 = strain_data.get_sliced_x_sliced_y_plot(x_fldid,
                                                      Xslice,
                                                      y_fldid,
                                                      Yslice,
                                                      idm.FieldId('strain',0),
                                                      0)
    X,Y,E220 = strain_data.get_sliced_x_sliced_y_plot(x_fldid,
                                                      Xslice,
                                                      y_fldid,
                                                      Yslice,
                                                      idm.FieldId('strain',1),
                                                      0)
    E120=[]
    for yt in Y.flatten():
        if yt > 0:
            tmp_mat = tmat
        else:
            tmp_mat = bmat
        tmp_esp = tmp_mat.stress_to_strain(res_stress)
        E120.append(tmp_esp[0,1])
    E120 = np.array(E120)
    E120.shape = X.shape


    # get actual strain
    X,Y,E11 = strain_data.get_sliced_x_sliced_y_plot(x_fldid,
                                                     Xslice,
                                                     y_fldid,
                                                     Yslice,
                                                     idm.FieldId('strain',0),
                                                     tidx)
    X,Y,E22 = strain_data.get_sliced_x_sliced_y_plot(x_fldid,
                                                     Xslice,
                                                     y_fldid,
                                                     Yslice,
                                                     idm.FieldId('strain',1),
                                                     tidx)
    X,Y,E12 = strain_data.get_sliced_x_sliced_y_plot(x_fldid,
                                                     Xslice,
                                                     y_fldid,
                                                     Yslice,
                                                     idm.FieldId('strain',2),
                                                     tidx)



    # tmp
    if False:
        tmat = bmat
        tmp_S11, tmp_S22, tmp_S12 = lefm.mode_2_subRayleigh_singular_stress_field(Cf,
                                                                                  tmat,
                                                                                  {sm.smd.Gamma : 1.1},
                                                                                  x=X.flatten()-xtip,
                                                                                  y=Y.flatten(),
                                                                                  rformat='element-wise')
        tmp_eps = [tmat.stress_to_strain(sm.CauchyStress([[s11,s12],[s12,s22]])) 
                   for s11,s22,s12 in zip(tmp_S11,tmp_S22,tmp_S12)]
        E11 = np.array([e[0,0] for e in tmp_eps])
        E22 = np.array([e[1,1] for e in tmp_eps])
        E12 = np.array([e[0,1] for e in tmp_eps])
        E11.shape = X.shape
        E22.shape = X.shape
        E12.shape = X.shape
        tmp_S11.shape = X.shape
        tmp_S22.shape = X.shape
        tmp_S12.shape = X.shape

    if False:
        S12=[]
        for e11,e22,e12,yt in zip(E11.flatten(),E22.flatten(),E12.flatten(),Y.flatten()):
            if yt > 0:
                tmp_mat = tmat
            else:
                tmp_mat = bmat
            tmp_sig = tmp_mat.strain_to_stress(sm.InfinitesimalStrain([[e11,e12],[e12,e22]]))
            S12.append(tmp_sig[0,1])
        S12 = np.array(S12)
        S12.shape = X.shape
    # end tmp

    # for debugging
    if False:
        Zplot = tmp_S12
        fig, ax = plt.subplots(1,1)
        ax.pcolormesh(X,Y,Zplot,vmin=np.nanmin(Zplot),vmax=np.nanmax(Zplot))
        ax.axis([X.min(), X.max(), Y.min(), Y.max()])


    # get velocity
    print('Load velocities')
    velo_data = dma(velo_group)
    tidx = velo_data.get_index_of_closest_time(idm.FieldId('time'),time)

    # find slice to load only part of full field (otherwise too much memory)
    # have to do the same way as get_xy_plot does to get correct slice
    XV0 = velo_data.get_field_memmap(x_fldid)[0,:]
    YV0 = velo_data.get_field_memmap(y_fldid)[0,:]

    p = XV0.argsort()
    XV0 = XV0[p]
    YV0 = YV0[p]
    Xr = XV0[::-1]
    Xmin = Xr[-1]
    lng = len(Xr) - np.where(Xr==Xmin)[0][0]
    XV0.shape = [-1,lng]
    YV0.shape = [-1,lng]

    soi = np.argsort(YV0, axis=1)
    sti = np.indices(YV0.shape)
    XV0 = XV0[sti[0], soi]
    YV0 = YV0[sti[0], soi]

    XV0 = XV0[:,0] - xtip
    YV0 = YV0[0]

    # approximate element size
    dxv = (np.max(XV0) - np.min(XV0)) / len(XV0)
    dyv = (np.max(YV0) - np.min(YV0)) / len(YV0)

    # make slices to load only part of full field
    xvE=xE+dxv # exterior limit
    XVslice = np.array([i for i,j in enumerate(XV0) if -xvE<j<xvE])

    yvE=yE+dyv # exterior limit
    YVslice = np.array([i for i,j in enumerate(YV0) if -yvE<j<yvE])
    
    XV,YV,V1t = velo_data.get_sliced_x_sliced_y_plot(x_fldid,
                                                     XVslice,
                                                     y_fldid,
                                                     YVslice,
                                                     idm.FieldId('velocity',0),
                                                     tidx)
    XV,YV,V2t = velo_data.get_sliced_x_sliced_y_plot(x_fldid,
                                                     XVslice,
                                                     y_fldid,
                                                     YVslice,
                                                     idm.FieldId('velocity',1),
                                                     tidx)
    
    # tmp
    if False:
        V1t,V2t = lefm.mode_2_subRayleigh_singular_velocity_field(Cf,
                                                                  tmat,
                                                                  {sm.smd.Gamma : 1.1},
                                                                  x=XV-xtip,
                                                                  y=YV,
                                                                  rformat='element-wise')
    # end tmp
        

    # need to find velocity on coordinates of strain
    # fast and dirty: will generate noise across crack surface
    V1mask = np.isfinite(V1t.flatten())
    V1 = griddata(np.array([[i,j] for i,j in zip(XV.flatten()[V1mask],
                                                 YV.flatten()[V1mask])]),
                  V1t.flatten()[V1mask],
                  np.array([[i,j] for i,j in zip(X.flatten(),Y.flatten())]),
                  method='linear')
    V1.shape = X.shape

    V2mask = np.isfinite(V2t.flatten())
    V2 = griddata(np.array([[i,j] for i,j in zip(XV.flatten()[V2mask],
                                                 YV.flatten()[V2mask])]),
                  V2t.flatten()[V2mask],
                  np.array([[i,j] for i,j in zip(X.flatten(),Y.flatten())]),
                  method='linear')
    V2.shape = X.shape

    # for debugging
    if False:
        fig, axs = plt.subplots(1,2)
        Xplot = XV
        Yplot = YV
        Zplot = V1t
        axs[0].pcolor(Xplot,Yplot,Zplot,
                      vmin=np.nanmin(Zplot[Zplot != np.inf]),
                      vmax=np.nanmax(Zplot[Zplot != np.inf]))
        axs[0].axis([np.nanmin(Xplot), np.nanmax(Xplot), 
                     np.nanmin(Yplot), np.nanmax(Yplot)])
        
        Xplot = X
        Yplot = Y
        Zplot = V1
        axs[1].pcolormesh(Xplot,Yplot,Zplot,
                          vmin=np.nanmin(Zplot[Zplot != np.inf]),
                          vmax=np.nanmax(Zplot[Zplot != np.inf]))
        axs[1].axis([np.nanmin(Xplot), np.nanmax(Xplot), 
                     np.nanmin(Yplot), np.nanmax(Yplot)])

        plt.show()

    # local coordinate system with x = X - xtip
    x = X - xtip
    y = Y

    # compute J-integral for bot half and top half box independently
    Jts = []
    Jbs = []
    Js = []
    for D in Ds:
        if boxshape == 'square':
            tnormals,tsides = lefm.get_top_half_box(x,y,l=-D,r=D,b=-D,t=D)
            bnormals,bsides = lefm.get_bot_half_box(x,y,l=-D,r=D,b=-D,t=D)
        elif boxshape == 'cohzoneplus':
            tnormals,tsides = lefm.get_top_half_box(x,y,l=-Xc,r=D,b=-D,t=D)
            bnormals,bsides = lefm.get_bot_half_box(x,y,l=-Xc,r=D,b=-D,t=D)

        Jt = lefm.j_integral_box_path(tnormals,
                                      tsides,
                                      x,y,
                                      tmat,
                                      v=Cf,
                                      eps11=E11-E110,
                                      eps22=E22-E220,
                                      eps12=E12,
                                      du1dt=V1,
                                      du2dt=V2)

        Jb = lefm.j_integral_box_path(bnormals,
                                      bsides,
                                      x,y,
                                      bmat,
                                      v=Cf,
                                      eps11=E11-E110,
                                      eps22=E22-E220,
                                      eps12=E12,
                                      du1dt=V1,
                                      du2dt=V2)
        
        Jts.append(Jt)
        Jbs.append(Jb)
        Js.append(Jt+Jb)

    return np.array(Js), np.array(Jts), np.array(Jbs)


if __name__ == "__main__":

    if not len(sys.argv) in [4,6]:
        sys.exit('Missing argument! usage: ./j-integral.py sname/sim-id time box-half-size')

    sim = str(sys.argv[1])
    time = float(sys.argv[2])
    Dmin = float(sys.argv[3])
    
    if len(sys.argv) == 6:
        Dmax = float(sys.argv[4])
        nbDs = int(sys.argv[5])
        Ds = np.linspace(Dmin,Dmax,nbDs)
    else:
        Ds = [Dmin]

    plot_j_integral(sim, time, Ds, 
                    # specifics for bimaterial project
                    top_mat_name='slider', bot_mat_name='base')

    plt.show()
