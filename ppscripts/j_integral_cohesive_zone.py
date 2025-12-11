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
from   ppscripts.input_parameters import get_wave_speed
from   ppscripts.save_ruptures import load_ruptures

import notes_rcParams as rcp


def plot_j_integral(snames, **kwargs):

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    with_legend = kwargs.get('with_legend',False)

    x_data = kwargs.get('plot_vs','time')

    # needs to know direction
    # otherwise j_integral will do differently from here
    if 'prop_dir' not in kwargs:
        kwargs['prop_dir'] = 'to_right'

    cidx = kwargs.get('coloridx',0)

    for sname in snames:

        sname = pp.sname_to_sname(sname)

        # get J integral values
        times,Jbs,Jts = j_integral(sname, **kwargs)

        if x_data == 'time':
            ax.plot(times, Jts, '.-', label='top half')
            ax.plot(times, Jbs, '.-', label='bot half')
            ax.plot(times, Jts+Jbs, '.-', label='tot')
            
            ax.set_xlabel('time')
        elif x_data == 'Cf':

            # options are: 'cp', 'cs', 'cR'
            # and 'matname-cp', 'matname-cs', 'matname-cR'
            norm_speed = kwargs.get('norm_speed','one')

            c0 = 1.
            if not norm_speed == 'one':
                # value is given
                if type(norm_speed) == float or type(norm_speed) == int:
                    c0 = norm_speed
                # value is a wave speed
                else:
                    c0 = get_wave_speed(sname,norm_speed,**kwargs)
                    
            # get rupture speed
            rpt = load_ruptures(sname)[0]
            rpt = rpt.first()
            Cf = rpt.get_propagation_speed(0,**kwargs)

            Cfs = np.array([Cf[np.argmin(abs(Cf[:,1]-t)),2] for t in times])

            # try to add simid to label
            simid=''
            try:
                simid = pp.sname_to_simid(sname)
            except:
                pass

            if not 'plot' in kwargs:
                      
                label=kwargs.get('label','')
                ax.plot(Cfs/c0, Jts, '-', 
                        color=rcp.tableau20[cidx%len(rcp.tableau20)],
                        label='{} top'.format(simid))
                cidx+=1
                ax.plot(Cfs/c0, Jbs, '-',
                        color=rcp.tableau20[cidx%len(rcp.tableau20)],
                        label='{} bot'.format(simid))
                cidx+=1
                ax.plot(Cfs/c0, Jbs+Jts, '-',
                        color=rcp.tableau20[cidx%len(rcp.tableau20)],
                        label='{} - {}'.format(simid,label))
                cidx+=2

            else:
                plot=kwargs.get('plot')
                plot_labels=kwargs.get('plot_labels')
                plot_colidcs=kwargs.get('plot_colidcs')
                Jdict = {
                    'top':Jts,
                    'bot':Jbs,
                    'tot':Jbs+Jts}
                
                for pt,ptl,ptc in zip(plot,plot_labels,plot_colidcs):
                    ax.plot(Cfs/c0, Jdict[pt], '-',
                            color=rcp.tableau20[ptc],
                            label=ptl)
                

            
            ax.set_xlabel('Cf/c0')
    
    ax.set_ylabel('J (J/m2)')

    if new_figure or with_legend:
        ax.legend(loc='best')

    return ax

def j_integral(sname, **kwargs):
    
    wdir = kwargs.get('wdir',"./data")
    sname = pp.sname_to_sname(sname)

    # traction from interface
    # displacement and/or velocity group 
    # thus three different groups
    group = kwargs.get('interface_group','interface')
    top_group = kwargs.get('top_group','slider_bottom')
    bot_group = kwargs.get('bot_group','base_top')

    res_precision = kwargs.get('res_precision',1e-2)

    input_data = pp.get_input_data(sname,**kwargs)
    code = pp.get_code_name(sname, **kwargs)
    if code == 'akantu':
        int_dict = input_data['friction-0']
        tau_r = float(int_dict['tau_r'])
    elif code == 'weak-interface':
        sig=kwargs.get('sig',float(input_data['normal_load']))
        if 'tau_r' in input_data:
            tau_r = float(input_data['tau_r'])
        elif 'mu_k' in input_data and 'normal_load' in input_data:
            muk=float(input_data['mu_k'])
            tau_r = abs(muk*sig)
    else:
        print("don't know code")
        raise RuntimeError
    tau_r = kwargs.get('tau_r',tau_r)
    adapt_tau_r = kwargs.get('adapt_tau_r',False)
    adapt_sig = kwargs.get('adapt_sig',False)
    
    # J integral version
    j_base = kwargs.get('j_base','dudx')

    # load the simulation data
    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)
    tdata = dma(top_group)
    bdata = dma(bot_group)

    # position
    if data.has_field(idm.FieldId('position',0)):
        x_fldid = idm.FieldId('position',0)
    elif data.has_field(idm.FieldId('coord',0)):
        x_fldid = idm.FieldId('coord',0)
    else:
        raise('Does not have any position information')

    # traction
    if data.has_field(idm.FieldId('friction_traction',0)):
        fric_fldid = idm.FieldId('friction_traction',0)
        pres_fldid = idm.FieldId('contact_pressure',1)
    elif data.has_field(idm.FieldId('cohesion',0)):
        fric_fldid = idm.FieldId('cohesion',0)
        pres_fldid = idm.FieldId('cohesion',1)
    else:
        raise('Do not know traction field')

    # displacement
    if tdata.has_field(idm.FieldId('displacement',0)):
        tu_fldid = idm.FieldId('displacement',0)
    elif tdata.has_field(idm.FieldId('top_disp',0)):
        tu_fldid = idm.FieldId('top_disp',0)
    else:
        raise('Do not know displacement field')
    if bdata.has_field(idm.FieldId('displacement',0)):
        bu_fldid = idm.FieldId('displacement',0)
    elif bdata.has_field(idm.FieldId('bot_disp',0)):
        bu_fldid = idm.FieldId('bot_disp',0)
    else:
        raise('Do not know displacement field')
    
    # velocity
    if tdata.has_field(idm.FieldId('velocity',0)):
        tvx_fldid = idm.FieldId('velocity',0)
        tvy_fldid = idm.FieldId('velocity',1)
    elif tdata.has_field(idm.FieldId('top_velo',0)):
        tvx_fldid = idm.FieldId('top_velo',0)
        tvy_fldid = idm.FieldId('top_velo',1)
    else:
        raise('Do not know velocity field')
    if bdata.has_field(idm.FieldId('velocity',0)):
        bvx_fldid = idm.FieldId('velocity',0)
        bvy_fldid = idm.FieldId('velocity',1)
    elif bdata.has_field(idm.FieldId('bot_velo',0)):
        bvx_fldid = idm.FieldId('bot_velo',0)
        bvy_fldid = idm.FieldId('bot_velo',1)
    else:
        raise('Do not know velocity field')


    # get rupture -> for finding later crack tip position
    rpt = load_ruptures(sname)[0]
    rpt = rpt.first()
    front = rpt.get_sorted_front()
    nuc_p, nuc_t = rpt.get_nucleation()
    nuc_i = np.argmin(abs(front[:,0]-nuc_p[0]))
    prop_dir = kwargs.get('prop_dir','to_right')

    if j_base == 'dudt':
        Cf = rpt.get_propagation_speed(0,**kwargs)

    times = data.get_full_field(idm.FieldId('time'))
    start_t = kwargs.get('start_time',0.)
    end_t = kwargs.get('end_time',times[-1])

    start_tidx = data.get_index_of_closest_time(idm.FieldId('time'),start_t)
    end_tidx = data.get_index_of_closest_time(idm.FieldId('time'),end_t)

    rJst = []
    rJsb = []
    rtime = []
    taurs = []
    for tidx in range(start_tidx,end_tidx):#[start_tidx+int(0.92*(end_tidx-start_tidx)):start_tidx+int(0.92*(end_tidx-start_tidx))+8]:

        time = times[tidx]

        # find crack tip position
        if prop_dir == 'to_right':
            ridx = np.argmin(abs(front[nuc_i:,2]-time))
            xtip = front[nuc_i+ridx,0]
            # equal time for different position
            ii = np.where(abs(front[nuc_i:,2]-front[nuc_i+ridx,2]) == 0)[0]
            xtip = front[nuc_i+ii[-1],0]
        elif prop_dir == 'to_left':
            ridx = np.argmin(abs(front[:nuc_i,2]-time))
            xtip = front[ridx,0]
            # equal time for different position
            ii = np.where(abs(front[:nuc_i,2]-front[ridx,2]) == 0)[0]
            xtip = front[ii[0],0]
        else:
            print('do not understand prop_dir: {}'.format(prop_dir))
            raise RuntimeError
        #print('crack tip position = {}'.format(xtip))

        # tractions
        x_position = data.get_field_at_t_index(x_fldid,0)
        tractions = data.get_field_at_t_index(fric_fldid,
                                              tidx)
        pressures = data.get_field_at_t_index(pres_fldid,
                                              tidx)
               
        # sort
        soi = np.argsort(x_position, axis=1)
        sti = np.indices(x_position.shape)
        x_position = x_position[sti[0],soi][0,:]
        tractions  = tractions[sti[0],soi][0,:]
        pressures  = pressures[sti[0],soi][0,:]

        tipidx = np.argmin(abs(x_position-xtip))
        tail_def = kwargs.get('tail_def','taur')

        # find position of cohesive zone tail
        if prop_dir == 'to_right':
            if tail_def == 'taur':
                tailidx = np.where(abs(tractions[:tipidx]-tau_r) 
                                   < res_precision*tau_r)[0]
                if not len(tailidx):
                    continue

                # eliminate a lucky hit while tractions passes by tau_r
                # will lose first entry, but we are interested in last entry
                if len(tailidx) > 2:
                    tailidx = [tailidx[i] for i in range(len(tailidx))[1:] 
                               if abs(tailidx[i]-tailidx[i-1]) == 1]
                    
                tailidx=tailidx[-1]
            elif tail_def == 'minimum':
                # try minimum
                tailidx = np.argmin(abs(tractions[:tipidx]))
                tau_r = tractions[tailidx]
                taurs.append(tau_r)
            elif tail_def == 'taur-behind-minimum':
                # try minimum
                minidx = np.argmin(abs(tractions[:tipidx]))

                tailidx = np.where(abs(tractions[:minidx]-tau_r) 
                                   < res_precision*tau_r)[0]
                if not len(tailidx):
                    continue

                tailidx=tailidx[-1]
            elif tail_def == 'nucleation':
                tailidx = np.argmin(abs(x_position-nuc_p[0]))

            xtail = x_position[tailidx]
            llim = tailidx
            # because [:rlim] needs plus one to include this position
            rlim = min(tipidx+1,len(x_position)) 
        elif prop_dir == 'to_left':
            if tail_def == 'taur':
                tailidx = tipidx+np.where(abs(tractions[tipidx:]-tau_r) 
                                          < res_precision*tau_r)[0]
                if not len(tailidx):
                    continue

                # eliminate a lucky hit while tractions passes by tau_r
                # will lose last entry, but we are interested in first entry
                if len(tailidx) > 2:
                    tailidx = [tailidx[i-1] for i in range(len(tailidx))[1:] 
                               if abs(tailidx[i]-tailidx[i-1]) == 1]
                tailidx=tailidx[0]
            elif tail_def == 'minimum':
                # try minimum
                tailidx = tipidx+np.argmin(abs(traction[tipidx:]))
                tau_r = tractions[tailidx]
                taurs.append(tau_r)
            elif tail_def == 'taur-behind-minimum':
                # try minimum
                minidx = tipidx+np.argmin(abs(tractions[tipidx:]))

                tailidx = minidx+np.where(abs(tractions[minidx:]-tau_r) 
                                          < res_precision*tau_r)[0]
                if not len(tailidx):
                    continue

                tailidx=tailidx[0]
            elif tail_def == 'nucleation':
                tailidx = np.argmin(abs(x_position-nuc_p[0]))

            xtail = x_position[tailidx]
            llim = tipidx
            # because [:rlim] needs plus one to include this position
            rlim = min(tailidx+1,len(x_position))
        else:
            print('do not understand prop_dir: {}'.format(prop_dir))
            raise RuntimeError
        #print('cohesive zone tail position = {}'.format(xtail))
        
        Xc = abs(xtip - xtail)

        tidx = bdata.get_index_of_closest_time(idm.FieldId('time'),time)
        x_bot = bdata.get_field_at_t_index(x_fldid,0)
        u1_bot = bdata.get_field_at_t_index(bu_fldid, tidx)
        v1_bot = bdata.get_field_at_t_index(bvx_fldid, tidx)
        v2_bot = bdata.get_field_at_t_index(bvy_fldid, tidx)

        # sort
        soi = np.argsort(x_bot, axis=1)
        sti = np.indices(x_bot.shape)
        x_bot = x_bot[sti[0],soi][0,:]
        u1_bot = u1_bot[sti[0],soi][0,:]
        v1_bot = v1_bot[sti[0],soi][0,:]
        v2_bot = v2_bot[sti[0],soi][0,:]
        du1dx_bot = lefm.ddx(u1_bot,x_bot)

        llim_bot = np.argmin(abs(x_bot - x_position[llim]))
        rlim_bot = np.argmin(abs(x_bot - x_position[rlim]))

        # check if same discretization
        if not len(x_position[llim:rlim]) == len(x_bot[llim_bot:rlim_bot]):
            print('not same discretization')
            raise RuntimeError

        tidx = tdata.get_index_of_closest_time(idm.FieldId('time'),time)
        x_top = tdata.get_field_at_t_index(x_fldid,0)
        u1_top = tdata.get_field_at_t_index(tu_fldid, tidx)
        v1_top = tdata.get_field_at_t_index(tvx_fldid, tidx)
        v2_top = tdata.get_field_at_t_index(tvy_fldid, tidx)

        # sort
        soi = np.argsort(x_top, axis=1)
        sti = np.indices(x_top.shape)
        x_top = x_top[sti[0],soi][0,:]
        u1_top = u1_top[sti[0],soi][0,:]
        v1_top = v1_top[sti[0],soi][0,:]
        v2_top = v2_top[sti[0],soi][0,:]
        du1dx_top = lefm.ddx(u1_top,x_top)

        llim_top = np.argmin(abs(x_top - x_position[llim]))
        rlim_top = np.argmin(abs(x_top - x_position[rlim]))

        # check if same discretization
        if not len(x_position[llim:rlim]) == len(x_top[llim_top:rlim_top]):
            print('not same discretization')
            raise RuntimeError

        trac = tractions # shorten following lines
        pres = pressures # shorten following lines
        if adapt_tau_r and prop_dir == 'to_right':
            c_tau_r = trac[llim]
        elif adapt_tau_r and prop_dir == 'to_left':
            c_tau_r = trac[rlim]
        else:
            c_tau_r = tau_r
        if adapt_sig and prop_dir == 'to_right':
            c_sig = pres[llim]
        elif adapt_sig and prop_dir == 'to_left':
            c_sig = pres[rlim]
        else:
            c_sig = sig
            
        if j_base == 'dudx':
            du1dxb = du1dx_bot
            du1dxt = du1dx_top
            Js = lefm.j_integral_cohesive_zone(x_position[llim:rlim],
                                               t1_bot=   trac[llim:rlim]-c_tau_r,
                                               t1_top= -(trac[llim:rlim]-c_tau_r),
                                               du1dx_bot=du1dxb[llim_bot:rlim_bot],
                                               du1dx_top=du1dxt[llim_top:rlim_top],
                                               rformat='per-side')
        elif j_base == 'dudt':
            v = Cf[np.argmin(abs(Cf[:,1]-time)),2]
            Js = lefm.j_integral_cohesive_zone(x_position[llim:rlim],
                                               t1 = -(trac[llim:rlim]-c_tau_r),
                                               t2 = -(pres[llim:rlim]-c_sig),
                                               du1dt_bot=v1_bot[llim_bot:rlim_bot],
                                               du1dt_top=v1_top[llim_top:rlim_top],
                                               du2dt_bot=v2_bot[llim_bot:rlim_bot],
                                               du2dt_top=v2_top[llim_top:rlim_top],
                                               v=v,
                                               rformat='per-side')

        rtime.append(time)
        rJsb.append(Js[0])
        rJst.append(Js[1])

        # for debugging
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(211)
            ax.plot(x_position[llim-20:rlim+20],
                    tractions[llim-20:rlim+20],'k--')
            ax.plot(x_position[llim:rlim],
                    tractions[llim:rlim],'.-',
                    label='{},{} = {}'.format(Js[0],Js[1],sum(Js)))
            ax.set_xlabel('x')
            ax.set_ylabel('friction force')
            ax.legend(loc='best')
            ax = fig.add_subplot(212)
            ax.plot(x_bot[llim_bot-20:rlim_bot+20],
                    u1_bot[llim_bot-20:rlim_bot+20],
                    'k--')
            ax.plot(x_top[llim_top-20:rlim_top+20],
                    u1_top[llim_top-20:rlim_top+20],
                    'k:')
            ax.plot(x_top[llim_top-20:rlim_top+20],
                    u1_bot[llim_bot-20:rlim_bot+20]-
                    u1_top[llim_top-20:rlim_top+20],
                    'k-')
            ax.plot(x_top[llim_top:rlim_top],
                    u1_bot[llim_bot:rlim_bot]-
                    u1_top[llim_top:rlim_top],
                    '.-')
            ax.set_xlabel('x')
            ax.set_ylabel('tangential disp')


    # plot evolution of tau_r
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(rtime,taurs)
        ax.set_xlabel('time')
        ax.set_ylabel('tau_r')

    return np.array(rtime), np.array(rJsb), np.array(rJst)

#------------------------------------------------------------------------------
if __name__ == "__main__":

    if not len(sys.argv) in [4,6]:
        sys.exit('Missing argument! usage: ./j-integral-cohesive-zone.py sname/sim-id start_time end_time')

    sim = str(sys.argv[1])
    start_time = float(sys.argv[2])
    end_time = float(sys.argv[3])

    dflt_avg = 0.005

    if len(sys.argv) == 4:
        plot_j_integral([sim],start_time=start_time,end_time=end_time,
                        avg_dist=dflt_avg)
    else:
        pvs = str(sys.argv[4])
        cf_norm = str(sys.argv[5])
        plot_j_integral([sim],start_time=start_time,end_time=end_time,
                        plot_vs=pvs,norm_speed=cf_norm,
                        avg_dist=dflt_avg)
        

    plt.show()
