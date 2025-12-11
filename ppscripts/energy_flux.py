#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
from multiprocessing import Pool, Process

import ifasha.datamanager as idm
import solidmechanics as sm

import ppscripts.postprocess as pp
from ppscripts.save_ruptures import load_ruptures


def dP_dl(sigma,n,dudt):
    """
    d rate of work of tractions / dl '
    dP_dl = sigma_ij  n_j  vel_i

    args:
        sigma: size (2,2,n)
        n: size 2
        dudt: size (2,n)
    """
    return (sigma[0,0]*n[0]+sigma[0,1]*n[1])*dudt[0] +\
            (sigma[1,0]*n[0]+sigma[1,1]*n[1])*dudt[1]

def d2U_dldt(sigma,gradu,crackspeed,n):
    """
    d rate of increse in strain energy /dl
    d2U_dldt = 0.5 sigma_ij u_i,j crackspeed n_x

    args:
        sigma: size (2,2,n)
        gradu: size (2,2,n) // works with infinitesimal strain too
        crackspeed: size 1
    """
    return 0.5*(sigma[0,0]*gradu[0,0] + sigma[0,1]*gradu[1,0] +\
                sigma[1,0]*gradu[0,1] + sigma[1,1]*gradu[1,1])*crackspeed*abs(n[0])

def d2K_dldt(dudt,rho,crackspeed,n):
    """
    d rate of increse in kinetic energy /dl
    d2K_dldt = rho vel_i vel_i crackspeed n_x

    args:
        dudt: size (2,n)
        rho: size (n) or 1 is cst
        crackspeed: size 1
    """
    return 0.5*rho*(dudt[0]*dudt[0] + dudt[1]*dudt[1])*crackspeed*abs(n[0])

def my_integrate(f,x0,x1,dx=None):

    if dx==None:
        return integrate.quad(f,x0,x1)[0]
    else:
        x=np.linspace(x0,x1,abs(x1-x0)/dx)
        y=f(x)
        if len(y.shape)==2:
            y=y[:,0]
        I= integrate.cumtrapz(y,x)[-1] #cumulated value
        #print(x.shape,y.shape,I.shape)
        return I

def integrate_square(f,loop,loop_contd=[],dx=0.0,fasym=None,is_asym=True):
    I = []

    x0,x1,x2,x3 = loop

    # L1 vertical path from x0 to x1
    normal = np.array([-1.0,0.0])
    assert x0[0]==x1[0]
    integrand = lambda tau: f(x0[0],tau,normal)
    I.append(my_integrate(integrand,x0[1],x1[1],dx))

    # L2 horizontal path from x1 to x2
    normal = np.array([0.0,1.0])
    assert x1[1]==x2[1]
    integrand = lambda tau: f(tau,x1[1],normal)
    I.append(my_integrate(integrand,x1[0],x2[0],dx))

    # L3 horizontal path from x2 to x3
    normal = np.array([1.0,0.0])
    assert x2[0]==x3[0]
    integrand = lambda tau: f(x2[0],tau,normal)
    I.append(my_integrate(integrand,x2[1],x3[1],dx))
    
    if fasym and is_asym:
        #print('1 block asym')
        # L1 vertical path from x1 to x0
        normal = np.array([-1.0,0.0])
        assert x0[0]==x1[0]
        integrand = lambda tau: fasym(x0[0],tau,normal)
        I.append(-my_integrate(integrand,x1[1],x0[1],dx)) # because integrate upwards

        # L2 horizontal path from x2 to x1
        normal = np.array([0.0,-1.0])
        assert x1[1]==x2[1]
        integrand = lambda tau: fasym(tau,x1[1],normal)
        I.append(my_integrate(integrand,x2[0],x1[0],dx))

        # L3 horizontal path from x3 to x2
        normal = np.array([1.0,0.0])
        assert x2[0]==x3[0]
        integrand = lambda tau: fasym(x2[0],tau,normal)
        I.append(-my_integrate(integrand,x3[1],x2[1],dx)) # integrate downwards
    

    try:
        x4,x5 = loop_contd
    except:
        pass
    else:
        #print('2 blocks')
        
        # L6 horizontal path from x5 to x0
        normal = np.array([-1.0,0.0])
        assert x5[0]==x0[0]
        integrand = lambda tau: f(x5[0],tau,normal)
        I.append(my_integrate(integrand,x5[1],x0[1],dx))

        # L5 horizontal path from x4 to x5
        normal = np.array([0.0,-1.0])
        assert x4[1]==x5[1]
        integrand = lambda tau: f(tau,x4[1],normal)
        I.append(my_integrate(integrand,x4[0],x5[0],dx))
    
        # L4 vertical path from x3 to x4
        normal = np.array([1.0,0.0])
        assert x3[0]==x4[0]
        integrand = lambda tau: f(x3[0],tau,normal)
        I.append(my_integrate(integrand,x3[1],x4[1],dx))


    return I

            
def gradu_to_stress_intp(x,y,gradu,
                         mat,mat_het=None,hy=np.infty):
    stress = np.zeros(gradu.shape)

    stress = sm.mat.strain_to_stress_2d(gradu,stress=stress)
    if mat_het!=None:
        yhidx = np.argmin(np.abs(y-hy))
        stress[:,:,:,yhidx:] = sm.mat_het.strain_to_stress_2d(gradu,stress=stress)[:,:,:,yhidx:]

    
    s00_intp = interpolate.interp2d(x,y,stress[0,0].T,kind='linear')
    s01_intp = interpolate.interp2d(x,y,stress[0,1].T,kind='linear')
    s10_intp = interpolate.interp2d(x,y,stress[1,0].T,kind='linear')
    s11_intp = interpolate.interp2d(x,y,stress[1,1].T,kind='linear')

    return s00_intp,s01_intp,s10_intp,s11_intp
    

fldp0=idm.FieldId('position',0)
fldp1=idm.FieldId('position',1)

def get_xy_fld(data,fldid,tidx):
    X,Y,fld = data.get_xy_plot(fldp0,fldp1,fldid,tidx)
    print(X.shape,Y.shape,fld.shape)
    x=X[:,0]
    y=Y[0]

    return x,y,fld

def get_intp2d_dif_f(data,fldid,tidx,orig=False):
    
    x,y,fld = get_xy_fld(data,fldid,tidx)
    x,y,fld0 = get_xy_fld(data,fldid,0)
    fld_intp = interpolate.interp2d(x,y,fld.T-fld0.T,kind='linear')
    if orig:
        return x,y,fld_intp,fld-fld0
    else:
        return x,y,fld_intp


def get_intp2d_f(data,fldid,tidx,orig=False):
    
    x,y,fld = get_xy_fld(data,fldid,tidx)
    fld_intp = interpolate.interp2d(x,y,fld.T,kind='linear')
    if orig:
        return x,y,fld_intp,fld
    else:
        return x,y,fld_intp

def add_field(data,bot,top,tidx):
    xb,yb,velb0,velb1=bot
    xt,yt,velt0,velt1=top

    assert(velb0.shape[0]==velt0.shape[0])

    # merge vel field
    totshape = (velb0.shape[0],velb0.shape[1]+velt0.shape[1]-1)
    veltot0 = np.zeros(totshape)
    veltot1 = np.zeros(totshape)
    print(totshape)
    veltot0[:,0:velb0.shape[1]] = velb0[:,:]
    veltot0[:,velb0.shape[1]:] = velt0[:,1:]
    veltot1[:,0:velb0.shape[1]] = velb1[:,:]
    veltot1[:,velb0.shape[1]:] = velt1[:,1:]

    # merge y
    y = np.array(list(yb)+list(yt[1:]))
    x = xt
    return x,y,veltot0,veltot1

def get_intp2d_vel(dma,tidx):
    print('load velocity data')
    data = dma('slider')
  
    x,y,vel0 = get_xy_fld(data,idm.FieldId('velocity',0),tidx)
    x,y,vel1 = get_xy_fld(data,idm.FieldId('velocity',1),tidx)
    
    try:
        data = dma('heterogeneity')
    except:
        pass
    else:
        xh,yh,velh0 = get_xy_fld(data,idm.FieldId('velocity',0),tidx)
        xh,yh,velh1 = get_xy_fld(data,idm.FieldId('velocity',1),tidx)
        x,y,vel0, vel1 = add_field(data,[x,y,vel0,vel1],[xh,yh,velh0,velh1],tidx)

    try:
        data = dma('base')
        addbase=True
    except: 
        pass
    else:
        xb,yb,velb0 = get_xy_fld(data,idm.FieldId('velocity',0),tidx)
        xb,yb,velb1 = get_xy_fld(data,idm.FieldId('velocity',1),tidx)
        x,y,vel0, vel1 = add_field(data,[xb,yb,velb0,velb1],[x,y,vel0,vel1],tidx)


    print('inpt',x.shape,y.shape,vel0.shape)
    vel0_intp = interpolate.interp2d(x,y,vel0.T,kind='linear')
    vel1_intp = interpolate.interp2d(x,y,vel1.T,kind='linear')
    return x,y, lambda eta,tau: np.array([vel0_intp(eta,tau),vel1_intp(eta,tau)])

def get_intp2d_gradu_stress(dma,tidx,**kwargs):
    print('load gradu data')
    input_data = pp.get_input_data(bname,**kwargs)
    mat     = input_data['material-1']
    mat_het = input_data['material-2']

    data = dma('full')
    if True:#False:#
        print('diff field')
        #diffgradu
        xq,yq,gradu_00,gradu_00_orig = get_intp2d_dif_f(data,idm.FieldId('gradu',0),tidx,orig=True)
        xq,yq,gradu_01,gradu_01_orig = get_intp2d_dif_f(data,idm.FieldId('gradu',1),tidx,orig=True)
        xq,yq,gradu_10,gradu_10_orig = get_intp2d_dif_f(data,idm.FieldId('gradu',2),tidx,orig=True)
        xq,yq,gradu_11,gradu_11_orig = get_intp2d_dif_f(data,idm.FieldId('gradu',3),tidx,orig=True)
    else:
        xq,yq,gradu_00,gradu_00_orig = get_intp2d_f(data,idm.FieldId('gradu',0),tidx,orig=True)
        xq,yq,gradu_01,gradu_01_orig = get_intp2d_f(data,idm.FieldId('gradu',1),tidx,orig=True)
        xq,yq,gradu_10,gradu_10_orig = get_intp2d_f(data,idm.FieldId('gradu',2),tidx,orig=True)
        xq,yq,gradu_11,gradu_11_orig = get_intp2d_f(data,idm.FieldId('gradu',3),tidx,orig=True)

                    
    gradu=np.array([[gradu_00_orig, gradu_01_orig],
                    [gradu_10_orig,gradu_11_orig]])
                    
    if 'off_fault_het' in input_data and input_data['off_fault_het']:
        yhstart = input_data['off_fault_het_y_start_position']
        assert input_data['off_fault_het_y_trans_position']>=max(yq)
        assert input_data['off_fault_het_x_start_position']==0
        assert input_data['off_fault_het_x_trans_position']>=max(xq)

        s_00,s_01,s_10,s_11 = gradu_to_stress_intp(xq,yq,gradu,mat,mat_het=mat_het,hy=yhstart)
    else:
        s_00,s_01,s_10,s_11 = gradu_to_stress_intp(xq,yq,gradu,mat)

    gradu = lambda eta,tau: np.array([[gradu_00(eta,tau),gradu_01(eta,tau)],
                                      [gradu_10(eta,tau),gradu_11(eta,tau)]])
    stress = lambda eta,tau: np.array([[s_00(eta,tau),s_01(eta,tau)],
                                       [s_10(eta,tau),s_11(eta,tau)]])
    return xq,yq, gradu, stress

def get_all_tidx(bname,**kwargs):
    wdir = kwargs.get('wdir','./data')
    dma = idm.DataManagerAnalysis(bname,wdir)
    return dma('slider').get_full_field(idm.FieldId('time'))
    
def energy_flux(bname,
                tidx,
                **kwargs):
    """
    compute energy flux using Freund 1979 (16)
    F = int_L (sigma_ij * n_j
    """
    wdir = kwargs.get('wdir','./data')

    x_fct = kwargs.get('x_fct',np.arange(0.3,0.0,-.001))
    y_fct = kwargs.get('x_fct',np.arange(0.3,0.0,-.001))

    avg_dist = kwargs.get('avg_dist',0.005)

    # add plot on ax of figure if already provided
    ax = kwargs.get('ax', None)
    new_figure = True if ax is None else False
    if kwargs.get('no_fig'): new_figure = False 
    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)  

    # ---------------------------------------------------------------------
    
    dma = idm.DataManagerAnalysis(bname,wdir)
    
    input_data = pp.get_input_data(bname,**kwargs)
    mat     = input_data['material-1']
    mat_het = input_data['material-2']

    assert mat['name']=='base' and mat_het['name']=='heterogeneity'

    if tidx==None:
        raise RuntimeError('Error! choose a time step',dma('slider').get_full_field(idm.FieldId('time')))

    # nodal field : velocity
    xn,yn,vel = get_intp2d_vel(dma,tidx)
    
    time_n = dma('slider').get_full_field(idm.FieldId('time'))[tidx]

    # quad point field : gradu
    xq,yq,gradu,stress = get_intp2d_gradu_stress(dma,tidx,**kwargs)

    time_q = dma('full').get_full_field(idm.FieldId('time'))[tidx]
    if abs(time_q-time_n)>1e-9:
        raise RuntimeError('no equal',time_q,time_n)
    time = time_q

    rpts = load_ruptures(bname,**kwargs)
    if len(rpts)>1: print('WARNING: multiple ruptures')

    prop_speed = rpts[0].get_propagation_speed(0,avg_dist).T #[x,t,?,v]
    crack_speed = prop_speed[2,np.argmin(np.abs(prop_speed[1]-time))]
    crack_pos =prop_speed[0,np.argmin(np.abs(prop_speed[1]-time))]
    print('at time',time,'\npos',crack_pos,'\nspeed',crack_speed)

    #-----------------------------------------------------------------------
    print(max(xq),max(yq))
    Lx = max(xq)*x_fct
    Ly = max(yq)*y_fct
    dx=np.diff(xn)[0]
    #print('Lx',Lx,'\nLy',Ly,'\ndx',dx)

    rho = mat['rho']
    assert mat['rho']==mat_het['rho']

    P=[]
    dU_dt=[]
    dK_dt=[]
    lxnew=[]
    lynew=[]

    for lx,ly in zip(Lx,Ly):
        if 2*dx>=lx/2 or 2*dx>=ly/2:
            print('skip')
            continue
        # just to be on grid line
        lx = xn[np.argmin(np.abs(xn-lx))] 
        ly = yn[np.argmin(np.abs(yn-ly))] 
        lxnew.append(lx)
        lynew.append(ly)

        #print('integration region size',lx,ly,dx)
        
        loop = np.array([[max(crack_pos-lx/2.0,0.0)   ,0.0],
                          [max(crack_pos-lx/2.0,0.0)   ,ly/2.0],
                          [min(crack_pos+lx/2.0,max(xn)),ly/2.0],
                          [min(crack_pos+lx/2.0,max(xn)),0.0]])
        loop_contd=[]
        is_asym=True
        if input_data['antisym_setup']==False:
            is_asym=False
            loop_contd = np.array([[min(crack_pos+lx/2.0,max(xn))   ,-ly/2.0],
                                   [max(crack_pos-lx/2.0,0.0),-ly/2.0]])
                                      

        #print('square',*loop)
        stress_a = lambda eta,tau: np.array([[-stress(eta,tau)[0,0], stress(eta,tau)[0,1]],
                                             [ stress(eta,tau)[1,0],-stress(eta,tau)[1,1]]])

        gradu_a = lambda eta,tau: np.array([[-gradu(eta,tau)[0,0], gradu(eta,tau)[0,1]],
                                            [ gradu(eta,tau)[1,0],-gradu(eta,tau)[1,1]]])

        vel_a = lambda eta,tau: np.array([ -vel(eta,tau)[0], vel(eta,tau)[1]])
        
        integrand   = lambda eta,tau,normal: dP_dl(  stress(eta,tau),normal,vel(eta,tau))
        integrand_a = lambda eta,tau,normal: dP_dl(stress_a(eta,tau),normal,vel_a(eta,tau))

        P.append(integrate_square(integrand,loop,loop_contd=loop_contd,dx=dx,fasym=integrand_a,is_asym=is_asym))
        
        integrand   = lambda eta,tau,normal: d2U_dldt(  stress(eta,tau),  gradu(eta,tau),crack_speed,normal) 
        integrand_a = lambda eta,tau,normal: d2U_dldt(stress_a(eta,tau),gradu_a(eta,tau),crack_speed,normal) 

        dU_dt.append(integrate_square(integrand,loop,loop_contd=loop_contd,dx=dx,fasym=integrand_a,is_asym=is_asym))
        
        integrand   = lambda eta,tau,normal: d2K_dldt(  vel(eta,tau),rho,crack_speed,normal)
        integrand_a = lambda eta,tau,normal: d2K_dldt(vel_a(eta,tau),rho,crack_speed,normal)

        dK_dt.append(integrate_square(integrand,loop,loop_contd=loop_contd,dx=dx,fasym=integrand_a,is_asym=is_asym))



    P =     np.array(P)
    dU_dt = np.array(dU_dt)
    dK_dt = np.array(dK_dt)
    
    lxnew = np.array(lxnew)
    lynew = np.array(lynew)
    
    F = P + dU_dt + dK_dt 
    G = F/crack_speed
    for ff,lbl in zip([G, F,P,dU_dt,dK_dt],['F','P','U','K']):
        print(lbl,np.average(ff))
    return G, F,P,dU_dt,dK_dt,lxnew,lynew,crack_speed

def plot_fig(bname,tidx,test=False):
    
    G, F,P,dU_dt,dK_dt,Lx,Ly,cf = energy_flux(bname,tidx,no_fig=True)
    Area=Lx*Ly

    fg,axs = plt.subplots(2,2,sharex=True,sharey=False,figsize=(8,6))
    axes=[axs[0,0],axs[0,1],axs[1,0],axs[1,1]]
    fname='{1}/energy_flux_{0}{1}_{2:0>2}.png'.format("",bname,tidx)
    
    x=Area**.5
    xlabel='$\sqrt{{Area}}$ [m]'
    x=Lx
    xlabel='$L_x$ [m]'
    plot_me([F,P,dU_dt,dK_dt],['$F$','$P$','$\dot U$','$\dot K$'],x,xlabel,fg,axs,axes,fname)

    if test:
        fg,axs = plt.subplots(4,2,sharex=True,sharey=False,figsize=(8,10))
        axes=[axs[0,0],axs[1,0],axs[2,0],axs[3,0],axs[1,1],axs[2,1],axs[3,1]]
        fname='{1}/energy_flux_{0}{1}_{2:0>2}.png'.format('test_',bname,tidx)
        plot_me([F,P,dU_dt,dK_dt],['$F$','$P$','$\dot U$','$\dot K$'],x,xlabel,fg,axs,axes,fname)
        
def plot_me(yfields,ylabels,x,xlabel,fg,axs,axes,fname):
    fg.suptitle(bname)
    for ff,lbl in zip(yfields,ylabels):
        for ax,f,tt in zip(axes[1:],ff.T,['$L_{left}$','$L_{up}$','$L_{right}$','$L_{left}^-$','$L_{up}^-$','$L_{right}^-$']):
            ax.set_title(tt)
            ax.plot(x,f,label=lbl)
        axes[0].set_title('L')

        axes[0].plot(x,[sum(f) for f in ff],label=lbl)

    for ax in axes:

        ax.set_ylabel('Energy Rate [J/s]')
        ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
    for ax in axs[-1,:]:
        ax.set_xlabel(xlabel)
    axes[-1].legend()    
    fg.tight_layout()
    print(fname)
    try:
        #plt.show()
        fg.savefig(fname)
    except:
        plt.show()



def plot_xy(bname,group,tidx,d1,d2):

    dma = idm.DataManagerAnalysis(bname,'data')
    
    if group=='stress':
        x,y,gradu,stress = get_intp2d_gradu_stress(dma,tidx)#,**kwargs)
        x= np.linspace(min(x),max(x),200)
        y= np.linspace(min(y),max(y),200)
        fld = stress(x,y)[d1,d2]
    elif group=="velocity":
        x,y,vel=get_intp2d_vel(dma,tidx)
        x= np.linspace(min(x),max(x),200)
        y= np.linspace(min(y),max(y),200)
        fld = vel(x,y)[d1]
    X,Y= np.meshgrid(x,y)
    print(X.shape,Y.shape, fld.shape)
    fg1=plt.pcolor(X,Y,fld)
    plt.colorbar(fg1)
    plt.show()
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) not in  [2,3,7]:
        sys.exit('Missing argument! usage: ./energy_flux.py bname/sim-id tidx')

    bname = str(sys.argv[1])
    try:
        tidxs  = [int(sys.argv[2])]
    except:
        print('all tidx')
        tidxs = range(len(get_all_tidx(bname)))

    try:
        sys.argv[3]=='xy'
    except:
        for tidx in tidxs:
            plot_fig(bname,tidx,test=True)
    else:
        group=sys.argv[4]
        d1=int(sys.argv[5])
        d2=int(sys.argv[6])
        plot_xy(bname,group,tidxs[0],d1,d2)
        

    '''
        def do_stuff(tidx): 
            plot_fig(bname,tidx)
            return True
        if False:
            pool = Pool(len(tidxs))
            res = pool.map(do_stuff,tidxs)
            print(res)
        else:
    '''
    
        
    
    #-------------------------------------------------------------------------------

