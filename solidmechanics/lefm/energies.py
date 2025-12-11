# energies.py
#
# J-integral
# 
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     2015/10/01
# @modified 2020/12/12

from __future__ import print_function, division, absolute_import

import numpy as np
#import math
#import cmath
#import os
#import collections

from ..definitions import smd
from .functions import kappa, epsilon_bimat_const
#from ..constitutive_law import ConstitutiveLaw
    
from ..cauchystress import CauchyStress
#from ifasha.solidmechanics import InfinitesimalStrain


def static_energy_release_rate_from_sif(**kwargs):
    """
    compute energy release rate using the stress intensity factor
    
    Kwargs:
      K1 (float) : mode I stress intensity factor (SIF)
      K2 (float) : mode II stress intensity factor (SIF)
      K3 (float) : mode III stress intensity factor (SIF)

      matprop (dict): material property for unimaterial config
      matprop1 (dict): material property of top mat for bimaterial config
      matprop2 (dict): material property of bottom mat for bimaterial config

    Returns:
      G (float): energy release rate
    """

    # get stress intensity factors
    K1 = kwargs.get('K1', 0.)
    K2 = kwargs.get('K2', 0.)
    K3 = kwargs.get('K3', 0.)
    
    # unimaterial configuration
    if 'matprop' in kwargs:
        matprop = kwargs.get('matprop')
        E_eff = matprop[smd.E]
        if matprop.is_plane_strain():
            E_eff /= (1-matprop[smd.nu]**2)
        mu = matprop[smd.mu]

        G = K1**2 / E_eff + K2**2 / E_eff + K3**2 / (2*mu)
    # bimaterial configuration
    elif 'matprop1' in kwargs and 'matprop2' in kwargs:
        matprop1 = kwargs.get('matprop1')
        mu1 = matprop1[smd.mu]
        kappa1 = kappa(matprop1)

        matprop2 = kwargs.get('matprop2')
        mu2 = matprop2[smd.mu]
        kappa2 = kappa(matprop2)

        eps = epsilon_bimat_const(matprop1,matprop2)

        # Hutchinson, Mear & Rice (1987) after eq. (2.2)
        K = K1 + 1j * K2

        # Hutchinson, Mear & Rice (1987) after eq. (2.5)
        # Park & Earmme (1986) eq. (12)
        KKbar = K * K.conjugate()
        G =((1+kappa1)/mu1 + (1+kappa2)/mu2) / (16*np.cosh(np.pi*eps)**2) * KKbar
        G = np.real(G)
    else:
        raise RuntimeError

    return G


def get_box_side(XY,xy,YX,yx,**kwargs):
    """
    finds boolean condition matrix for box side. 

    Args:
      XY(float): coordinate for side to find
      xy(array): coordinate map for which side needs to be found
      YX(tuple): min and max coord in other direction to limit box side
      yx(array): coordinate map of other direction

    Kwargs:
      fix_yx(tuple): says whether YX limit is fix or should be adapted

    Return:
      condition(array): boolean array that delimites the side of box in xy and yx

    """

    # find closest xy to value of choice
    xy_unique = np.unique(xy)
    idx = np.argmin(abs(xy_unique-XY))
    XY = xy_unique[idx]

    # find closest yx to value of choice
    yxf = kwargs.get('fix_yx',(False,False))
    yx_unique = np.unique(yx)
    YX_exact = []
    for i,f in zip(YX,yxf):
        if f: # limit is fixed
            YX_exact.append(i)
        else:
            idx = np.argmin(abs(yx_unique-i))
            YX_exact.append(yx_unique[idx])
    YX = YX_exact

    # apply conditions and combine them
    xyb = xy == XY
    yxb0 = yx >= YX[0]
    yxb1 = yx <= YX[1]
    yxb = np.logical_and(yxb0,yxb1)

    return np.logical_and(xyb,yxb)


def get_box(xcoord,ycoord,**kwargs):
    """
    builds boolean map and sort filter to determine a box 
    around a crack tip that is at the origin of the coordinate system.
    it works on a x,y grid

    Args:
      xcoord(array): x coordinate of x,y grid mesh
      ycoord(array): y coordinate of x,y grid mesh

    Kwargs:
      l(float): position of left box side
      r(float): position of right box side
      b(float): position of bottom box side
      t(float): position of top box side

    Return:
      normals(list): list of normals for each side of box
      bmaps(list of arrays): + bool maps used to filter x and y for each side 
                             + sort slides to sort coordinates for integration 

    Notes:
      Yes, one could also return the x,y coordinates for each side. However, that would not work for analyzing simulation/experimental data as x,y will be given.
      Can be used on a grid created by x,y = np.meshgrid(...)
    """
    
    l=kwargs.get('l')
    r=kwargs.get('r')
    b=kwargs.get('b')
    t=kwargs.get('t')

    normals = []
    bmaps = []

    # left side
    normal = [-1,0]
    bmap = get_box_side(l,xcoord,(b,t),ycoord)
    sort = ycoord[bmap].argsort()
    normals.append(normal)
    bmaps.append((bmap,sort))

    # bottom side
    normal = [0,-1]
    bmap = get_box_side(b,ycoord,(l,r),xcoord)
    sort = xcoord[bmap].argsort()
    normals.append(normal)
    bmaps.append((bmap,sort))

    # right side
    normal = [1,0]
    bmap = get_box_side(r,xcoord,(b,t),ycoord)
    sort = ycoord[bmap].argsort()
    normals.append(normal)
    bmaps.append((bmap,sort))

    # top side
    normal = [0,1]
    bmap = get_box_side(t,ycoord,(l,r),xcoord)
    sort = xcoord[bmap].argsort()
    normals.append(normal)
    bmaps.append((bmap,sort))

    return normals,bmaps

def get_top_half_box(xcoord,ycoord,**kwargs):
    """
    builds boolean map and sort filter to determine a top half of a box 
    around a crack tip that is at the origin of the coordinate system.
    it works on a x,y grid

    Args:
      xcoord(array): x coordinate of x,y grid mesh
      ycoord(array): y coordinate of x,y grid mesh

    Kwargs:
      l(float): position of left box side
      r(float): position of right box side
      t(float): position of top box side

    Return:
      normals(list): list of normals for each side of box
      bmaps(list of arrays): + bool maps used to filter x and y for each side 
                             + sort slides to sort coordinates for integration 

    Notes: see also get_box() function
      Can be used on a grid created by x,y = np.meshgrid(...)
    """

    l=kwargs.get('l')
    r=kwargs.get('r')
    t=kwargs.get('t')
    
    normals = []
    bmaps = []

    # right side
    normal = [1,0]
    bmap = get_box_side(r,xcoord,(0.,t),ycoord,fix_yx=(True,False))
    sort = ycoord[bmap].argsort()
    normals.append(normal)
    bmaps.append((bmap,sort))

    # top side
    normal = [0,1]
    bmap = get_box_side(t,ycoord,(l,r),xcoord)
    sort = xcoord[bmap].argsort()
    normals.append(normal)
    bmaps.append((bmap,sort))

    # left side
    normal = [-1,0]
    bmap = get_box_side(l,xcoord,(0.,t),ycoord,fix_yx=(True,False))
    sort = ycoord[bmap].argsort()
    normals.append(normal)
    bmaps.append((bmap,sort))

    return normals,bmaps

def get_bot_half_box(xcoord,ycoord,**kwargs):
    """
    builds boolean map and sort filter to determine a bottom half of a box 
    around a crack tip that is at the origin of the coordinate system.
    it works on a x,y grid

    Args:
      xcoord(array): x coordinate of x,y grid mesh
      ycoord(array): y coordinate of x,y grid mesh

    Kwargs:
      l(float): position of left box side
      r(float): position of right box side
      b(float): position of bottom box side

    Return:
      normals(list): list of normals for each side of box
      bmaps(list of arrays): + bool maps used to filter x and y for each side 
                             + sort slides to sort coordinates for integration 

    Notes: see also get_box() function
      Can be used on a grid created by x,y = np.meshgrid(...)
    """

    l=kwargs.get('l')
    r=kwargs.get('r')
    b=kwargs.get('b')
    
    normals = []
    bmaps = []

    # left side
    normal = [-1,0]
    bmap = get_box_side(l,xcoord,(b,0.),ycoord,fix_yx=(False,True))
    sort = ycoord[bmap].argsort()
    normals.append(normal)
    bmaps.append((bmap,sort))

    # bottom side
    normal = [0,-1]
    bmap = get_box_side(b,ycoord,(l,r),xcoord)
    sort = xcoord[bmap].argsort()
    normals.append(normal)
    bmaps.append((bmap,sort))

    # right side
    normal = [1,0]
    bmap = get_box_side(r,xcoord,(b,0.),ycoord,fix_yx=(False,True))
    sort = ycoord[bmap].argsort()
    normals.append(normal)
    bmaps.append((bmap,sort))

    return normals,bmaps

def j_integral_box_side(normal,xy,matprop,**kwargs):
    """
    Computes line integral for static J-integral 
    along straight vertical or horizontal path

    Args:
      normal(tuple): outward pointing normal of side
      xy(array): coordinates of points in x or y direction depending on normal
      matprop(dict): dict with material properties, i.e. smd.Gamma

    Kwargs:
      sig11(array): array of stress sigma_xx 
      sig22(array): array of stress sigma_yy 
      sig12(array): array of stress sigma_xy 

      eps11(array): array of strain epsilon_xx 
      eps22(array): array of strain epsilon_yy 
      eps12(array): array of strain epsilon_xy 

      du1dx(array): array of x-gradient of displacement in x
      du2dx(array): array of x-gradient of displacement in y

      du1dt(array): array of particle velocity in x
      du2dt(array): array of particle velocity in y

      v(float): rupture velocity

    Return:
      J integral value along path(float)

    Notes:
      - static and dynamic solution are equal for v=0
      - orientation of integration along path is not explicitly coded
        because two effects eliminate each other
        ex. integral along top side: int_right^left f(x) ds
        path: x = C-s thus: dx/ds = -1 thus: ds = -dx
        want to compute integral with increasing x thus need to swich int bound
        int_right^left f(x) ds = - int_left^right f(x) ds
        combined: int_right^left f(x) ds = int_left^right f(x) dx
    """


    sig11 = kwargs.get('sig11',None)
    sig22 = kwargs.get('sig22',None)
    sig12 = kwargs.get('sig12',None)

    eps11 = kwargs.get('eps11',None)
    eps22 = kwargs.get('eps22',None)
    eps12 = kwargs.get('eps12',None)

    du1dx = kwargs.get('du1dx',None)
    du2dx = kwargs.get('du2dx',None)

    du1dt = kwargs.get('du1dt',None)
    du2dt = kwargs.get('du2dt',None)

    v = kwargs.get('v',0.) # rupture speed

    dJ = []
    for i in range(len(xy)):

        # can provide strain or stress to compute J-integral
        if 'sig11' in kwargs:
            stress = CauchyStress([[sig11[i],sig12[i]],
                                   [sig12[i],sig22[i]]])
            strain = matprop.stress_to_strain(stress)

        elif 'eps11' in kwargs:
            strain = InfinitesimalStrain([[eps11[i],eps12[i]],
                                          [eps12[i],eps22[i]]])
            stress = matprop.strain_to_stress(strain)

        else:
            print('provide stress (sig11,...) or strain (eps11,...)')
            raise 

        # strain energy density
        W = matprop.strain_energy_density(strain)

        # surface traction on path
        T = stress.surface_traction(normal)
      
        # static solution
        if v < 1e-7 * matprop[smd.cR]:
            if du1dx is None or du2dx is None:
                raise RuntimeError('need to provide du1dx and du2dx')
            dudx = np.array([du1dx[i],du2dx[i]])
            # Rice 1968, eq. 2
            dJ.append(W*normal[0] - np.dot(T,dudx))

        # dynamic solution
        else:
            if du1dt is None or du2dt is None:
                raise RuntimeError('need to provide du1dt and du2dt')
            dudt = np.array([du1dt[i],du2dt[i]])

            # Freund 1979, eq. 16, but turn into J = F/v
            # 0.5*sig_ij*u_i,j = W
            # du_i/dt / v = du_i/dt * dt/da 
            #             = du_i/da 
            #             = du_i/dx * dx/da 
            #             = du_i/dx * (-1)
            dJ.append(W*normal[0] 
                      + np.dot(T,dudt) / v 
                      + 0.5*matprop[smd.rho] * np.dot(dudt,dudt) * normal[0])

    return np.trapz(dJ, x=xy)

def j_integral_box_path(normals,sides,x,y,matprop,**kwargs):
    """
    Computes j-integral along path that is a box around crack tip

    Args:
      normals(list): list of box side normals
      sides(list): list of tuples with bool map and sort filter for each side
      x(array): coordinate x of grid around crack tip
      y(array): coordinate y of grid around crack tip
      matprop(dict): material properties, with i.e. smd.Gamma

    Kwargs:
      sig11(array): array of stress sigma_xx 
      sig22(array): array of stress sigma_yy 
      sig12(array): array of stress sigma_xy 

      eps11(array): array of strain epsilon_xx 
      eps22(array): array of strain epsilon_yy 
      eps12(array): array of strain epsilon_xy 

      du1dx(array): array of x-gradient of displacement in x
      du2dx(array): array of x-gradient of displacement in y

      du1dt(array): array of particle velocity in x
      du2dt(array): array of particle velocity in y

      v(float): rupture velocity

      rformat(string): format of return (auto,per-side)

    Return: J integral
      J(float): if auto rformat 
      J(list): list of J values for each side of the box
    """

    sig11 = kwargs.get('sig11',None)
    sig22 = kwargs.get('sig22',None)
    sig12 = kwargs.get('sig12',None)

    eps11 = kwargs.get('eps11',None)
    eps22 = kwargs.get('eps22',None)
    eps12 = kwargs.get('eps12',None)

    du1dx = kwargs.get('du1dx',None)
    du2dx = kwargs.get('du2dx',None)

    du1dt = kwargs.get('du1dt',None)
    du2dt = kwargs.get('du2dt',None)

    v = kwargs.get('v',0.) # rupture speed

    if 'sig11' in kwargs:
        seformat = 'stress-based'
    else:
        seformat = 'strain-based'

    if 'du1dx' in kwargs:
        uformat = 'dudx-based'
    else:
        uformat = 'dudt-based'

    J = []
    for normal,side in zip(normals,sides):
        if abs(normal[0]):
            coord = y
        elif abs(normal[1]):
            coord = x
        else:
            print("couldn't figure out coordinate")
            raise

        iput = {'v':v}
        
        if seformat == 'stress-based':
            iput['sig11'] = sig11[side[0]][side[1]]
            iput['sig22'] = sig22[side[0]][side[1]]
            iput['sig12'] = sig12[side[0]][side[1]]
        elif seformat == 'strain-based':
            iput['eps11'] = eps11[side[0]][side[1]]
            iput['eps22'] = eps22[side[0]][side[1]]
            iput['eps12'] = eps12[side[0]][side[1]]
        else:
            raise

        if uformat == 'dudx-based':
            iput['du1dx'] = du1dx[side[0]][side[1]]
            iput['du2dx'] = du2dx[side[0]][side[1]]
        elif uformat == 'dudt-based':
            iput['du1dt'] = du1dt[side[0]][side[1]]
            iput['du2dt'] = du2dt[side[0]][side[1]]
        else:
            raise

        Js = j_integral_box_side(normal,coord[side[0]][side[1]],matprop,
                                 **iput)
        J.append(Js)

    rformat = kwargs.get('rformat','auto') # return format
    if rformat == 'auto':
        return np.sum(J)
    elif rformat == 'per-side':
        return J
    else:
        raise RuntimeError('Do not recognize return format "rformat" parameter!')

def j_integral_cohesive_zone(x,**kwargs):
    """
    Computes j-integral along path that is along cohesive zone

    Args:
      x(array): coordinate x of grid around crack tip

    Kwargs:
      t1(array): array of cohesive traction sigma_xy
                 def: t1 = t1_bot = - t1_top
      t2(array): array of cohesive traction sigma_yy
                 def: t2 = t2_bot = - t2_top

      du1dx_top(array): array of x-gradient of displacement in x above coh zone
      du1dx_bot(array): array of x-gradient of displacement in x below coh zone

      du1dt_top(array): array of particle velocity in x above coh zone
      du1dt_bot(array): array of particle velocity in x below coh zone
      du2dt_top(array): array of particle velocity in y above coh zone
      du2dt_bot(array): array of particle velocity in y below coh zone

      rformat(string): format of return (auto,per-side)

    Return: J integral
      J(float): if auto rformat 
      J(list): list of J values for each side of the box [J_bot, J_top]

    """

    rformat = kwargs.get('rformat','auto') # return format: auto, per-side

    t1 = kwargs.get('t1',None)
    t2 = kwargs.get('t2',None)

    du1dx_bot = kwargs.get('du1dx_bot',None)
    du1dx_top = kwargs.get('du1dx_top',None)

    du1dt_bot = kwargs.get('du1dt_bot',None)
    du1dt_top = kwargs.get('du1dt_top',None)
    du2dt_bot = kwargs.get('du2dt_bot',None)
    du2dt_top = kwargs.get('du2dt_top',None)
    v = kwargs.get('v',0.) # rupture speed

    # slip rate: du1dt_bot - du1dt_top
    du1dt = kwargs.get('du1dt',None)
    
    if du1dx_bot is not None and du1dx_top is not None:
        raise RuntimeError('formula probably not correct')
        # Freund 1979, eq. (19) slip for each side
        J_bot = np.trapz(t1*du1dx_bot,x=x)
        J_top = np.trapz(t1*du1dx_top,x=x)
    elif du1dt_bot is not None \
         and du1dt_top is not None \
         and du2dt_bot is not None \
         and du2dt_top is not None and v != 0:
        # Freund book (1990) eq. 5.3.14
        # def: t1 = t1_bot = - t1_top
        # def: t2 = t2_bot = - t2_top
        F_bot = np.trapz(t1*du1dt_bot + t2*du2dt_bot, x=x)
        F_top = - np.trapz(t1*du1dt_top + t2*du2dt_top, x=x)
        J_bot = F_bot / v
        J_top = F_top / v
    elif du1dt is not None and v != 0:
        if reformat == 'per-side':
            raise RuntimeError('cannot per-side with du1dt only')
        F = np.trapz(t1*du1dt, x=x)
        J_bot = 0.5 * F / v
        J_top = 0.5 * F / v
    else:
        print("don't have enough input")
        raise RuntimeError

    if rformat == 'auto':
        return J_bot+J_top
    elif rformat == 'per-side':
        return [J_bot,J_top]
    else:
        raise RuntimeError('Do not recognize return format "rformat" parameter!')


# ----------------------------------------------
# test unimaterial set-up: energy release rate
# ----------------------------------------------
def test_err_unimaterial():
    from ..linearelasticity import LinearElasticMaterial as lem
    mat1 = lem({smd.E : 2.00e9,
                smd.nu : 0.25,
                smd.rho: 1000,
                smd.pstress: True})
    K2 = 1e4
    Gsol = 0.05 # solution

    Gst = static_energy_release_rate_from_sif(K2=K2, matprop=mat1)

    if abs(Gst - Gsol) > 1e-6*Gsol:
        print('unimat G = {}'.format(Gst))
        raise RuntimeError('Failed unimat Gst')


# ----------------------------------------------
# test bimaterial set-up with twice same material: energy release rate
# ----------------------------------------------
def test_err_bimaterial_1():
    from ..linearelasticity import LinearElasticMaterial as lem
    mat1 = lem({smd.E : 2.00e9,
                smd.nu : 0.25,
                smd.rho: 1000,
                smd.pstress: True})
    K1=0
    K2=1e4
    Gsol=0.05 # solution
    
    mat2 = lem({smd.E : 3.00e9,
                smd.nu : 0.3,
                smd.rho: 1000,
                smd.pstress: True})

    Gst = static_energy_release_rate_from_sif(K1=K1,K2=K2,
                                              matprop1=mat1,
                                              matprop2=mat1)

    if abs(Gst - Gsol) > 1e-6*Gsol:
        print(Gst)
        raise RuntimeError('Failed pseudo bimat Gst')

# ----------------------------------------------
# test j-integral for static problem
# ----------------------------------------------
def test_j_static():
    from ..linearelasticity import LinearElasticMaterial as lem
    mat = lem({smd.E : 5.65e9,
               smd.nu : 0.33,
               smd.rho: 1180,
               smd.pstress: True})

    iface = {smd.Gamma : 1.} # solution
    v0 = 0
    
    # area and discretization of interest
    D = 0.01
    N = 401
    dx=dy=2*D/float(N-1)
    x,y = np.meshgrid(np.linspace(-D,D,N),
                      np.linspace(-D,D,N))

    # LEFM solution for singular crack in area of interest
    from .fields import mode_2_subRayleigh_singular_stress_field as sigma_field
    from .fields import mode_2_subRayleigh_singular_displacement_field as disp_field
    import warnings
    with warnings.catch_warnings(): # to avoid printing division by zero warnings
        warnings.simplefilter("ignore")
        sigxx,sigyy,sigxy = sigma_field(v0, mat, iface,
                                        x=x, y=y,
                                        rformat='element-wise')
    ux,uy = disp_field(v0, mat, iface,
                       x=x, y=y,
                       rformat='element-wise')

    # find axis of x direction: used for gradient 
    x_grad = np.gradient(x)
    x_axis = 1 if np.min(x_grad[0]) == np.max(x_grad[0]) == 0 else 0
    
    duxdx = np.gradient(ux,dx,dy)[x_axis]
    duydx = np.gradient(uy,dx,dy)[x_axis]

    ps = np.arange(0.5,1.,0.2) # don't use outer most box because dudx computation is of lower degree
    Js = []
    for p in ps:
        normals,sides = get_box(x,y,l=-p*D,r=p*D,b=-p*D,t=p*D)

        J = j_integral_box_path(normals,
                                sides,
                                x,y,
                                mat,
                                sig11=sigxx,
                                sig22=sigyy,
                                sig12=sigxy,
                                du1dx=duxdx,
                                du2dx=duydx)
        Js.append(J)
    Js = np.array(Js)
        
    rel_error = np.abs((Js-iface[smd.Gamma])/iface[smd.Gamma])
    if (rel_error > 1e-5).any():
        print(rel_error)
        raise RuntimeError('J integral for static solution wrong.')

# ----------------------------------------------
# test j-integral for static problem per side
# ----------------------------------------------
def test_j_static_per_side():
    from ..linearelasticity import LinearElasticMaterial as lem
    mat = lem({smd.E : 5.65e9,
               smd.nu : 0.33,
               smd.rho: 1180,
               smd.pstress: True})

    iface = {smd.Gamma : 1.} # solution
    v0 = 0
    
    # area and discretization of interest
    D = 0.01
    N = 401
    dx=dy=2*D/float(N-1)
    x,y = np.meshgrid(np.linspace(-D,D,N),
                      np.linspace(-D,D,N))

    # LEFM solution for singular crack in area of interest
    from .fields import mode_2_subRayleigh_singular_stress_field as sigma_field
    from .fields import mode_2_subRayleigh_singular_displacement_field as disp_field
    import warnings
    with warnings.catch_warnings(): # to avoid printing division by zero warnings
        warnings.simplefilter("ignore")
        sigxx,sigyy,sigxy = sigma_field(v0, mat, iface,
                                        x=x, y=y,
                                        rformat='element-wise')
    ux,uy = disp_field(v0, mat, iface,
                       x=x, y=y,
                       rformat='element-wise')

    # find axis of x direction: used for gradient 
    x_grad = np.gradient(x)
    x_axis = 1 if np.min(x_grad[0]) == np.max(x_grad[0]) == 0 else 0
    
    duxdx = np.gradient(ux,dx,dy)[x_axis]
    duydx = np.gradient(uy,dx,dy)[x_axis]

    p=0.95
    normals,sides = get_box(x,y,l=-p*D,r=p*D,b=-p*D,t=p*D)
    xy=[y,x,y,x]
    Js=[]
    for side in range(4):
        J = j_integral_box_side(normals[side],
                                xy[side][sides[side][0]][sides[side][1]],
                                mat,
                                sig11=sigxx[sides[side][0]][sides[side][1]],
                                sig22=sigyy[sides[side][0]][sides[side][1]],
                                sig12=sigxy[sides[side][0]][sides[side][1]],
                                du1dx=duxdx[sides[side][0]][sides[side][1]],
                                du2dx=duydx[sides[side][0]][sides[side][1]])
        Js.append(J)
    
    J = np.sum(Js)    
    rel_error = np.abs((J-iface[smd.Gamma])/iface[smd.Gamma])
    if rel_error > 1e-5:
        print(rel_error)
        raise RuntimeError('J integral for static per side solution wrong.')

    

# ----------------------------------------------
# test j-integral for dynamic problem
# ----------------------------------------------
def test_j_dynamic():
    from ..linearelasticity import LinearElasticMaterial as lem
    mat = lem({smd.E : 5.65e9,
               smd.nu : 0.33,
               smd.rho: 1180,
               smd.pstress: True})

    iface = {smd.Gamma : 1.} # solution
    v0 = 0.9 * mat[smd.cR]

    # area and discretization of interest
    D = 0.01
    N = 401
    dx=dy=2*D/float(N-1)
    x,y = np.meshgrid(np.linspace(-D,D,N),
                      np.linspace(-D,D,N))

    # LEFM solution for singular crack in area of interest
    from .fields import mode_2_subRayleigh_singular_stress_field as sigma_field
    from .fields import mode_2_subRayleigh_singular_velocity_field as velo_field
    import warnings
    with warnings.catch_warnings(): # to avoid printing division by zero warnings
        warnings.simplefilter("ignore")
        sigxx,sigyy,sigxy = sigma_field(v0, mat, iface,
                                        x=x, y=y,
                                        rformat='element-wise')

        vx,vy = velo_field(v0, mat, iface,
                           x=x, y=y,
                           rformat='element-wise')

    ps = np.arange(0.5,1.,0.2) # don't use outer most box because dudx computation is of lower degree
    Js = []
    for p in ps:
        normals,sides = get_box(x,y,l=-p*D,r=p*D,b=-p*D,t=p*D)

        J = j_integral_box_path(normals,
                                sides,
                                x,y,
                                mat,
                                v=v0,
                                sig11=sigxx,
                                sig22=sigyy,
                                sig12=sigxy,
                                du1dt=vx,
                                du2dt=vy)
        Js.append(J)
    Js = np.array(Js)
        
    rel_error = np.abs((Js-iface[smd.Gamma])/iface[smd.Gamma])
    if (rel_error > 1e-4).any():
        print(rel_error)
        raise RuntimeError('J integral for dynamic solution wrong.')


if (__name__ == '__main__'):
    print('unit tests: lefm energies')
    test_err_unimaterial()
    test_err_bimaterial_1()
    test_j_static()
    test_j_static_per_side()
    test_j_dynamic()
    print('success!')
