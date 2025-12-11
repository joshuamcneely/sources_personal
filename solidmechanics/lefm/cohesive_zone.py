# cohesive_zone.py
#
# cohesive zone sizes
# 
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     2015/10/01
# @modified 2020/12/21

from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.integrate import quad

from ..definitions import smd
from .utilities import complex_quadrature, \
    my_integralSingular, check_cohesive_zone_f
from .functions import compute_alphas, compute_D, compute_AII

def compute_Xc(matprop,iface,v):
    assert(v>=0 and v<matprop[smd.cp])
    check_cohesive_zone_f(iface)
    if v< matprop[smd.cR]:
        return compute_Xc_subRayleigh(matprop,iface,v)
    elif v>=matprop[smd.cs]:
        return compute_Xc_intersonic(matprop,iface,v)
    else:
        raise RuntimeError('invalid crack propagation velocity v: cR < v < cs')

def compute_Xc_subRayleigh(matprop,iface,v):
    """Compute cohesive zone size at given speed.

    From Freund 1990, eq 6.2.35

    Args:
       matprop(dict): material properties
       iface(dict): interface properties
       v(float): crack tip speed 
    
    Returns:
       float
    """
    Xc0 = compute_Xc0(matprop,iface)
    alpha_s, alpha_d = compute_alphas(v,matprop)    
    D = compute_D(alpha_s,alpha_d)
    AII = compute_AII(v,alpha_s,D,matprop)
    return Xc0/AII


def compute_Xc0(matprop,iface):
    """Compute the cohesive zone size of a static crack.

    Linear: Palmer and Rice 1973.
    General: Freund 1990, eq. 6.2.34,35
    
    Args:
       matprop(dict): material properties
       iface(dict): interface properties
           Gamma (float)
           cohesive_zone_f (function)
          
    Returns:
       Xc0(float): end zone size
    
    """

    E = float(matprop[smd.E])
    nu = float(matprop[smd.nu])
    if smd.Gamma in iface and smd.tauc in iface and smd.taur in iface:
        Gamma = iface[smd.Gamma]
        tau_p = iface[smd.tauc]-iface[smd.taur]
        dc = Gamma*2./tau_p
    elif smd.mus in iface and smd.muk in iface and smd.sigi in iface and smd.dc in iface:
        tau_p = (iface[smd.mus]-iface[smd.muk])*iface[smd.sigi]
        dc = iface[smd.dc]
        Gamma = tau_p*dc*0.5
    else:
        print(iface)
        raise RuntimeError('not enough interface information')
        
    if matprop.is_plane_strain():
        Eeq = E/(1. -nu**2)
    else:
        Eeq = E

    if not smd.cohesive_zone_f in iface:
        # print('Assume linear cohesive zone')
        # Palmer and Rice 1973
        Xc0 = 9. * np.pi / 32. * Eeq / 2. * dc / tau_p
    else:
        # Freund 1990 eq. 6.2.32 6.2.35
        ff = lambda x: iface[smd.cohesive_zone_f](x) / np.sqrt(np.abs(x))
        Xc0 = np.pi * Eeq * Gamma / 2.0 *(tau_p * quad(ff,-1,0)[0])**-2
    return Xc0


def compute_Xc_intersonic(matprop,iface,v):
    """Compute cohesive zone size at given superhsear speed

    Broberg p354 6.3.70

    Args:
       matprop(dict): material properties

       iface(dict): interface properties
          Gamma (float)
          cohesive_zone_f (function) e.g. lambda x: 1+x s.t. x in [-1,0]

       v(float): crack tip speed 
       tauc_taur (float): tau_peak - tau_res
    Returns:
       float
    """
    assert(v>matprop[smd.cs])

    tau_p = iface[smd.tauc]-iface[smd.taur]
    
    G = iface[smd.Gamma]
    mu = matprop[smd.mu]
    
    k = matprop[smd.cs]/matprop[smd.cp]
    b = v/matprop[smd.cp] # p332
    ap = (1-b**2)**0.5
    bs = ((b/k)**2-1)**0.5
    g = 1.0/np.pi*np.arctan(4*ap*bs/(bs**2-1)**2)
    YII = (1-k**2) * b**2 * np.sin(np.pi*g)/(2* k**2 * (1-b**2)**0.5)

    dz = 1e-3

    z = np.arange(dz,1,dz) # the integral of I is singular at z=1;

    # 6.3.70 has two internal integrals, named here by I1 and I2

    if not iface.get(smd.cohesive_zone_f):
        # Linear cohesive zone
        # for linear cohesive zone (otherwise this hould be an integral)
        # I1=1/g;
        I1 = 1.0/g
        tau = lambda x: 1+x
    else:
        # calculate I1 for general cohesive zone
        # Need to check accurecy at various Cf
        tau = iface[smd.cohesive_zone_f]
        ILowerBoundary = -1.0
        I1 = np.zeros(len(z),dtype=np.complex);

        for j in range(len(z)):
            I1[j] = calcI1sym(tau,ILowerBoundary,-z[j],g);

    #calculate I2
    I2 = np.zeros(len(z),dtype=np.complex);
    for j in range(len(z)):
        I2_integrand = lambda w: 1.0 /(w-z[j])/((w)**(1-g))
        I2[j] = complex_quadrature(I2_integrand,1,np.inf)[0] # behaves different than matlab integrate difference 1e-4
 
    wd_integrand = tau(-z) * z**(1-g) * (I1 + tau(-z)*I2);
    
    wd = np.trapz(wd_integrand,dx=dz);

    xc = G*mu/tau_p**2 * (np.pi*(1-k**2))/(YII*np.sin(np.pi*g)*wd); #p.354 6.3.69

    return xc.real

def calcI1sym(tau,ILowerBoundary,z,g):
    """
    Args:
        tau (function)
        ILowerBoundary (float)
        z (float)
        g (float)
    """
    # Integral is done in non dimensional coordinats

    #g=g+0j
    
    I = lambda w: -(tau(z)- tau(w))/(w-z)/((-w)**(1-g));

    #------Decomposition to sym and Asym
    v = lambda w: 1/2*( I(w)+ I(2*z-w));
    #h=@(w) 1/2*( I(w)- I(2*z-w));
    l=min(z-ILowerBoundary,0-z);

    #ILowerBoundary = ILowerBoundary+0j
    #z=z+0j

    #I1=integral(v,z-l/2,z+l/2)+integral(I,-1,z-l/2)+integral(I,z+l/2,0,'RelTol',1e-3,'AbsTol',1e-5);
    #I1=integral(I,-1,z-l/2)+2*integral(v,z-l/2,z)+my_integralSingular(I,1-g,z+l/2,0);
    I1 = complex_quadrature(I,ILowerBoundary,z-l/2)[0] + 2*complex_quadrature(v,z-l/2,z)[0] + my_integralSingular(I,1-g,z+l/2,0)[0];

    #I1=( I1-1j*pi*tau(z)/(-z)^(1-g) );%Second part in (...) needed for Principal value.
    return I1
#---



def compute_Xc0_intersonic_old(setup_prop):
    """Compute the end zone size of a shear crack propagating at an intersonic velocity.

    See "Broberg 1989"

    Note: the plane stress case is not implemented yet.
    
    Args:
       setup_prop(dict): loading and boundary conditions
    
    Returns:
       Xc0_int(float): end zone size
    
    Raises:
       RuntimeError: if plane stress material

    """
    E = float(setup_prop[smd.E])
    nu = float(setup_prop[smd.nu])
    if smd.tauc in setup_prop.keys():
        tauc = float(setup_prop[smd.tauc])
        taur = float(setup_prop[smd.taur])
        Gamma = float(setup_prop[smd.Gamma])
    else:
        tauc = float(setup_prop[smd.mus])*float(setup_prop[smd.sigi])
        taur = float(setup_prop[smd.muk])*float(setup_prop[smd.sigi])
        Gamma = compute_Gamma(setup_prop)


    if mat.is_plane_strain(setup_prop):
        Xc0_int =  Gamma * E / ( 2. * (1. + nu) * (tauc - taur)**2)
    else:
        raise RuntimeError('not implemented')
        
    return Xc0_int


def test_calcI1sym():
    tau = lambda x:1+x
    z=-0.2
    g=0.5
    I = calcI1sym(tau,-1,z,g)
    if abs(I-2.0):
        print(np.abs(I - 2.0))
        raise RuntimeError

def test_supershear_Xc():
    from ..linearelasticity import LinearElasticMaterial as lem
    import warnings
    with warnings.catch_warnings(): # to avoid printing warning for incomplete material
        warnings.simplefilter("ignore")

        nu=0.33
        mat = lem({smd.nu : nu,
                   smd.pstress : True,
                   smd.mu : 1.,
                   smd.cp :1.,
                   smd.cs : ((1-2*nu)/(2*(1-nu)))**0.5})

    iface = {smd.Gamma: 1.0,
             smd.tauc:1.0,
             smd.taur:0.0,
             smd.cohesive_zone_f : lambda x:1+x}

    xc = compute_Xc_intersonic(mat, iface, mat[smd.cp]*0.8)

    correct=1.723
    err = abs((xc-correct)/correct)
    if err > 1e-4:
        print(xc)
        raise RuntimeError('supershear cohesive zone wrong.')



if (__name__ == '__main__'):
    print('unit tests: lefm cohesive zone')
    test_calcI1sym()
    test_supershear_Xc()
    print('success!')
