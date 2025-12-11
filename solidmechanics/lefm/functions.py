# functions.py
#
# functions that are used for various things in LEFM
# 
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     2015/10/01
# @modified 2020/12/11

from __future__ import print_function, division, absolute_import

import numpy as np
#import math
import cmath
#import os
import collections

from ..definitions import smd
from ..constitutive_law import ConstitutiveLaw as Material

#from ifasha.solidmechanics.lefm.utilities import *

def compute_alphas(v,matprop):
    """
    Compute the factor Alpha_s and Alpha_d 
    as defined in "Freund 1990, ch. 4.2 & 4.3" 
    or in "Freund J. Geophys. Res., 1979 eq. 5 & 6
    
    Args:
        v(float) : crack tip speed
        matprop(dict) : material properties
    
    Returns:
        alpha_s(float)
        alpha_d(float)
    """
    
    c_p = float(matprop[smd.cp])
    c_s = float(matprop[smd.cs])

    alpha_s = np.sqrt(1 - v**2/c_s**2)
    alpha_d = np.sqrt(1 - v**2/c_p**2)

    return alpha_s, alpha_d


def compute_D(alpha_s,alpha_d):
    """
    Compute the factor D as defined in "Freund 1990, eq. (4.3.8)" 
    or in "Freund J. Geophys. Res., 1979 eq. 8 (R = D)"
    
    Args:
        alpha_s(float) : 
        alpha_d(float) : 
    
    Returns:
        D(float):
    """
    return 4 * alpha_d * alpha_s - (1 + alpha_s**2)**2

# see p. 234 Freund (5.3.11)
def compute_AII(v,alpha_s,D,matprop):
    """Compute the function AII as defined in "Freund 1990, eq. (5.3.11)

    AII is the function linked to the mode II sollicitation. 
    It is an universal function: it does not depend on the loading condition 
    or the configuration of the analyzed body. It depends on the 
    crack tip speed and the material properties.
    
    Args:
        v(float) : crack tip speed 
        alpha_s(float) : see function compute_alphas
        D(float): see function compute_D
        matprop(dict) : material properties
    
    Returns:
        AII(float)
    """
   
    nu = float(matprop[smd.nu])
    c_s = float(matprop[smd.cs])

    # static solution
    if v < 0.00001 * c_s:
        return 1.

    if matprop.is_plane_strain():
        # see p.234 Freund (5.3.11)
        A2 = v**2 * alpha_s / ((1 - nu) * c_s**2 * D)
    else:
        # see p.234 Freund (5.3.11)
        A2 = v**2 * alpha_s * (1 + nu) / (c_s**2 * D)

    return A2

def kappa(matprop):
    """
    Compute kappa value 
    plane strain : 3 - 4*nu
    plane stress : (3 - nu) / (1 + nu)
    
    Args:
      matprop (dict): material properties

    Returns: 
      kappa (float)

    Notes:
    References: Park & Earmme (1986), Suo & Hutchinson (1990)
    equal to eta in Rice & Sih (1965)
    """

    if matprop.is_plane_strain():
        kappa = 3 - 4*matprop[smd.nu]
    else:
        kappa = (3 - matprop[smd.nu]) / (1 + matprop[smd.nu])

    return kappa

def Dundurs_parameters(matprop1,matprop2):
    """
    compute Dundurs' parameters (1969) slightly modified following 
    Suo and Hutchinson (1990) and Hutchinson, Mear & Rice (1987)
    
    Args:
      matprop1 (dict): top material
      matprop2 (dict): bottom material

    Returns:
      alpha, beta: Dundurs' parameters
    """

    kappa1 = kappa(matprop1)
    kappa2 = kappa(matprop2)
    
    # Suo & Hutchinson (1990) eq. (2.1)
    Gamma = matprop1[smd.nu] / matprop2[smd.nu]

    alpha = (Gamma * (kappa2+1) - (kappa1+1)) / (Gamma * (kappa2+1) + (kappa1+1))
    beta  = (Gamma * (kappa2-1) - (kappa1-1)) / (Gamma * (kappa2+1) + (kappa1+1))

    return alpha,beta

def epsilon_bimat_const(matprop1,matprop2):
    """
    Bimaterial constant epsilon for interface between two materials

    Args:
      matprop1 (dict): material property of top material
      matprop2 (dict): material property of bottom material

    Returns:
      epsilon (float): bimaterial constant

    Notes:
      References: 
      Rice & Sih (1965) eq. (6)
      Hutchinson, Mear & Rice (1987) eq. (2.2) & (2.15)
      Suo & Hutchinson (1990) eq. (2.2)
    """
    alpha,beta = Dundurs_parameters(matprop1,matprop2)
    return 1 / (2*np.pi) * np.log((1-beta)/(1+beta))


def compute_S(setup_prop):
    """
    Compute the dimensionless stress change

    Args:
       setup_prop(dict): loading and boundary conditions
    
    Returns:
       S(float)
    """

    taui = float(setup_prop[smd.taui])
    
    if smd.tauc in setup_prop.keys():
        tauc = float(setup_prop[smd.tauc])
        taur = float(setup_prop[smd.taur])
    else:
        tauc = float(setup_prop[smd.mus])*float(setup_prop[smd.sigi])
        taur = float(setup_prop[smd.muk])*float(setup_prop[smd.sigi])

    return (tauc-taui)/(taui-taur)


def compute_kII(v,matprop):
    """
    Compute the stress intensity factor kII in the case of a bi-lateral crack"
    
    The computation of kII is based on 
    "Freund 1990, eq. (6.3.42) , eq.(6.3.65) , eq.(6.3.66) and  eq.(6.3.68)

    Args:
        v(float) : crack tip speed
        matprop(dict) : material properties
    
    Returns:
        kII(float): stress intensity factor
    """
    
    c_d = matprop[smd.cp]
    c_s = matprop[smd.cs]
    
    # Freund's book: page 324 just after eq. 6.3.42
    R = lambda z: (c_s**(-2) - 2*z**2)**2 + 4*z**2 * cmath.sqrt(c_d**(-2) - z**2) * cmath.sqrt(c_s**(-2) - z**2)

    # Freund's book: page 332 eq. 6.3.65 and 6.3.66
    Integrand = lambda z: R(1j*z) * (v**(-2) + z**2)**(-1.5) * (c_s**(-2) + z**2)**(-0.5)
    I = v /c_s**2 / scpreal(complex_quadrature( Integrand, 0, np.inf)[0])

    # Freund's book: page 332 eq. 6.3.68 (but a should be b)
    return -I * R(1/v) * v * c_s**2 / np.sqrt(v**(-2) - (c_s)**(-2))

def compute_gII(speeds,matprop):
    """
    compute g_II(c_f) for bi-laterial crack
    
    Args: 
        speeds (array): crack speeds
        matprop (dict): material properties

    where g_II(c_f) = A_II(c_f) * k_II(c_f)**2
    Freund's book: page 397 eq. 7.4.5
    """

    # in case speeds is not a list
    if not isinstance(speeds, collections.Iterable):
        speeds = [speeds]

    gII = []
    for v in speeds:
        alpha_s, alpha_d = compute_alphas(v, matprop)
        D = compute_D(alpha_s,alpha_d)
        gII.append(compute_AII(v,alpha_s,D,matprop) 
                   * (scpreal(compute_kII(v,matprop)))**2)

    return np.array(gII)


def compute_Gamma(setup_prop):
    """Compute the fracture energy.

    Args:
       setup_prop(dict): loading and boundary conditions
    
    Returns:
       eom(float): array crack length at given speed
    
    Raises:
       RuntimeError: not possible to compute Gamma with given parameters (one is missing)

    """

    if smd.mus in setup_prop and smd.muk in setup_prop and smd.dc in setup_prop and smd.sigi in setup_prop:
        tauc = float(setup_prop[smd.mus])*float(setup_prop[smd.sigi])
        taur = float(setup_prop[smd.muk])*float(setup_prop[smd.sigi])
        dc = float(setup_prop[smd.dc])
        
        return 0.5 * (tauc - taur) * dc
    else:
        print('Does not know how to compute Gamma with given parameters')
        raise RuntimeError


def compute_tau_and_Gamma(iface):
    if smd.tauc in iface.keys():
        #dc = iface[smd.Gamma]*2./(tauc-taur)
        pass
    elif smd.mus in iface.keys():
        iface[smd.tauc] = float(iface[smd.mus])*float(iface[smd.sigi])
        iface[smd.taur] = float(iface[smd.muk])*float(iface[smd.sigi])
        iface[smd.Gamma] = (iface[smd.tauc]-iface[smd.taur])*iface[smd.dc]*0.5
    
    
def compute_critHalfLength(Gamma, delta_tau, **kwargs):
    """Compute the Griffith's half length Lc

    Note: the implemented HalfLength is valid for a central crack. In the case of a lateral crack, the value should be multiply by 0.8.

    Args:
       Gamma(float): fracture energy
       delta_tau(float): stress drop (e.g. applied stress - residual stress)

    Kwargs:
       matprop(dict): material properties for uni-material configuration
       matprop1(dict): material prop for (top) bi-material configuration
       matprop2(dict): material prop for (bottom) bi-material configuration

    Returns:
       Lc(float): critical half length 
    """

    # uni-material set-up
    if 'matprop' in kwargs:
        matprop = kwargs.get('matprop')

        E = float(matprop[smd.E])
        nu = float(matprop[smd.nu])
        
        if matprop.is_plane_strain():
            Lc =  E / ((1.+nu)*(1.-nu)) * Gamma /(np.pi * (delta_tau)**2)
        else:
            Lc = E * Gamma / (np.pi * (delta_tau)**2)

    # bi-material set-up
    elif 'matprop1' in kwargs and 'matprop2' in kwargs:
        matprop1 = kwargs.get('matprop1')
        matprop2 = kwargs.get('matprop2')

        # temporarly length used to compute
        tmpL = kwargs.get('tmpL',0.01)
        
        K1 = K1_static(tmpL, 'bilat_bimat_inf_domain',
                       tauinf=delta_tau,
                       matprop1=matprop1,
                       matprop2=matprop2)
        K2 = K2_static(tmpL, 'bilat_bimat_inf_domain',
                       tauinf=delta_tau,
                       matprop1=matprop1,
                       matprop2=matprop2)
    
        Gstat = static_energy_release_rate_from_sif(K1=K1,
                                                    K2=K2,
                                                    matprop1=matprop1,
                                                    matprop2=matprop2)
        GstatpL = Gstat / tmpL
        Lc = Gamma / GstatpL
        
    else:
        print('cannot compute critical length')
        raise RuntimeError
        
    return Lc
