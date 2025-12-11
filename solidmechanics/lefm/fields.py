# fields.py
#
# compute LEFM near-tip fields
# 
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     2015/10/01
# @modified 2020/12/13

from __future__ import print_function, division, absolute_import

import warnings
import numpy as np
import math
from scipy.integrate import quad

from ..definitions import smd
from ..utilities import polar_to_cartesian, cartesian_to_polar
from ..cauchystress import CauchyStress
from ..constitutive_law import ConstitutiveLaw
from .utilities import complex_quadrature, \
    my_integralSingular, check_cohesive_zone_f
from .functions import *
from .cohesive_zone import compute_Xc_subRayleigh, compute_Xc_intersonic

def mode_2_subRayleigh_singular_stress_field(v,matprop,iface,**kwargs):
    """
    Compute the singular stress field for mode II sollicitation 
    on polar coordinates with the subRayleigh solution.
    
    Args:
        v(float) : crack tip speed     
        matprop(dict) : material properties
        iface(dict): Gamma
    kwargs:
       x,y (array): cartesian coordinates centered at crack tip
       r,theta (array): polar coordinate centered at crack tip
       rformat(string): return format (auto,element-wise)

    Returns:
        stresses(depends on rformat): matrix of the stresses

    Notes:
    A solution exists only if the crack tip speed is not higher 
    than the Rayleigh wave speed. In order to compute the stress field, 
    the factor alpha_s, alpha_d and D have to be computed (see compute_alphas 
    and compute_D). The value of the function AII is given by 
    "Freund 1990, eq. (5.3.11)". The stress intensity factor KII is defined 
    by "Freund 1990, eq. (5.3.10)". The factors gamma_d, gamma_s, theta_d 
    and theta_s are computed as in "Freund 1990, eq. (4.3.12)". 
    The angular variations factor and the angular varation functions can 
    be found in "Freund 1990, eq. (4.3.24)". When all these term are computed, 
    the stresses can be determined based on "Freund 1990, eq. (4.3.23)".
    """

    if 'x' in kwargs and 'y' in kwargs:
        x = np.array(kwargs.get('x'))
        y = np.array(kwargs.get('y'))
        # switch to cartesian coordinates
        r, theta = cartesian_to_polar(x,y)
    elif 'r' in kwargs and 'theta' in kwargs:
        r     = np.array(kwargs.get('r'))
        theta = np.array(kwargs.get('theta'))
    else:
        print('provide either cartesian or polar coordinates '
              +'e.g., x=x_array,y=...')
        raise RuntimeError
    
    E   = float(matprop[smd.E])
    nu  = float(matprop[smd.nu])
    rho = float(matprop[smd.rho])
    c_p = float(matprop[smd.cp])
    c_s = float(matprop[smd.cs])

    Gamma = float(iface[smd.Gamma])

    # check if solution is possible:
    c_R = float(matprop[smd.cR])
    if v > c_R:
        warnings.warn('rupture speed > Rayleigh wave speed!')

    alpha_s, alpha_d = compute_alphas(v,matprop)
    D = compute_D(alpha_s, alpha_d)
    A2 = compute_AII(v, alpha_s, D, matprop)
        
    # see p. 234 Freund (5.3.10)
    if matprop.is_plane_strain():
        K2 = math.sqrt( Gamma * E / ((1 - nu**2) * A2) ) 
    else:
        K2 = math.sqrt( Gamma * E / A2)

    # Freund 1990, eq. (2.1.4)
    if v < 0.00001 * c_s:
        E_11 = -np.sin(0.5*theta) * (2 + np.cos(0.5*theta) * np.cos(1.5*theta))
        E_12 =  np.cos(0.5*theta) * (1 - np.sin(0.5*theta) * np.sin(1.5*theta))
        E_22 =  np.sin(0.5*theta) * np.cos(0.5*theta) * np.cos(1.5*theta)

    else:
        # Freund 1990, eq. (4.3.12)
        theta_s = np.arctan2(np.sin(theta),np.cos(theta) / alpha_s)
        theta_d = np.arctan2(np.sin(theta),np.cos(theta) / alpha_d)
        
        # Freund 1990, eq. (4.3.12)
        gamma_s = np.sqrt(1 - (v * np.sin(theta) / c_s)**2)
        gamma_d = np.sqrt(1 - (v * np.sin(theta) / c_p)**2)
        
        # angular variation factors E_11, E_22, E_12, see p. 171 Freund
        fasd = np.sin(0.5 * theta_d)/np.sqrt(gamma_d)
        fass = np.sin(0.5 * theta_s)/np.sqrt(gamma_s)
        facd = np.cos(0.5 * theta_d)/np.sqrt(gamma_d)
        facs = np.cos(0.5 * theta_s)/np.sqrt(gamma_s)
        
        E_11 = - 2 * alpha_s / D * ((1 + 2 * alpha_d**2 - alpha_s**2) * fasd
                                    - (1 + alpha_s**2) * fass)
        
        E_12 = 1. / D * (4 * alpha_d * alpha_s * facd
                         - (1 + alpha_s**2)**2 * facs)
        
        E_22 = 2 * alpha_s * (1 + alpha_s**2) / D * (fasd - fass)
        
    # compute stress
    Sxx = K2 / np.sqrt(2 * math.pi * r) * E_11
    Syy = K2 / np.sqrt(2 * math.pi * r) * E_22
    Sxy = K2 / np.sqrt(2 * math.pi * r) * E_12
    
    # make structure for return
    rformat = kwargs.get('rformat','auto') # return format

    if rformat == 'auto':
        if len(r.shape)==2:
            print('2d array')
            print(r.shape, Sxx.shape)
            stresses = np.zeros((r.shape[0],r.shape[1],2,2))
            print(stresses.shape)
            for i in range(len(r[0])):
                for j in range(len(r[0])):
                    stresses[i,j]=CauchyStress(np.array([[Sxx[i,j],Sxy[i,j]],
                                                         [Sxy[i,j],Syy[i,j]]]))
        else:
            stresses = [CauchyStress(i) for i in np.array([[Sxx,Sxy],
                                                           [Sxy,Syy]]).T]
        return stresses

    elif rformat == 'element-wise':
        return Sxx,Syy,Sxy
    else:
        raise RuntimeError('Do not recognize return format "rformat" parameter!')

def mode_2_subRayleigh_singular_velocity_field(v,matprop,iface,**kwargs):
    """
    Compute the singular velocity field for mode II subRayleigh crack.
    
    Args:
        v(float) : crack tip speed     
        matprop(dict) : material properties
        iface(dict): Gamma
    kwargs:
       x,y (array): cartesian coordinates centered at crack tip
       r,theta (array): polar coordinate centered at crack tip
       rformat(string): return format (auto,element-wise)

    Returns:
        velocities(depends on rformat): arrays of velocities
    """

    if 'x' in kwargs and 'y' in kwargs:
        x = np.array(kwargs.get('x'))
        y = np.array(kwargs.get('y'))
        # switch to cartesian coordinates
        r, theta = cartesian_to_polar(x,y)
    elif 'r' in kwargs and 'theta' in kwargs:
        r     = np.array(kwargs.get('r'))
        theta = np.array(kwargs.get('theta'))
    else:
        print('provide either cartesian or polar coordinates')
        raise RuntimeError
    
    E   = float(matprop[smd.E])
    mu  = float(matprop[smd.mu])
    nu  = float(matprop[smd.nu])
    rho = float(matprop[smd.rho])
    c_p = float(matprop[smd.cp])
    c_s = float(matprop[smd.cs])

    Gamma = float(iface[smd.Gamma])

    # check if solution is possible:
    c_R = float(matprop[smd.cR])
    if v > c_R:
        warnings.warn('rupture speed > Rayleigh wave speed!')

    alpha_s, alpha_d = compute_alphas(v,matprop)
    D = compute_D(alpha_s, alpha_d)
    A2 = compute_AII(v, alpha_s, D, matprop)
        
    # see p. 234 Freund (5.3.10)
    if matprop.is_plane_strain():
        K2 = math.sqrt( Gamma * E / ((1 - nu**2) * A2) ) 
    else:
        K2 = math.sqrt( Gamma * E / A2)

    # Freund 1990, eq. (4.3.12)
    theta_s = np.arctan2(np.sin(theta),np.cos(theta) / alpha_s)
    theta_d = np.arctan2(np.sin(theta),np.cos(theta) / alpha_d)
    
    # Freund 1990, eq. (4.3.12)
    gamma_s = np.sqrt(1 - (v * np.sin(theta) / c_s)**2)
    gamma_d = np.sqrt(1 - (v * np.sin(theta) / c_p)**2)
    
    # angular variation factors as appear in Freund 1990, eq. (4.3.25)
    fasd = np.sin(0.5 * theta_d)/np.sqrt(gamma_d)
    fass = np.sin(0.5 * theta_s)/np.sqrt(gamma_s)
    facd = np.cos(0.5 * theta_d)/np.sqrt(gamma_d)
    facs = np.cos(0.5 * theta_s)/np.sqrt(gamma_s)

    # Freund 1990, eq. (4.3.25)
    E_1 = alpha_s / D  * (2 * fasd - (1 + alpha_s**2) * fass)
    E_2 = - 1 / D * (2 * alpha_d * alpha_s * facd - (1 + alpha_s**2) * facs)

    # Freund 1990, eq. (4.3.25)
    Vx = K2 / np.sqrt(2 * math.pi * r) * v / mu * E_1
    Vy = K2 / np.sqrt(2 * math.pi * r) * v / mu * E_2

    # static solution
    if v < 0.00001 * c_s:
        Vx.fill(0)
        Vy.fill(0)

    # make structure for return
    rformat = kwargs.get('rformat','auto') # return format
    if rformat == 'auto' or rformat == 'element-wise':
        return Vx,Vy
    else:
        raise RuntimeError('Do not recognize return format "rformat" parameter!')

def mode_2_subRayleigh_singular_displacement_field(v,matprop,iface,**kwargs):
    """
    Compute the singular displacement field for mode II subRayleigh crack.
    
    Args:
        v(float) : crack tip speed     
        matprop(dict) : material properties
        iface(dict): Gamma
    kwargs:
       x,y (array): cartesian coordinates centered at crack tip
       r,theta (array): polar coordinate centered at crack tip
       rformat(string): return format (auto,element-wise)

    Returns:
        displacements(depends on rformat): arrays of displacements
    """

    if 'x' in kwargs and 'y' in kwargs:
        x = np.array(kwargs.get('x'))
        y = np.array(kwargs.get('y'))
        # switch to cartesian coordinates
        r, theta = cartesian_to_polar(x,y)
    elif 'r' in kwargs and 'theta' in kwargs:
        r     = np.array(kwargs.get('r'))
        theta = np.array(kwargs.get('theta'))
    else:
        print('provide either cartesian or polar coordinates')
        raise RuntimeError
    
    E   = float(matprop[smd.E])
    mu  = float(matprop[smd.mu])
    nu  = float(matprop[smd.nu])
    rho = float(matprop[smd.rho])
    c_p = float(matprop[smd.cp])
    c_s = float(matprop[smd.cs])

    Gamma = float(iface[smd.Gamma])

    # check if solution is possible:
    c_R = float(matprop[smd.cR])
    if v > c_R:
        warnings.warn('rupture speed > Rayleigh wave speed!')

    alpha_s, alpha_d = compute_alphas(v,matprop)
    D = compute_D(alpha_s, alpha_d)
    A2 = compute_AII(v, alpha_s, D, matprop)
        
    # see p. 234 Freund (5.3.10)
    if matprop.is_plane_strain():
        K2 = math.sqrt( Gamma * E / ((1 - nu**2) * A2) ) 
    else:
        K2 = math.sqrt( Gamma * E / A2)

    # static solution
    if v < 0.00001 * c_s:
        # Anderson, table 2.2
        U =  K2 / (2 * mu) * np.sqrt(r / (2 * math.pi))
        if matprop.is_plane_strain():
            k = 3 - 4*nu
        else:
            k = (3 - nu) / (1 + nu)
        Ux = U * np.sin(0.5*theta) * (k + 1 + 2 * np.cos(0.5*theta)**2)
        Uy = - U * np.cos(0.5*theta) * (k - 1 - 2 * np.sin(0.5*theta)**2)
    else:
        raise RuntimeError('singular disp field not coded for dynamic solution') 

    # make structure for return
    rformat = kwargs.get('rformat','auto') # return format
    if rformat == 'auto' or rformat == 'element-wise':
        return Ux,Uy
    else:
        raise RuntimeError('Do not recognize return format "rformat" parameter!')

# ------------------------------------------------------------------------
def mode_2_bimaterial_singular_stress_field(v,mattop,matbot,**kwargs):
    """

    """

    if 'x' in kwargs and 'y' in kwargs:
        x = np.array(kwargs.get('x'))
        y = np.array(kwargs.get('y'))
    elif 'r' in kwargs and 'theta' in kwargs:
        r     = np.array(kwargs.get('r'))
        theta = np.array(kwargs.get('theta'))
        # switch to cartesian coordinates
        x, y = polar_to_cartesian(r,theta)
    else:
        print('provide either cartesian or polar coordinates '
              +'e.g., x=x_array,y=...')
        raise RuntimeError

    # make sure complex numbers are used:
    x = np.array(x,dtype=np.complex)
    y = np.array(y,dtype=np.complex)

    mu_t = float(mattop[smd.mu])
    mu_b = float(matbot[smd.mu])

    # dynamic stress intensity factor
    K2 = kwargs.get('K2')

    b_t, a_t = compute_alphas(v,mattop)
    b_b, a_b = compute_alphas(v,matbot)
    D_t = compute_D(b_t,a_t)
    D_b = compute_D(b_b,a_b)

    #print('at,bt',a_t,b_t)
    #print('ab,bb',a_b,b_b)

    alpha = - ((1-b_b**2)*(a_b*(1+b_t**2) + a_t*(1+b_b**2))*mu_t + a_t*D_b*(mu_b - mu_t)) / \
            (((1+b_t**2)*(1+b_b**2-2*a_b*b_b) + 2*a_b*b_t*(1-b_b**2))*mu_t + D_b*mu_b)

    beta = (((1+b_b**2)*(1+b_t**2-2*a_t*b_t) + 2*a_t*b_b*(1-b_t**2))*mu_b + D_t*mu_t) / \
           (((1+b_t**2)*(1+b_b**2-2*a_b*b_b) + 2*a_b*b_t*(1-b_b**2))*mu_t + D_b*mu_b)

    gamma = - ((1-b_t**2)*(a_t*(1+b_b**2)+a_b*(1+b_t**2))*mu_b + a_b*D_t*(mu_t - mu_b)) / \
            (((1+b_t**2)*(1+b_b**2-2*a_b*b_b) + 2*a_b*b_t*(1-b_b**2))*mu_t + D_b*mu_b)

    #print(alpha,beta,gamma)
    #print(-(1+b_t**2)/2/b_t,-(1+b_b**2)/2/b_b )

    z_a_t = x + 1j*a_t*y # x + i a_t y
    z_b_t = x + 1j*b_t*y # x + i b_t y

    z_a_b = x + 1j*a_b*y # x + i a_t y
    z_b_b = x + 1j*b_b*y # x + i b_t y

    z_i_a_t = 1/np.sqrt(z_a_t)
    z_i_b_t = 1/np.sqrt(z_b_t)
    z_i_a_b = 1/np.sqrt(z_a_b)
    z_i_b_b = 1/np.sqrt(z_b_b)

    E_11_t = (1 - b_t**2 + 2*a_t**2) * z_i_a_t.imag \
             + 2 * b_t * alpha * z_i_b_t.imag
    E_22_t = (1 + b_t**2) * z_i_a_t.imag \
             + 2 * b_t * alpha * z_i_b_t.imag
    E_12_t = 2 * a_t * z_i_a_t.real \
             + (1 + b_t**2) * alpha * z_i_b_t.real

    E_11_b = (1 - b_b**2 + 2*a_b**2) * beta * z_i_a_b.imag \
             + 2 * b_b * gamma * z_i_b_b.imag
    E_22_b = (1 + b_b**2) * beta * z_i_a_b.imag \
             + 2 * b_b * gamma * z_i_b_b.imag
    E_12_b = 2 * a_b * beta * z_i_a_b.real \
             + (1 + b_b**2) * gamma * z_i_b_b.real

    prefactor = K2 / np.sqrt(2*math.pi) / (2*a_t + (1 + b_t**2) * alpha)

    Sxx_t = prefactor * E_11_t
    Syy_t = - prefactor * E_22_t
    Sxy_t = prefactor * E_12_t

    Sxx_b = prefactor * E_11_b
    Syy_b = - prefactor * E_22_b
    Sxy_b = prefactor * E_12_b

    Sxx = np.zeros_like(Sxx_t)
    Syy = np.zeros_like(Syy_t)
    Sxy = np.zeros_like(Sxy_t)

    # combined solutions
    tcond = y >= 0
    np.copyto(Sxx, Sxx_t, where=tcond)
    np.copyto(Syy, Syy_t, where=tcond)
    np.copyto(Sxy, Sxy_t, where=tcond)

    bcond = y < 0
    np.copyto(Sxx, Sxx_b, where=bcond)
    np.copyto(Syy, Syy_b, where=bcond)
    np.copyto(Sxy, Sxy_b, where=bcond)

    # make structure for return
    rformat = kwargs.get('rformat','auto') # return format

    if rformat == 'auto':
        if len(r.shape)==2:
            print('2d array')
            print(r.shape, Sxx.shape)
            stresses = np.zeros((r.shape[0],r.shape[1],2,2))
            print(stresses.shape)
            for i in range(len(r[0])):
                for j in range(len(r[0])):
                    stresses[i,j]=CauchyStress(np.array([[Sxx[i,j],Sxy[i,j]],
                                                         [Sxy[i,j],Syy[i,j]]]))
        else:
            stresses = [CauchyStress(i) for i in np.array([[Sxx,Sxy],
                                                           [Sxy,Syy]]).T]
        return stresses

    elif rformat == 'element-wise':
        return Sxx,Syy,Sxy
    else:
        raise RuntimeError('Do not recognize return format "rformat" parameter!')


# ------------------------------------------------------------------------
def mode_2_cohesive_stress_field(v,matprop,iface,**kwargs):
    assert(v>=0 and v<matprop[smd.cp])
    check_cohesive_zone_f(iface)
    if v < matprop[smd.cR]:
        return mode_2_subRayleigh_cohesive_stress_field(v,
                                                        matprop,
                                                        iface,
                                                        **kwargs)
    elif v > matprop[smd.cs]:
        return mode_2_intersonic_cohesive_stress_field(v,
                                                       matprop,
                                                       iface,
                                                       **kwargs)
    else:
        raise RuntimeError('invalid crack propagation velocity v: cR < v < cs')


def mode_2_subRayleigh_cohesive_stress_field(v,matprop,iface,**kwargs):
    """
    subRayleigh steady state solution with general cohesive zone 

    Args:
        v (float): crack propagation velocity [m/s]
        matprop (dict): E,nu,mu,c_s,c_R
        iface (dict): interface properties
            cohesive_zone_f = lambda w: 1;     %Dugdale
            cohesive_zone_f = lambda w: (1+w); %linear weakening cohesive zone in Normlized Units(w=z/Xc)
            cohesive_zone_f = lambda w: 1-(-w).^1.8;
            cohesive_zone_f = lambda w: exp(w/10); exponential
            Note:
                ILowerBoundary = -1 ;%Lower boundary of the integration. cohesive zone size 
                IUpperBoundary = 0 
            smd.Gamma (float)
            smd.tauc (float)
            smd.taur (float)
            smd.Xc (float) [instead of computing it with above variables]
    kwargs:
       x,y (array): cartesian coordinates centered at crack tip
       r,theta (array): polar coordinate centered at crack tip
   
    Returns:
        Stresses [xid,i,j] shape(len(x),2,2)

       
    Notes:
    Based on Ilya's script CrackSolutionGeneralCohesive.m from 2015/10/15 
    (see email)
    Formulas from Samudrala, Huang & Rosakis, JGR 2002 for general case
    Formulas from Poliakov, Dmowska & Rice, JGR 2002 for linear cohesive case
    interface is characterized by: Gamma and Xc

    A solution exists only if the crack tip speed is not higher 
    than the Rayleigh wave speed. In order to compute the stress field, 
    the factor alpha_s, alpha_d and D have to be computed (see compute_alphas 
    and compute_D). The value of the function AII is given by 
    "Freund 1990, eq. (5.3.11)". The stress intensity factor KII is defined by 
    "Freund 1990, eq. (5.3.10)". The factors gamma_d, gamma_s, theta_d and 
    theta_s are computed as in "Freund 1990, eq. (4.3.12)". Formulas for the 
    general case comes from "Samudrala, Huand & Rosakis, JGR 2002". Formulas 
    for the special linear cohesive case rom "Poliakov, Dmowska & Rosakis, 
    JGR 2002" and from "Broberg 1999 eq. (6.2.6) - (6.2.8). Both solutions 
    required to work with complex coordinates.
    """
    if 'x' in kwargs and 'y' in kwargs:
        x = np.array(kwargs.get('x'))
        y = np.array(kwargs.get('y'))
    elif 'r' in kwargs and 'theta' in kwargs:
        print('Polar coordinates r, theta')
        r = np.array(kwargs.get('r'))
        theta = np.array(kwargs.get('theta'))
        # switch to cartesian coordinates
        x, y = polar_to_cartesian(r,theta)
    else:
        print('provide either cartesian or polar coordinates e.g., x=x_array,y=...')
        raise RuntimeError
    
    # make sure complex numbers are used:
    x = np.array(x,dtype=np.complex)
    y = np.array(y,dtype=np.complex)
    
    # make sure y!=0
    if (y==0).all():y+=1e-6; warnings.warn('y==0 diverges, adding epsilon')

    E   = float(matprop[smd.E])
    nu  = float(matprop[smd.nu])
    mu  = float(matprop[smd.mu])
    c_s = float(matprop[smd.cs])

    Gamma = float(iface[smd.Gamma])
        
    # check if solution is possible:
    if v > matprop[smd.cR] or v==0:
        warnings.warn('invalid speed v={}: A0 diverges'.format(v))

    #-----------

    # general LEFM functions
    alpha_s, alpha_d = compute_alphas(v,matprop)
    D = compute_D(alpha_s, alpha_d)
    A2 = compute_AII(v, alpha_s, D, matprop)

    # see p. 234 Freund (5.3.10)
    if matprop.is_plane_strain():
        K2 = math.sqrt( Gamma * E / ((1 - nu**2) * A2) ) 
    else:
        K2 = math.sqrt( Gamma * E / A2)

    # complex variables
    z_d = x + 1j*alpha_d*y
    z_s = x + 1j*alpha_s*y

    # ------ linear cohesive zone solution --------------
    # ------(fast: less numerical intergrations)---------
    if not iface.get(smd.cohesive_zone_f):
        print('linear cohesive zone solution')
        if not smd.Xc in iface:
            R = compute_Xc_subRayleigh(matprop,iface,v)
        else:
            R = iface[smd.Xc]
        
        if iface.get(smd.tauc):
            tau_p = iface[smd.tauc]-iface[smd.taur]
        else:
            G = Gamma
            K = K2
            tau_p = K*np.sqrt(9*math.pi/32./R )
        print('tau_p',tau_p)

        M = lambda z: 2/math.pi * tau_p * ((1+z/R)*np.arctan(1/np.sqrt(z/R))  - np.sqrt(z/R))

        Sxx_tmp = (1+2*alpha_d**2-alpha_s**2)*M(z_d) - (1+alpha_s**2)*M(z_s)
        Syy_tmp = M(z_d) - M(z_s)
        Sxy_tmp = 4*alpha_s*alpha_d*M(z_d) - (1+alpha_s**2)**2 * M(z_s)

        Sxx = 2 * alpha_s / D * Sxx_tmp.imag
        Syy = -2*alpha_s * (1+alpha_s**2)/D * Syy_tmp.imag
        Sxy = 1/D * Sxy_tmp.real


    # ------ general solution -------------- (slow)
    else:
        print('general solution')

        # Eq. 15
        A0 = -1j/mu/np.sqrt(2*math.pi)*2*alpha_s/D*K2

        # Eq. 16c (because Gamma is given instead of computed)
        # e.g., linear cohesive zone: iface[smd.cohesive_zone_f] = lambda x: 1+x

        if not smd.Xc in iface:
            L = compute_Xc_subRayleigh(matprop,iface,v)
        else:
            L = iface[smd.Xc]

        if iface.get(smd.tauc):
            tau_p = iface[smd.tauc]-iface[smd.taur]
        else:
            ff = lambda x: iface[smd.cohesive_zone_f](x) / np.sqrt(np.abs(x))
            tau_p = K2 / np.sqrt(2/math.pi) / (np.sqrt(L) * quad(ff, -1, 0)[0])
        print('tau_p',tau_p)

        fff = lambda x, z: np.sqrt(np.abs(x)) * iface[smd.cohesive_zone_f](x) / (x-z)

        # Eq. 13
        I_d = tau_p * np.sqrt(L)*computeI(fff,z_d/L)
        F   = computeF(z_d,I_d,A0,alpha_s,mu,D)

        # Eq. 11
        I_s = tau_p * np.sqrt(L)*computeI(fff,z_s/L)
        F_s = computeF(z_s,I_s,A0,alpha_s,mu,D)
        G   = computeG(F_s,alpha_s)

        # Broberg 1999 - Eq. 6.2.6 - 6.2.8 - replace x,y by \eta_1,\eta_2
        # Samudarala et al. - Eq. 6a+b
        # compute \partial \Phi / \partial \eta_1 by taking into account
        # \partial z_l / \partial \eta_1
        Sxx = (1 - alpha_s**2 + 2* alpha_d**2) * mu * F + 2*alpha_s*mu*G
        Syy = (1 + alpha_s**2) * mu * F + 2*alpha_s*mu*G
        Sxy = 2*alpha_d*mu*F + (1+alpha_s**2) * mu * G
        Sxx = Sxx.real
        Syy = -Syy.real
        Sxy = -Sxy.imag

    stresses = [CauchyStress(i) for i in np.array([[Sxx,Sxy],[Sxy,Syy]]).T]

    return stresses
    
#---------------
def mode_2_intersonic_cohesive_stress_field(v,matprop,iface,**kwargs):
    """
    The function calculates elastic fields of supershear rupture 
    according to Broberg p348. 
    
    Args:
        v (float): velocity>cs
        matprop (dict): material properties
        iface (dict): interface properties
            cohesive_zone_f = lambda w: 1;     %Dugdale
            cohesive_zone_f = lambda w: (1+w); %linear weakening cohesive zone in Normlized Units(w=z/Xc)
            cohesive_zone_f = lambda w: 1-(-w).^1.8;
            cohesive_zone_f = lambda w: exp(w/10); exponential
            Note:
                ILowerBoundary = -1 ;%Lower boundary of the integration. cohesive zone size 
                IUpperBoundary = 0 
            smd.Gamma (float)
            smd.tauc (float)
            smd.taur (float)
    kwargs:
       x,y (array): cartesian coordinates centered at crack tip
       r,theta (array): polar coordinate centered at crack tip
       CauchyStress (bool): 
       Ux (bool) displacement in x direction *2 for slip, if all(y==0)

    Returns:
        Stresses [xid,i,j] shape(len(x),2,2)
        Ux [xid] shape (len(x)

    Notes:
        Based on Ilya's script Crack_SupershearCohesiveForhNew.m 08/15/2017
        Broberg p348
    """
    assert(v>=matprop[smd.cs])
    if 'x' in kwargs and 'y' in kwargs:
        x = np.array(kwargs.get('x'))
        y = np.array(kwargs.get('y'))
    elif 'r' in kwargs and 'theta' in kwargs:
        print('Polar coordinates r, theta')
        r = np.array(kwargs.get('r'))
        theta = np.array(kwargs.get('theta'))
        # switch to cartesian coordinates
        x, y = polar_to_cartesian(r,theta)
    else:
        print('provide either cartesian or polar coordinates '
              +'e.g., x=x_array,y=...')
        raise RuntimeError

    if (y==0).all() and (np.abs(x)<1e-8).any():
            warnings.warn('integral might not converge for x=0,y=0'
                          +' (suggestion: add epsilon to x)')

    if (np.abs(y)<1e-8).all() and (y!=0).all():
        warnings.warn('integral might not converge for |y|<1e-8'
              +' (suggestion: use y=0 instead)')
    
    mu  = float(matprop[smd.mu])
    c_s = float(matprop[smd.cs])
    c_p = float(matprop[smd.cp])
    
    Gamma = float(iface[smd.Gamma])
    
    tau_p = iface[smd.tauc]-iface[smd.taur]

    #----------Calculations done only on the lower half plane,y<0.
    #If y>0 , calculations done for y<0 and then Sxx and Syy multiplied by -1.
    
    if (y>=0).all():
        LowerHalfPlane=False;
        y=-y; #to make sure that y <0
    elif (y<0).all():
        LowerHalfPlane=True;
    else:
        raise RuntimeError('y must be either lower or upper half plane')
    #----------------

    k = c_s/c_p;#general
    b = v/c_p;#p332
    
    ap = (1-b**2)**0.5;
    bs = ((b/k)**2-1)**0.5;
    alpha = bs/ap;
    y = ap*y; # normlized units eq.6.3.22
    phi_f = arcsin(k/b);
    g = 1.0/np.pi * np.arctan(4*ap*bs/(bs**2-1)**2);
    
    #------Frictional properties
    # Unlike Sub-Rayleigh two frictional properties should be specified.

    tau = iface.get(smd.cohesive_zone_f, lambda z: 1.0 + z)
    ILowerBoundary=-1;#Lower boundary of the integration. cohesive zone size
    
    Xc  = kwargs.get('Xc',compute_Xc_intersonic(matprop, iface, v))
    Xc0 = kwargs.get('Xc0',compute_Xc_intersonic(matprop, iface, 2**0.5*c_s))

    #----x and y should be normlized by Xc
    x = x/Xc;
    y = y/Xc;
    
    #------------Cohesive zone calculation
    F = np.zeros(len(x),dtype=np.complex);
    f = np.zeros(len(x),dtype=np.complex);
    
    for j in range(len(x)):
        #--------calc F_tmp for later calculation of f
        if ((x[j]-alpha*y[j]<0)):
            if (x[j]-alpha*y[j] >= ILowerBoundary):
                #(x[j]-alpha*y)  # to follow the calculation
                Ftmp = calcFsym(matprop,g,ap,tau,ILowerBoundary,x[j]-alpha*y[j]);
            else:
                Ftmp = calcFnumerical(matprop,g,ap,tau,ILowerBoundary,x[j]-alpha*y[j]);
            f[j] = -(bs**2-1)/(4*bs)*(Ftmp+np.conjugate(Ftmp)); #eq.6.3.37
        else:
            f[j]=0;
            Ftmp=0;
        
        #--------calc F eq.6.3.49
        # Symbolic calculation is long. When it is not necessary numerical
        # integration is used.
        if y[j]!=0:
            F[j]=calcFnumerical(matprop,g,ap,tau,ILowerBoundary,x[j]+1j*y[j]);
        elif x[j]>0:
            F[j]=calcFnumerical(matprop,g,ap,tau,ILowerBoundary,x[j]+1j*y[j]);
        else:
            F[j]=Ftmp;

    f=f*tau_p;
    F=F*tau_p;
    #F=F*tau_p*0;% for Mach cone only

    x=x*Xc;
    y=y/ap*Xc;

    Sxx,Syy,Sxy = calcStress(F,f,matprop,ap,bs);
    
    if(LowerHalfPlane==False):
        Sxx=-Sxx;
        Syy=-Syy;
        y=-y;

    stresses = [CauchyStress(i) for i in np.array([[Sxx,Sxy],[Sxy,Syy]]).T]

    if kwargs.get('Ux'):
        assert((y==0).all())
        Uxx=1/(2*mu)/(2*(1-k**2))*(Sxx-(1-2*k**2)*Syy);
        Uyy=1/(2*mu)/(2*(1-k**2))*(Syy-(1-2*k**2)*Sxx);
        Uxy=1/(2*mu)*Sxy;
        #strain=matlaw.stress_to_strain_2d(stress,matprop):
        Ux = np.cumsum(Uxx[-1::-1])*(x[0]-x[1]); # [m] for slip multiply by 2.
        Ux=Ux[-1::-1]
        return stresses, Ux
    else:
        return stresses



def calcFsym(material,g,ap,tau,ILowerBoundary,z):
    """
    Integral is done in non dimensional coordinats
    Broberg eq.6.3.49
    """
    
    # tau_p and rp (Xc) are defined at begining
    tau_p = 1.0;
    rp = 1.0; # (Xc)
    mu = material[smd.mu]
    
    I= lambda w: tau(w)/(w-z)/((-w)**(1-g));

    #------Decomposition to sym and Asym
    v = lambda w: 1.0/2*( I(w)+ I(2*z-w));

    l=min(z-ILowerBoundary,0-z);
    I1 = complex_quadrature(I,ILowerBoundary,z-l/2)[0] + 2*complex_quadrature(v,z-l/2,z)[0] + my_integralSingular(I,1-g,z+l/2,0)[0];

    F = tau_p * rp**(g-1) * (I1 - 1j*np.pi*tau(z)/(-z)**(1-g)); #Second part in (...) needed for Principal value.

    z = z-1e-22j;      #allways work at lower half plane!!!!!!!!
    F = 1j * np.sin(np.pi*g)*z**(1-g)/(2*np.pi*mu*ap)*F;

    return F


def calcFnumerical(material,g,ap,tau,ILowerBoundary,z):
    """Integral is done in non dimentional coordinats
    Broberg eq.6.3.49
    """
    
    #-----Numerical integration
    tau_p = 1.0;
    rp = 1.0; #(Xc)

    mu = material[smd.mu]

    I = lambda w: tau(w)/(w-z)/((-w)**(1-g));
    F = tau_p*rp**(g-1) * my_integralSingular(I,1-g,ILowerBoundary,0)[0];
    
    z = z-1e-22j; #allways work at lower half plane!!!!!!!!
    F = 1j * np.sin(np.pi*g)*z**(1-g)/(2*np.pi*mu*ap)*F;

    return F


def calcStress(F,f,material,ap,bs):
    """
    Broberg eq6.3.31-6.3.33
    """
    mu = material[smd.mu]
    k = material[smd.cs]/material[smd.cp]
    
    Sxy = mu*(1j*ap*(F-np.conjugate(F))+(bs**2-1)*f);
    Syy = mu/2*(  (bs**2-1)*( F+np.conjugate(F) )+4*bs*f  );
    Sxx = mu/2/k**2*(  (1-(1-2*k**2)*ap**2)*(F+np.conjugate(F))-4*k**2*bs*f );

    return Sxx.real, Syy.real, Sxy.real


def arcsin(x):
    """Compute the arcsin

    Notes : for some obscure reasons numpy.arcsin() gives the conjugate of the corresponding Matlabb asin() when the imaginary is zero
   
    Args:
       x(float)

    Return:
       float

    """

    if x.real>1 and x.imag==0:
        return np.conjugate(np.arcsin(x))    
    else:
        return np.arcsin(x)

def computeI(f,z):
    """
    Compute an aproximation of the integral part of the equation (13) as 
    defined in "Samudrala, Huand & Rosakis, JGR 2002", using a 
    complexe quadrature method.
    
    This integral is a part of the function F''(Z), which corresponds to 
    the singular terms of the solution of an inhomogeneous Hilbert problem 
    defined in "Samudrala, Huand & Rosakis, JGR 2002, eq. (12).
    
    Args:
       f(function) : f is a function which depends on the cohesive law
       z(complex) : complex coordinates

    Returns:
       I(float) : approximate integral of f over [0, -L]
    """
    
    I = np.zeros_like(z)
    for i, value in np.ndenumerate(z):
        ft = lambda x: f(x,z[i])
        I[i] = complex_quadrature(ft,-1,0)[0]
        #I[i] = quad(ft,-1,0)[0]
    return I


def computeF(z,I,A0,alpha_s,mu,D):
    """
    Compute the function F''(z) as defined in "Samudrala, Huand & Rosakis, 
    JGR 2002,  eq. (13)".

    The function F''(Z) corresponds to the singular terms of the solution of 
    an inhomogeneous Hilbert problem defined in "Samudrala, Huand & Rosakis, 
    JGR 2002, eq. (12). The expression for F''(z) corresponds to eq. (13). 
    The F function is an analytic function with respect to its complex argument 
    z everywhere in the z plane except on the crack faces. This function is 
    used to describe the stress and displacement field.
    
    Args:
       z(complex) : argument of the function F(). z is a complex
       I(float) : complex quadrature, corresponds to the integral in eq. (13)
       A0(float) : constant defined by eq. (15)
       alpha_s(float) : see compute_alphas
       mu(float): shear modulus
       D(float): see compute_D
    
    Returns:
        F''(float) : value of F''(z)
    """
    
    return  A0 / np.sqrt(z) + 4*alpha_s/mu/D/np.sqrt(z)/(2*math.pi*1j) * I

def computeG(F,alpha_s):
    """
    Compute the function G''(z) as defined in "Samudrala, Huand & Rosakis, 
    JGR 2002, eq. (11)

    The G function is an analytic function with respect to its complex 
    argument z everywhere in the z plane except on the crack faces. This 
    function is used to describe the stress and displacement field.

    Args:
       F(float) : F is a function, see computeF
       alpha_s(float) : see compute_alphas

    Returns:
       G(float) : value of the function G''(z)
    """
    
    return -0.5*(1+alpha_s**2) / alpha_s * F


def test_unimat_lefm_stress():

    from ..linearelasticity import LinearElasticMaterial as lem
    mat = lem({smd.E : 5.65e9,
               smd.nu : 0.33,
               smd.rho: 1180,
               smd.pstress: True})
    iface = {smd.Gamma : 1}
    v0 = 0.1 * mat[smd.cR]

    D = 0.001
    x=np.array([[-D,0,D],[-D,0,D]])
    y=np.array([[-D,-D,-D],[D,D,D]])
    
    # LEFM solution for singular crack in area of interest
    with warnings.catch_warnings(): # to avoid printing division by zero warnings
        warnings.simplefilter("ignore")
        Sxx,Syy,Sxy = mode_2_subRayleigh_singular_stress_field(v0, mat, iface,
                                                               x=x, y=y,
                                                               rformat='element-wise')

    Sxy_sol=np.array([[ 413169.19782752, 332996.23853604, 474878.06735571],
                      [ 413169.19782752, 332996.23853604, 474878.06735571]])
    Sxx_sol=np.array([[ 1215880.53055206, 1007582.73129333,  719793.04100892],
                      [-1215880.53055206,-1007582.73129333, -719793.04100892]])
    Syy_sol=np.array([[ 260882.96471869, 336816.83149092,-108710.35355112],
                      [-260882.96471869,-336816.83149092, 108710.35355112]])
    
    Sxy_error = np.abs((Sxy-Sxy_sol)/Sxy_sol)
    Sxx_error = np.abs((Sxx-Sxx_sol)/Sxx_sol)
    Syy_error = np.abs((Syy-Syy_sol)/Syy_sol)

    if (Sxy_error > 1e-6).any():
        print(Sxy_error)
        raise RuntimeError('unimaterial lefm stress Sxy wrong')
    if (Sxx_error > 1e-6).any():
        print(Sxx_error)
        raise RuntimeError('unimaterial lefm stress Sxx wrong')
    if (Syy_error > 1e-6).any():
        print(Syy_error)
        raise RuntimeError('unimaterial lefm stress Syy wrong')

def test_bimat_lefm_stress_1():
        
    from ..linearelasticity import LinearElasticMaterial as lem
    mat = lem({smd.E : 5.65e9,
               smd.nu : 0.33,
               smd.rho: 1180,
               smd.pstress: True})
    v0 = 0.1 * mat[smd.cR]
    K2=75033.08994469508 # equivalent to Gamma=1 at v0=0.1cR
        
    D = 0.001
    x=np.array([[-D,0,D],[-D,0,D]])
    y=np.array([[-D,-D,-D],[D,D,D]])
    
    # LEFM solution for singular crack in area of interest
    with warnings.catch_warnings(): # to avoid printing division by zero warnings
        warnings.simplefilter("ignore")
        Sxx,Syy,Sxy = mode_2_bimaterial_singular_stress_field(v0, mat, mat, K2=K2,
                                                              x=x, y=y,
                                                              rformat='element-wise')

    Sxy_sol=np.array([[ 413169.19782752, 332996.23853604, 474878.06735571],
                      [ 413169.19782752, 332996.23853604, 474878.06735571]])
    Sxx_sol=np.array([[ 1215880.53055206, 1007582.73129333,  719793.04100892],
                      [-1215880.53055206,-1007582.73129333, -719793.04100892]])
    Syy_sol=np.array([[ 260882.96471869, 336816.83149092,-108710.35355112],
                      [-260882.96471869,-336816.83149092, 108710.35355112]])

    Sxy_error = np.abs((Sxy-Sxy_sol)/Sxy_sol)
    Sxx_error = np.abs((Sxx-Sxx_sol)/Sxx_sol)
    Syy_error = np.abs((Syy-Syy_sol)/Syy_sol)

    if (Sxy_error > 1e-6).any():
        print(Sxy_error)
        raise RuntimeError('bimaterial lefm stress Sxy wrong')
    if (Sxx_error > 1e-6).any():
        print(Sxx_error)
        raise RuntimeError('bimaterial lefm stress Sxx wrong')
    if (Syy_error > 1e-6).any():
        print(Syy_error)
        raise RuntimeError('bimaterial lefm stress Syy wrong')

def test_bimat_lefm_stress_2():
        
    from ..linearelasticity import LinearElasticMaterial as lem
    mat1 = lem({smd.E : 5.65e9,
                smd.nu : 0.33,
                smd.rho: 1180,
                smd.pstress: True})
    mat2 = lem({smd.E : 2.9e9,
                smd.nu : 0.39,
                smd.rho: 1200,
                smd.pstress: True})
    iface = {smd.Gamma : 1}
    v0 = 0.95 * mat2[smd.cR] # faster
    K2=1e6
    
    D = 0.001
    x=np.array([[-D,0,D],[-D,0,D]])
    y=np.array([[-D,-D,-D],[D,D,D]])
    
    # LEFM solution for singular crack in area of interest
    with warnings.catch_warnings(): # to avoid printing division by zero warnings
        warnings.simplefilter("ignore")
        Sxx,Syy,Sxy = mode_2_bimaterial_singular_stress_field(v0, mat1, mat2, K2=K2,
                                                              x=x, y=y,
                                                              rformat='element-wise')

    Sxy_sol=np.array([[ 7582500.17025908,  -934427.22986141,  5760081.32992936],
                      [ 7661839.24061867, -1083606.41478631,  3311637.83528208]])
    Sxx_sol=np.array([[ 27140267.17078058,  20363396.60535048,  14412908.92236248],
                      [-25659286.7717234,  -19580560.93584936, -16317158.04411699]])
    Syy_sol=np.array([[ -1276040.81061447,   3714887.93453552,  -4698704.6318093 ],
                      [-11204823.97647905, -14325821.97522225,   1761517.1411222 ]])

    Sxy_error = np.abs((Sxy-Sxy_sol)/Sxy_sol)
    Sxx_error = np.abs((Sxx-Sxx_sol)/Sxx_sol)
    Syy_error = np.abs((Syy-Syy_sol)/Syy_sol)

    if (Sxy_error > 1e-6).any():
        print(Sxy_error)
        raise RuntimeError('bimaterial lefm stress Sxy wrong')
    if (Sxx_error > 1e-6).any():
        print(Sxx_error)
        raise RuntimeError('bimaterial lefm stress Sxx wrong')
    if (Syy_error > 1e-6).any():
        print(Syy_error)
        raise RuntimeError('bimaterial lefm stress Syy wrong')


def test_supershear_lefm_coh_stress():
    from ..linearelasticity import LinearElasticMaterial as lem
    mat = lem({smd.E : 5.65e9,
               smd.nu : 0.33,
               smd.rho : 1180,
               smd.pstress : True})
    iface={smd.Gamma : 1.12,
           smd.tauc :   1e6,
           smd.taur :   0.0,
           smd.cohesive_zone_f : lambda x:1+x}

    D = 0.005
    x=np.array([-D,0,D])
    y=np.array([D,D,D])
    
    stress = mode_2_intersonic_cohesive_stress_field(mat[smd.cp]*0.8,
                                                     mat, iface,
                                                     x=x, y=y)
    
    [[Sxx,Sxy],[Sxy,Syy]]=np.array(stress).T

    Sxy_sol=np.array([ 203811.19080326,  434697.18234926,  347596.58167062])
    Sxx_sol=np.array([-919204.17514009, -546901.34939688, -160463.28940195])
    Syy_sol=np.array([ 12527.8407875,   18618.97467307,   5462.89001595])

    Sxy_error = np.abs((Sxy-Sxy_sol)/Sxy_sol)
    Sxx_error = np.abs((Sxx-Sxx_sol)/Sxx_sol)
    Syy_error = np.abs((Syy-Syy_sol)/Syy_sol)

    if (Sxy_error > 1e-6).any():
        print(Sxy_error)
        raise RuntimeError('supershear lefm stress Sxy wrong')
    if (Sxx_error > 1e-6).any():
        print(Sxx_error)
        raise RuntimeError('supershear lefm stress Sxx wrong')
    if (Syy_error > 1e-6).any():
        print(Syy_error)
        raise RuntimeError('supershear lefm stress Syy wrong')

    

if (__name__ == '__main__'):
    print('unit tests: lefm fields')
    test_unimat_lefm_stress()
    test_bimat_lefm_stress_1()
    test_bimat_lefm_stress_2()
    test_supershear_lefm_coh_stress()
    print('success!')
