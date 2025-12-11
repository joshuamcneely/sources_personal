#!/usr/bin/env python
"""
crackSolution_SelfSimilarFreund.py

Code to computes Self Similar Solution
e.g. instantaneous acceleration for 0 to cf
cf = cst

There is no warranty for this code

@author Gabriele Alberitni <ga288@cornell.edu>
@date     2016/09/20
@modified 2016/09/20
"""

from __future__ import print_function, division, absolute_import

import numpy as np
from mpmath import ellipe, ellipf

from ..definitions import smd
from .utilities import complex_quadrature
from .functions import *

def complex_ellipf(phi,m):
    """Integrate with complex elliptical integral of first kind
    
    Args:
       phi(complex) :  
       m(float)     : 

    Return:
       complex part of the integral
    """
    
    integral = ellipf(phi,m)
    return np.complex(integral)

def complex_ellipe(phi,m):
    """Integrate with complex elliptical integral of second kind
    
    Args:
       phi(complex) :  
       m(float)     : 

    Return:
       complex part of the integral
    """

    integral = ellipe(phi,m)
    return np.complex(integral)

def heaviside(x):
    """Compute the Heaviside function of x
    
    Args:
       x(float)

    Return:
       float

    """

    return 0.5 * (np.sign(x) + 1)

def arcsin(x):
    """Compute the arcsin.

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
#--------------------------------------------------------

def Rayleigh(z, matprop):
    """ Compute the value of the Rayleigh wave function.

    See Freund, 1990 just after eq. 6.3.42

    Args:
       z(float): 1/c [s/m]
       matprop(dict): material properties

    Returns:
       complex

    """
    cp = matprop[smd.cp]
    cs = matprop[smd.cs]
    z=z+0j
    return (cs**(-2)-2.0*z**2)**2 + 4.*z**2 * (cp**(-2)-z**2)**0.5 * (cs**(-2)-z**2)**0.5  

def compute_D(c, matprop) : # compute D (freund 4.3.8)
    """ Compute the D function, see Freund 1990 eq. 4.3.8

    Args:
       c(float)     : crack tip speed
       matprop(dict): material properties 

    """

    # Due to numerical error R has an imaginary part
    return -c**4.0*np.real(Rayleigh(1.0/c, matprop))  
   
def compute_I_II(z,matprop):
    """ See Freund 1990, eq. 6.3.65 & 6.3.66

    Args:
       z(float): 1/c [m/s]
       matprop(dict): material properties 

    Return:
       float
    
    """
    cs = matprop[smd.cs]
    I_II_integrand = lambda eta: Rayleigh(1.0j*eta,matprop) * (z**2 + eta**2)**-1.5 * (cs**(-2) + eta**2)**-0.5
    return (cs**2*z * complex_quadrature(I_II_integrand, 0., np.inf )[0] )**(-1)

def compute_I_I(z,matprop):
    """ See Freund 1990, eq. 6.3.44

    Args:
       z(float)     : 1/c [s/m]
       matprop(dict): material properties 

    Return:
       float
    """
    cs = matprop[smd.cs]
    cp = matprop[smd.cp]
    I_I_integrand = lambda eta: Rayleigh(1.0j*eta,matprop) * (z**2 + eta**2)**-1.5 * (cp**(-2) + eta**2)**-0.5
    return (cs**2*z * complex_quadrature(I_I_integrand, 0., np.inf )[0] )**(-1)

def K_II_K_II_0(c,matprop):
    """ See Freund 1990, eq. 6.3.68

    Check I_II by computing K_II(t,c)/K_II_0

    Args:
       c(float)     : [m/s]
       matprop(dict): material properties 

    Return:
       float
    """


    # @input: c [m/s]
    # check I_II by computing K_II(t,c)/K_II_0
    # freund (6.3.68) error in book: a should be b
    cs = matprop[smd.cs]
    return -1.0*compute_I_II(1.0/c,matprop)*Rayleigh(1.0/c,matprop)*cs**2*c*(c**-2 - cs**-2)**-0.5

def zeta(x,y,c): 
    """ Freund eq.(6.3.23)*v  heaviside affects only for Gxt. 
    
    Variable change to solve problem. Here real values of v should be used. x , y stand for  x/l y/l  (l=vt).
    
    Args:
       x/l(float) [-]
       y/l(float) [-]
       c(float) = v/cp or v/cs [-]

    Return:
       float

    """
    rsq = (x**2 + y**2)
    ret = x/rsq 
    ret+=  heaviside(1 - rsq*c**2)*1.0j*y/rsq* (1+0j - rsq *c**2)**0.5 
    ret+= -heaviside(-1 + rsq*c**2)*0j
    return  ret
    # "eq.6.3.23" 
    # z=  lambda x,y,v : x/(x**2 + y**2) + 1.0j*y/(x**2 + y**2)* (1 - (x**2 + y**2) *(v)**2)**0.5

def compute_beta_tau_l_G(cf,matprop,*args,**kwargs):
    """ Freund eq.(6.3.66)

    Args:
       cf(float) = crack speed [m/s]
       matprop(dict) = material properties
       l(float) = crack length [m]
       tau_0(float) = Fat field loading [Pa]
       G = Energy release rate [J/m2]

    Return:
       beta(float)
       tau_0(float)
       l(float)
       G(float)
    """
    cs = matprop[smd.cs]
    cp = matprop[smd.cp]
    mu = matprop[smd.mu]

    l     = kwargs.get('l',0)
    tau_0 = kwargs.get('tau_0',0)
    G     = kwargs.get('G',0)
     
    if sum([1 for i in [l, tau_0, G] if i==0]) is not 1:
        print('\nERROR: you have to specify EXACTLY 2 parameters out of [tau_0,L,G]')
        raise RuntimeError()

    #-------------------------------------------------------
    ## Broberg p.330 eq (6.1.1)
    # to get rid of the assumption of plane strain or plane stress
    # plane strain: (1- k**2) = 1/(2*(1-nu))
    k = cs/cp 
    alpha_s, alpha_d = compute_alphas(cf,matprop)

    ## First calc K_II
    # Following Broberg p.334,336

    D = compute_D(alpha_s,alpha_d)
    A_II = compute_AII(cf,alpha_s,D,matprop)
    I_II = compute_I_II(1.0/cf,matprop)

    if G:
        # Broberg (6.2.42)  Freund (5.3.10)
        K_II = (G*mu*4.*(1.-k**2)/A_II)**0.5 

        if l:
            # take singular K_II and l then calculate tau_0
            # Freund eq 6.3.67 (Mistake 6.3.68? see K_II_K_II_0 above)
            tau_0 = -K_II / (I_II*Rayleigh(1.0/cf,matprop) / (cs**-2 * cf**-1 *(cf**-2 - cs**-2)**0.5) *(np.pi*l)**0.5 )   
            tau_0 = np.real(tau_0) # Due to numerical error 

        else:
            # take singular K_II and tau_0 then calculate l
            # Freund eq 6.3.67 (Mistake 6.3.68? see K_II_K_II_0 above)
            l = (-K_II / (I_II*Rayleigh(1.0/cf,matprop) / (cs**-2* cf**-1 *(cf**-2 - cs**-2)**0.5) *tau_0*(np.pi)**0.5 ))**2 
            l = np.real(l) # Due to numerical error 

    if not G:
        # Freund eq 6.3.67 (Mistake 6.3.68? see K_II_K_II_0 above)    
        K_II = -I_II*Rayleigh(1.0/cf,matprop) /(cs**-2* cf**-1 *(cf**-2 - cs**-2)**0.5)* tau_0 * (np.pi*l)**0.5
        # Broberg (6.2.42)  Freund (5.3.10)
        G = K_II**2*A_II/(mu*4.0*(1.0-k**2))
        G = np.real(G)
        
    # freund (6.3.66)
    beta = np.real(tau_0/mu * I_II) # Due to numerical error 

    return beta, tau_0, l, G


def self_similar_solution(v_cR,x,y,matprop, *args,**kwargs):
    """ Symmetric expansion of a shear crack -Freund 6.3.3 p330

    Stress field at a given time / crack length. Use for x>0,y>0. if y<0 Sxy change sign-I don't understand why- probably some branch cut that I haven't taken into account
    
    Args:
       v_cR(float): float norm. crack speed < 1 [-]
       y(float): float off fault distance [m]
       x(float): array of discretised spatial domain [m]. xmin = 0.8*l/v_cR/cR*cs, xmax = l/v_cR/cR*cp, dx = (xmax-xmin)*1e-3
       matprop(dict) : material properties
    
    kwargs: (exactly 2 of them)
       l(float): crack length = vt [m]
       tau_0(float) far field stress [Pa]
       G(float) energy release rate i.e. fracture energy [J/m2]

    Return:
       Stress(float): stress at y Stress[dir,x] dir=[0:xx, 1:yy, 2:xy] [Pa]
       vel(float): [m/s]
       x(float):
       tau_0(float):
       l(float):
       G(float):
   
    """

    # complete material properties to get mu and G and velocities
    cp = matprop[smd.cp]
    cs = matprop[smd.cs]
    cR = matprop[smd.cR]
    mu = matprop[smd.mu]
    
    l     = kwargs.get('l',0)
    tau_0 = kwargs.get('tau_0',0)
    G     = kwargs.get('G',0)
    
    if sum([1 for i in [l, tau_0, G] if i==0]) is not 1:
        print('\nERROR: you have to specify exactly two parameters out of [l,tau_0,G]')
        raise RuntimeError()

    # crack propagation speed
    cf = v_cR*cR
    if cf > cR: # to allow complex parameters
        cf += 0j
        x = np.array(x, dtype=np.complex)
    try: 
        len(x)
        x = np.array(x)
    except: # so that loop over space works even if x is a float
        x = [x]

    beta, tau_0, l, G = compute_beta_tau_l_G(cf,matprop,**kwargs) 

    if kwargs.get('l')==None:
        print("you didn't specify the crack length l\n... rescaling x")
        xmin = 0.8*l/v_cR/cR*cs
        xmax = l/v_cR/cR*cp
        x = (x-(np.min(x)))/(np.max(x)-np.min(x)) #spatial discretisation
        x = (x*(xmax-xmin)+xmin)

    #-------------------------------------------------------

    # The integrals are written in normalized units : z->z*cf, therefore a,b \
    # are also normalized  units. x_i=x_i/(t*cf)

    # calculate the potentials. For F  v/cp should be used,  for G use v/cs
    
    def Fxy( z,v ):
        Fxy_1 = lambda z1,v1: z1 *(1 - z1**2)**0.5 *(-v1**2 + z1**2) \
        - 2.0* v1**2 *(-1 + z1**2)*(1 - z1**2/v1**2)**0.5 * complex_ellipe(arcsin(z1),v1**-2) \
        + v1**2.0* (-1 + z1**2)* (1 - z1**2/v1**2)**0.5* complex_ellipf(arcsin(z1),v1**-2)  
        return  2.0j*Fxy_1(z,v) / ((v**2 - z**2)**0.5*(-1 + z**2)) 

    def Fxx( z,v ):
        return  -2.0*(2 - z**2)/(1 - z**2)**0.5 
    
    def Fyy( z,v ):
        return  -2.0*(-2 + v**2 + z**2)/(1 - z**2)**0.5 
    
    def Gxy( z,v ):
        return  (4 - v**2 - 2.0*z**2)/(1 - z**2)**0.5 

    
    def Gxx( z,v ):
        Gxx_1 = lambda z1,v1:  (-2 + v1**2)*z1* (1 - z1**2)**0.5*(-v1**2 + z1**2) - v1**2* (-4 + 3.0 *v1**2)* (-1 + z1**2)*(1 - z1**2/v1**2)**0.5*complex_ellipe(arcsin(z1),v1**-2) +2.0* v1**2.0* (-1 + v1**2)* (-1 + z1**2) *(1 - z1**2/v1**2)**0.5*complex_ellipf(arcsin(z1),v1**-2) 
        return  1.0j*Gxx_1(z,v) / ( (v**2-1)*(v**2 - z**2)**0.5* (-1 + z**2)) 

    def Gyy( z,v ):
        Gyy_1 = lambda z1,v1:  -v1**2 *(-4 + v1**2) *(-1 + z1**2)*(1 - z1**2/v1**2)**0.5*complex_ellipe(arcsin(z1),v1**-2)  + (-2 + v1**2)* ( z1*(1 - z1**2)**0.5 * ( -v1**2 + z1**2) + v1**2.0* (-1 + z1**2)* (1 - z1**2/v1**2)**0.5*complex_ellipf(arcsin(z1),v1**-2))
        return  1.0j*Gyy_1(z,v) / ((v**2 - z**2)**0.5*(-1 + z**2)) 

    def Fxt( z,v ):
        return   2.0*(z/(1 - z**2)**0.5 - arcsin(z)) 

    def Fyt( z,v ):
        return  -2.0j* ((v**2 - z**2)**0.5/(1 - z**2)**0.5 - np.log( (1 - z**2)**0.5 +(v**2 - z**2)**0.5 ) ) 
    
    def Gyt( z,v ):
        return  ((-2 + v**2)*z)/(1 - z**2)**0.5 + 2.0*arcsin(z) 

    
    def Gxt( z,v ):
        Gxt_1 = lambda z1,v1:  - 1.0j*(  (-2 + v1**2) *(v1**2 - z1**2) + 2.0* (-1 + v1**2)*(v1**2 - z1**2)**0.5* (-1 + z1**2)**0.5 * np.arctan(  (-1 + z1**2)**0.5/(v1**2 - z1**2)**0.5) )  
        return  Gxt_1(z,v)/ (  (-1 + v**2) *(1 - z**2)**0.5* (v**2 - z**2)**0.5 ) 

    #calculate stress
    #here real values of v should be used. x1, y1 stand for  x/l y/l  (l=vt).

    def SxyTmp(x1,y1,v):
        #6.3.21
        return  beta*mu* (v/cs)**(-2) *np.real(2.0* Fxy(zeta(x1, y1, v/cp), v/cp) + Gyy(zeta(x1, y1, v/cs), v/cs) -Gxx(zeta(x1, y1, v/cs), v/cs)) 
    def Sxy(x1,y1,v):
        return   tau_0 +SxyTmp(x1, y1, v) 

    def Sxx(x1,y1,v):
        #(*6.3.21*)
        return   beta*mu*(v/cs)**(-2)* np.imag((cp/cs)**2.0* Fxx(zeta(x1, y1, v/cp),v/cp) + ((cp/cs)**2 - 2)* Fyy(zeta(x1, y1, v/cp),v/cp) + 2.0*Gxy(zeta(x1, y1, v/cs), v/cs)  ) 

    def Syy(x1,y1,v):
        #(*6.3.21*)
        return   beta*mu* (v/cs)**(-2) *np.imag( ((cp/cs)**2 - 2.0)* Fxx(zeta(x1, y1, v/cp), v/cp) + (cp/cs)**2 *Fyy(zeta(x1, y1, v/cp), v/cp) - 2.0* Gxy(zeta(x1, y1, v/cs), v/cs) ) 

    def vx(x1,y1,v):
        return   v*(v/cs)**(-2)*beta*np.imag( Fxt(zeta(x1, y1, v/cp), v/cp) + Gyt(zeta(x1, y1, v/cs), v/cs) ) 

    def vyTmp(x1,y1,v):
        return  v*(v/cs)**(-2)*beta*np.real(Fyt(zeta(x1, y1, v/cp), v/cp) - Gxt(zeta(x1, y1, v/cs), v/cs) ) 
  
    def vy (x1,y1,v):
        return   -vyTmp(cp/v,0.,v)+vyTmp(x1,y1,v) 


    Stress = np.zeros((3,len(x)))
    vel = np.zeros((2,len(x)))
    for n in range(len(x)):
        Stress[2,n]=Sxy(x[n]/l,y/l,cf)
        Stress[0,n]=Sxx(x[n]/l,y/l,cf)
        Stress[1,n]=Syy(x[n]/l,y/l,cf)
        vel[0,n]=vx(x[n]/l ,y/l, cf)
        vel[1,n]=vy(x[n]/l ,y/l, cf)

    return Stress, vel, x, tau_0, l, G

