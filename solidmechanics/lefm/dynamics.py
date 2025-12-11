# dynamics.py
#
# LEFM dynamics, e.g., equation of motion
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
import math
import os
import collections
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from ..definitions import smd

from .functions import compute_gII


def subRayleigh_equation_of_motion(crack_lengths,matprop,**kwargs):
    """
    general theory for subRayleigh equation of motion

    Args:
      - crack_lengths (float or list): crack lengths for which Cf is computed
      - matprop : material properties

    **kwargs:
      - crack_type (str): bilateral or semi-infinite
        
      - x (array): coordinates along interface [0,Xmax] - only needed for non-uniform problems
      - tau0 (array): non-uniform prestress - need x to be defined
      - Gamma (array): non-uniform fracture energy - need x to be defined

      - uni_tau0 (float): uniform prestress
      - uni_Gamma (float): uniform fracture energy

      - nb_root_sec (int): number of section C_s < C_f < C_p is cut into to search for roots (default = 5)
        
    Returns:
      - eom (array): eom[:,0] = crack length
                     eom[:,1] = rupture speed
    """

    # --------------
    # prepare input data
    # --------------

    # get coordinates and check if non-uniform problem
    if 'x' in kwargs:
        coordinate = kwargs.get('x')
        uniform_problem = False
    else:
        uniform_problem = True

    # crack type: bilateral etc.
    crack_type = kwargs.get('crack_type')

    # get properties
    if uniform_problem:
        tau0 = float(kwargs.get('uni_tau0'))
        Gamma = float(kwargs.get('uni_Gamma'))
    else:
        # get prestress
        if 'uni_tau0' in kwargs:
            utau0 = kwargs.get('uni_tau0')
            itau0 = lambda t: utau0 # tau0 interpolation
        elif 'tau0' in kwargs:
            vtau0 = kwargs.get('tau0')
            if not vtau0.shape == coordinate.shape:
                print('tau0 does not have same shape as x: ',
                      vtau0.shape,coordinate.shape)
                raise RuntimeError
            itau0 = interp1d(coordinate,vtau0)
        else:
            print('you need to define a prestress: uni_tau0 or tau0')
            raise RuntimeError
            
        # get fracture energy
        if 'uni_Gamma' in kwargs:
            uGamma = kwargs.get('uni_Gamma')
            iGamma = lambda t: uGamma # Gamma interpolation
        elif 'Gamma' in kwargs:
            vGamma = kwargs.get('Gamma')
            if not vGamma.shape == coordinate.shape: #Gamma interpolation
                print('Gamma does not have same shape as x: ',
                      vGamma.shape,coordinate.shape)
                raise RuntimeError
            iGamma = interp1d(coordinate,vGamma)
        else:
            print('you need to define a fracture energy: uni_Gamma or Gamma')
            raise RuntimeError
            
    # check that x goes from zero to something
    if not uniform_problem and min(coordinate) > 0:
        print('Your x has to be [0,Xmax] - including 0! '
              +'Your x starts with {}'.format(min(coordinate)))
        raise RuntimeError

    # in case crack_lengths is not a list
    if not isinstance(crack_lengths, collections.Iterable):
        crack_lengths = [crack_lengths]
    
    # check crack length does not exceed x
    if not uniform_problem and max(crack_lengths) > max(coordinate):
        import warnings
        warning='max crack length exceeded x coord: modified crack length [{} -> {}]'.format(max(crack_lengths),
                                                                                             max(coordinate))
        warnings.warn(warning)
        crack_lengths = np.array(crack_lengths)
        crack_lengths = crack_lengths[np.where(crack_lengths < max(coordinate))[0]]
    
    # number of sections the supershear domain is cut into to check for roots
    nb_sec = kwargs.get('nb_root_sec', 5)
                    
    # k = c_s/c_p <- this contains already the plane stress/strain information
    # k = float(matprop[smd.cs]) / float(matprop[smd.cp])

    E_eff = matprop[smd.E]
    if mat.is_plane_strain(matprop):
        E_eff /= (1-matprop[smd.nu]**2)

    # --------------
    # equation of motion
    # --------------

    # UNIFORM PROBLEM ---------------------------

    if uniform_problem:

        if crack_type == 'bilateral':

            cfs = np.linspace(0,matprop[smd.cR],1001)[:-1]
            gIIs =compute_gII(cfs,matprop)
            # do I need np.real() on gIIs ?
        
            a = Gamma * E_eff / math.pi / gIIs / tau0**2

            # reduce to area of interest
            fltr_max = np.where(a < max(crack_lengths))[0]
            fltr_min = np.where(a > min(crack_lengths))[0]
            fltr = np.intersect1d(fltr_min, fltr_max)

            # create return structure
            eom = np.zeros((len(fltr),2))
            eom[:,0] = a[fltr]
            eom[:,1] = cfs[fltr]
            
            return eom

        else:
            raise RuntimeError("subRayleigh equation of motion "\
                               +"for uniform set-up "\
                               +"is only coded for bilateral crack. "\
                               +"set crack-type='bilateral' or code a new type")

    # NON-UNIFORM PROBLEM ---------------------------
    if crack_type == 'bilateral':
        # a will be defined later: crack length
        K_II_integrand = lambda t: itau0(t) * np.sqrt( a / (a**2 - t**2))
    else:
        raise RuntimeError("You need to define the crack type: "
                           +"crack-type='bilateral'")

    eom = list()
    for a in crack_lengths:

        Gamma = iGamma(a)
        
        KII = 2/np.sqrt(math.pi) * quad(K_II_integrand,0,a)[0]
        dG = lambda v: KII**2 / E_eff * compute_gII(v,matprop) / Gamma - 1 
    
        # check if there is a solution
        cR_sec = np.linspace(1e-6*matprop[smd.cR],
                             (1-1e-6)*matprop[smd.cR],
                             nb_sec+1)
        dG_sec = [dG(i) for i in cR_sec] 

        # check if there is a sign change within one section
        sign_change = [i*j < 0 for (i,j) in zip(dG_sec[:-1],dG_sec[1:])] 
        roots = [i for i, x in enumerate(sign_change) if x]
        
        for i in roots:
            cf = brentq(dG,cR_sec[i],cR_sec[i+1])
            eom.append([a, cf])# * float(matprop[smd.cR])])
       
    return np.array(eom)

def supershear_g_G(k,**kwargs):
    """
    get factors needed for supershear equation of motion

    This returns the g and the G = B*Gamma term for a series of cf/cL values 
    according to the definition as provided in Kammer et al. 2018.

    Args:
      - k (float): Cs/Cl ratio of shear and longitudinal wave speeds 
                   (material property)

    **kwargs:
      - cohesive_zone_type (string): e.g., 'linear'

    Returns:
      cf/cL,g,G : all numpy arrays
    """

    # to be replaced with a function that computes these mastercurves
    if kwargs.get('cohesive_zone_type') == 'linear':
        # find path to master curve files
        dirname, filename = os.path.split(os.path.abspath(__file__))
        mastercurve_folder = 'supershear-mastercurves'

        # 3 decimals precision for k in filename 
        # file: 0=Cf/CL 1=g(Cf/CL) 2=G_{my}
        fname = 'k{:.3f}'.format(k).replace('.','')+'.csv'
        fpath = os.path.join(dirname,mastercurve_folder,fname)
        try:
            master_curve = np.loadtxt(fpath,delimiter=',',dtype=float)
        except IOError:
            print("!!! It seems you do not have the master curve file "
                  +"for this k={}".format(k))
            raise RuntimeError
    else:
        print('You need to specify a cohesive zone shape: '
              +'cohesive_zone_type=linear')
        raise RuntimeError
        
    # return cfcls, gs, Gmys
    return master_curve[:,0], master_curve[:,1], master_curve[:,2]


def supershear_equation_of_motion(crack_lengths,matprop,**kwargs):
    """
    general theory for supershear equation of motion

    This function computes the theoretical prediction for 
    supershear equation of motion. It can handle various configurations 
    including varying interface properties and pre-stress. The underlying 
    theory is based on Ilya Svetlizky's notes (2017/03/28). It may not 
    be entirely rigorous.
    
    Args:
      - crack_lengths (float or list): crack lengths for which Cf is computed
      - matprop : material properties
      
    **kwargs:
        - crack_type (str): bilateral or semi-infinite
        - cohesive_zone_type (str): linear
        
        - x (array): coordinates along interface [0,Xmax] - only needed for non-uniform problems
        - tau0 (array): non-uniform prestress - need x to be defined
        - taup (array): non-uniform peak strength - need x to be defined
        - Gamma (array): non-uniform fracture energy - need x to be defined

        - uni_tau0 (float): uniform prestress
        - uni_taup (float): uniform peak strength
        - uni_Gamma (float): uniform fracture energy

        - nb_root_sec (int): number of section C_s < C_f < C_p is cut into to search for roots (default = 5)
        
    Returns:
      - eom (array): eom[:,0] = crack length
                     eom[:,1] = rupture speed
    
    """

    # --------------
    # prepare input data
    # --------------

    # crack type: bilateral etc.
    crack_type = kwargs.get('crack_type')

    # get coordinates and check if non-uniform problem
    if 'x' in kwargs:
        coordinate = kwargs.get('x')
        uniform_problem = False
    else:
        uniform_problem = True

    # get properties
    if uniform_problem:
        tau0 = float(kwargs.get('uni_tau0'))
        taup = float(kwargs.get('uni_taup'))
        Gamma = float(kwargs.get('uni_Gamma'))
    else:
        # get prestress
        if 'uni_tau0' in kwargs:
            utau0 = kwargs.get('uni_tau0')
            itau0 = lambda t: utau0 # tau0 interpolation
        elif 'tau0' in kwargs:
            vtau0 = kwargs.get('tau0')
            if not vtau0.shape == coordinate.shape:
                print('tau0 does not have same shape as x: ',
                      vtau0.shape,coordinate.shape)
                raise RuntimeError
            itau0 = interp1d(coordinate,vtau0)
        else:
            print('you need to define a prestress: uni_tau0 or tau0')
            raise RuntimeError
            
        # get peak strength
        if 'uni_taup' in kwargs:
            utaup = kwargs.get('uni_taup')
            itaup = lambda t: utaup # taup interpolation
        elif 'taup' in kwargs:
            vtaup = kwargs.get('taup')
            if not vtaup.shape == coordinate.shape:
                print('taup does not have same shape as x: ',
                      vtaup.shape,coordinate.shape)
                raise RuntimeError
            itaup = interp1d(coordinate,vtaup) # taup interpolation
        else:
            print('you need to define a peak strength: uni_taup or taup')
            raise RuntimeError

        # get fracture energy
        if 'uni_Gamma' in kwargs:
            uGamma = kwargs.get('uni_Gamma')
            iGamma = lambda t: uGamma # Gamma interpolation
        elif 'Gamma' in kwargs:
            vGamma = kwargs.get('Gamma')
            if not vGamma.shape == coordinate.shape: #Gamma interpolation
                print('Gamma does not have same shape as x: ',
                      vGamma.shape,coordinate.shape)
                raise RuntimeError
            iGamma = interp1d(coordinate,vGamma)
        else:
            print('you need to define a fracture energy: uni_Gamma or Gamma')
            raise RuntimeError
            
    # check that x goes from zero to something
    if not uniform_problem and min(coordinate) > 0:
        print('Your x has to be [0,Xmax] - including 0! '
              +'Your x starts with {}'.format(min(coordinate)))
        raise RuntimeError

    # in case crack_lengths is not a list
    if not isinstance(crack_lengths, collections.Iterable):
        crack_lengths = [crack_lengths]
    
    # check crack length does not exceed x
    if not uniform_problem and max(crack_lengths) > max(coordinate):
        import warnings
        warning='max crack length exceeded x coord: modified crack length [{} -> {}]'.format(max(crack_lengths),
                                                                                             max(coordinate))
        warnings.warn(warning)
        crack_lengths = np.array(crack_lengths)
        crack_lengths = crack_lengths[np.where(crack_lengths < max(coordinate))[0]]
    
    # number of sections the supershear domain is cut into to check for roots
    nb_sec = kwargs.get('nb_root_sec', 5)
                    
    # k = c_s/c_p <- this contains already the plane stress/strain information
    k = float(matprop[smd.cs]) / float(matprop[smd.cp])

    # --------------
    # get master-curve
    # --------------

    cfcls, gs, Gmys = supershear_g_G(k,**kwargs)

    # interpolation of g and Gmy
    ig   = interp1d(cfcls, gs)
    iGmy = interp1d(cfcls,Gmys)
    
    # --------------
    # equation of motion
    # --------------

    # UNIFORM PROBLEM ---------------------------

    # the uniform solution is based on Broberg (1994) book p.419 
    # on a self-similar expanding singular crack propagating symmetrically 
    # in both direction with a constant inter-sonic velocity according to 
    # the developed general theory (see below and Ilya's document), 
    # the semi-infinite solution would result in the same uniform solution.
    if uniform_problem:
        if crack_type == 'bilateral':
            # characteristic length
            a0 = Gamma * float(matprop[smd.mu]) / taup**2
            
            # compute crack length corresponding to rupture speed
            a = a0 / Gmys / (tau0 / taup)**(1/gs)
            
            # reduce to area of interest
            fltr = np.where(a < max(crack_lengths))[0]
            
            # create return structure
            eom = np.zeros((len(fltr),2))
            eom[:,0] = a[fltr]
            eom[:,1] = cfcls[fltr] * float(matprop[smd.cp])
            
            return eom
        else:
            raise RuntimeError('Uniform problem works for bilateral crack only.')

    # NON-UNIFORM PROBLEM ---------------------------
    
    # compute K_homo (g) [Ilya's document just before equation (e6)]
    # semi-infinite crack
    if crack_type == 'semi-infinite':
        Khomos = 1 / gs
        # a will be defined later: crack length
        Ks_integrand = lambda t,v: itau0(t) / ((a-t)**(1-ig(v)))
    # bi-lateral crack
    elif crack_type == 'bilateral':
        Khomos = np.array([quad(lambda t: 1 / (1-t**2)**(1-tt),0,1)[0] for tt in gs])
        # a will be defined later: crack length
        Ks_integrand = lambda t,v: itau0(t) * ( a / (a**2-t**2) )**(1-ig(v))
    else:
        print("You need to define the crack type: "
              +"crack-type='bilateral' or 'semi-infinite'")
        raise RuntimeError
    # interpolation of K_homo
    iKhomo = interp1d(cfcls, Khomos)

    eom = list()
    for a in crack_lengths:

        taup  = itaup(a)
        Gamma = iGamma(a)

        # if ever this integral causes problem: checkout "improper integrals"
        # there is a useful transformation in: Ch.4.4 of Numerical Recipes in C
        Ks = lambda v: quad(Ks_integrand,0,a,args=(v,))[0]
        dG = lambda v: taup**2 / matprop[smd.mu] * (Ks(v) / iKhomo(v) / taup)**(1/ig(v)) * iGmy(v) / Gamma - 1

        # check if there is a solution
        cfcl_sec = np.linspace(cfcls[0],cfcls[-1],nb_sec+1)
        dG_sec = [dG(i) for i in cfcl_sec] 
        # check if there is a sign change within one section
        sign_change = [i*j < 0 for (i,j) in zip(dG_sec[:-1],dG_sec[1:])] 
        roots = [i for i, x in enumerate(sign_change) if x]
        
        for i in roots:
            cfcl = brentq(dG,cfcl_sec[i],cfcl_sec[i+1])
            eom.append([a, cfcl * float(matprop[smd.cp])])
        
    return np.array(eom)


def integrate_equation_of_motion(eom,**kwargs):
    """
    computes the space-time data for an equation of motion.

    This integrates an equation of motion computed by either 
    subRayleigh_equation_of_motion or supershear_equation_of_motion. 

    Args:
      - eom (np.array): equation of motion where eom[:,0] are crack lengths and eom[:,1] crack speeds
    
    **kwargs:
      - none so far: one could add integration type (forward, backward etc)
    
    Returns:
      - integ_eom: integrated equation of motion = space-time data: integ_eom[:,0]=space integ_eom[:,1]=time
    """

    # test if there is several speeds for a given position, if so -> fail
    if not len(eom[:,0]) == len(set(eom[:,0])):
        raise RuntimeError('The provided equation of motion '
                           +'has multiple speeds for a single position')

    # sort the equation of motion w.r.t. space
    ind = np.lexsort((eom[:,0],eom[:,1]))
    eom = np.array([eom[i] for i in ind])
    
    # forward integrate to get time: t_i+1 = t_i + dx_i/v_i
    integ_eom = []
    integ_eom.append([eom[0,0],0])
    for i in range(len(eom)-1):
        if eom[i,1] == 0:
            break
        dx = eom[i+1,0] - eom[i,0]
        integ_eom.append([eom[i+1,0], integ_eom[-1][1] + dx/eom[i,1]])
    return np.array(integ_eom)



def test_supershear_eom():
    from ..linearelasticity import LinearElasticMaterial as lem
    mat = lem({smd.E : 5.65e9,
               smd.nu : 0.35,
               smd.rho : 1170,
               smd.pstrain : True})

    # coordinate
    dx=0.0001
    X=0.8
    x = np.arange(0,X,dx)

    # peak strength
    tp = np.ones_like(x)*1e6
    # fracture energy
    Gamma = np.ones_like(x)

    l0 = Gamma[0] * mat[smd.mu] / tp[0]**2

    # pre-stress uniform (to compare with uniform solution)
    t0 = np.ones_like(x) * 0.3e6

    # compute uniform solution but only up to to maximal crack length provided (here 0.1254)
    eom = supershear_equation_of_motion(np.array([0.0002496, 0.03, 0.1254]),
                                        mat,
                                        uni_Gamma=Gamma[0],
                                        uni_tau0=t0[0],
                                        uni_taup=tp[0],
                                        crack_type='bilateral',
                                        cohesive_zone_type='linear',
                                        nb_root_sec=20)

    x_l0_end = eom[-1,0]/l0
    v_cp_end = eom[-1,1]/mat[smd.cp]
    v_cp_sol = 0.734
    if abs((v_cp_end - v_cp_sol)/v_cp_sol) > 1e-8:
        print(v_cp_end)
        raise RuntimeError('supershear equation of motion for uniform set-up is wrong')
    
    
    # check code for non-uniform set-up, with uniform values
    # -> should give same result
    eom = supershear_equation_of_motion(np.array([0.1254]),
                                        mat,
                                        x=x,
                                        Gamma=Gamma,
                                        tau0=t0,
                                        taup=tp,
                                        crack_type='bilateral',
                                        cohesive_zone_type='linear',
                                        nb_root_sec=20)
    v_cp_end = max(eom[:,1])/mat[smd.cp]
    if abs((v_cp_end - v_cp_sol)/v_cp_sol) > 1e-3:
        print(v_cp_sol, v_cp_end)
        raise RuntimeError('supershear equation of motion for non-uniform set-up is wrong')



if (__name__ == '__main__'):
    print('unit tests: lefm dynamics')
    test_supershear_eom()
    print('success!')
