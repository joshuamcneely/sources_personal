from __future__ import print_function, division, absolute_import

from enum import Enum, auto

class smd(Enum):
    # elastic material properties
    E = auto()
    nu = auto()
    rho = auto()
    llambda = auto()
    mu = auto()
    kpa = auto()
    pstress = auto()
    pstrain = auto()

    # wave speeds
    cp = auto()
    cs = auto()
    cR = auto()
    k = auto()  #cs/cp

    # cohesive law/zone
    cohesive_zone_f = auto() # function of space (function itself)
    cohesive_zone_s = auto() # function of space (string description)
    Xc = auto()
    Xc0 = auto()
    Xc0_int = auto()

    # friction
    mus = auto()
    muk = auto()
    dc = auto()
    Lc = auto()
    Gamma = auto()
    tauc = auto()
    taur = auto()
    
    #loading
    sigi = auto()
    taui = auto()
    S = auto()

