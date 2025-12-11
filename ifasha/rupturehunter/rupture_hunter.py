#!/usr/bin/env python
#
# rupture_hunter.py
#
# Code to track ruptures
# There is no warranty for this code
#
# @version 2.0
# @author David Kammer <dkammer@ethz.ch>
# @date     2013/05/21
# @modified 2020/12/22

from __future__ import print_function, division, absolute_import

import sys
import copy
import numpy as np
from itertools import count
from ast import literal_eval
from scipy import signal, interpolate


class SlipPeriod(object):

    """Class used to identify the period of slips

    """
    sep = ':'
    
    def __init__(self,start,end,rptindex=-1):
        """Initiates an object

        Args:
            start (float): period starting time
            end (float): period ending time
            rptindex (int): rupture index

        """
        self.start         = start
        self.end           = end
        self.rupture_index = rptindex

    def __eq__(self,other):
        if not self.start == other.start:
            return False
        if not self.end == other.end:
            return False
        if not self.rupture_index == other.rupture_index:
            return False
        return True

    @classmethod
    def overlap(cls,sp1,sp2):
        """Function to judge whether two periods overlap or not

        Args:
            sp1 (SlipPeriod): slip period 1
            sp2 (SlipPeriod): slip period 2

        Returns:
            bool: True if overlapped; False if not

        """
        return not (sp1.start > sp2.end or sp2.start > sp1.end) 

    def get_period(self):
        """Function used to compute the length of a given period

        Returns:
            float: the period length

        """
        return self.end - self.start

    def write_string(self):
        return str(self.start)+SlipPeriod.sep+str(self.end)+SlipPeriod.sep+str(self.rupture_index)

    def read_string(self,string):
        data=string.split(SlipPeriod.sep)
        self.start = float(data[0])
        self.end = float(data[1])
        self.rupture_index = int(data[2])

    def __repr__(self):
        """Return a string containing a printable representation of an object.

        Returns:
            string: containing period start/end time and rupture index

        """
        out_string = "SP:"
        out_string += " s={}".format(self.start)
        out_string += " e={}".format(self.end)
        out_string += " r={}".format(self.rupture_index)
        return out_string


class SlipPeriodsAtX(list):

    """
    Class used to identify the Slip Periods at a given grid point(gp). 
    Inherited from list.

    """
    sep = '::'
    
    def __init__(self,gp):
        """Initiates an object

        Args:
            gp (tuple): grid point coordinates, (x, y)

        """
        list.__init__(self)
        self.gp = gp

    def set_rupture_index(self,idx):
        """Function used to set rupture index

        Args:
            idx (int): index

        """
        for sp in self:
            sp.rupture_index = idx

    def get_overlaps(self,slip_period):
        """Function used to list all the slips whose period overlaps a given slip period

        Args:
            slip_period (SlipPeriod): the given slip

        Returns:
            list: all the slips whose period overlaps slip_period

        """
        osp = list()
        for sp in self:
            if SlipPeriod.overlap(slip_period,sp):
                osp.append(sp)
        return osp

    def first(self):
        """
        return copy of this spax with only first slip period of this spax
        
        """
        spaxcpy = SlipPeriodsAtX(self.gp)
        
        starts = [sp.start for sp in self]
        idx = np.argmin(starts)
        spaxcpy.append(copy.deepcopy(self[idx]))

        return spaxcpy

    def write_string(self):
        string = str(self.gp)
        for sp in self:
            string+=SlipPeriodsAtX.sep+sp.write_string()
        return string

    def read_string(self,string):
        data=string.split(SlipPeriodsAtX.sep)
        gpdata = data[0].strip('(').strip(')').strip()
        gpdata = gpdata.split(',')
        gpdata = [float(g.strip()) for g in gpdata ]
        self.gp = tuple(gpdata)

        for dd in data[1:]:
            sp = SlipPeriod(None,None)
            sp.read_string(dd)
            self.append(sp)

    def __repr__(self,spnb=2):
        """Return a string containing a printable representation of an object.

        Args:
            spnb (int): parameter to control the output

        Returns:
            string:

        """
        out_string = list()
        out_string.append("SPAX at gp={}:".format(self.gp))
        [out_string.append(sp.__repr__()) for sp in self]
        space = "\n" + " ".join(["" for i in range(spnb)])
        return space.join(out_string)

        
class SPAXGrid(object):

    """Class used to identify the grid

    """
    sep = ':'
    
    def __init__(self):
        """Initiates an object

        """
        self.xgrid = set()
        self.ygrid = set()
        self.sorted_xgrid = None
        self.sorted_ygrid = None
        self.modified = False
    
    def __eq__(self,other):
        if not self.xgrid == other.xgrid:
            return False
        if not self.ygrid == other.ygrid:
            return False
        if not self.sorted_xgrid == other.sorted_xgrid:
            return False
        if not self.sorted_ygrid == other.sorted_ygrid:
            return False
        if not self.modified == other.modified:
            return False
        return True

    def add(self,gp):
        """Function used to add the x coordinate and y coordinate

        Args:
            gp (tuple): x and y coordinate, (x, y)

        """
        self.xgrid.add(gp[0])
        self.ygrid.add(gp[1])
        self.modified = True

    def write_string(self):
        # save unsorted grid
        if self.modified or self.sorted_xgrid is None:
            return 'unsorted'+SPAXGrid.sep+str(self.modified)\
                +SPAXGrid.sep+str(list(self.xgrid))+SPAXGrid.sep+str(list(self.ygrid))
        else:
            return 'sorted'+SPAXGrid.sep+str(self.modified)\
                +SPAXGrid.sep+str(self.sorted_xgrid)+SPAXGrid.sep+str(self.sorted_ygrid)

    def read_string(self,string):
        data=string.split(SPAXGrid.sep)
        if data[0] == 'unsorted':
            self.xgrid = set(literal_eval(data[2]))
            self.ygrid = set(literal_eval(data[3]))
            self.sorted_xgrid = None
            self.sorted_ygrid = None
            self.modified = data[1]=='True'
        elif data[0] == 'sorted':
            self.sorted_xgrid = literal_eval(data[2])
            self.sorted_ygrid = literal_eval(data[3])
            self.xgrid = set(self.sorted_xgrid)
            self.ygrid = set(self.sorted_ygrid)
            self.modified = data[1]=='True'

    def get_grid_point_list(self):
        """Function to get the list of grid points

        Returns:
            list: the all pairs of coordinates, (x, y)

        """
        gpl = list()
        for x in self.xgrid:
            for y in self.ygrid:
                gpl.append((x,y))
        return gpl

    def sort_grid(self):
        """Function used to get sorted grid

        """
        if self.modified:
            self.sorted_xgrid = sorted(self.xgrid)
            self.sorted_ygrid = sorted(self.ygrid)
            self.modified = False
            
    def get_neighbor_grid_points(self,gp):
        """Function used to obtain the neighbor of a given grid point

        Args:
            gp (tuple): x and y coordinate, (x, y)

        """
        self.sort_grid()
        ix = self.sorted_xgrid.index(gp[0])
        iy = self.sorted_ygrid.index(gp[1])
        gpl = list()
        for x in range(max(0,ix-1),min(ix+2,len(self.sorted_xgrid))):
            for y in range(max(0,iy-1),min(iy+2,len(self.sorted_ygrid))):
                gpl.append((self.sorted_xgrid[x],self.sorted_ygrid[y]))
        return gpl

    def get_x_min_max(self):
        """Function used to get the max and min of x coordinates

        Returns:
            tuple: min and max of x coordinates

        """
        self.sort_grid()
        return (self.sorted_xgrid[0],self.sorted_xgrid[-1])

    def get_y_min_max(self):
        """Function used to get the max and min of y coordinates

        Returns:
            tuple: min and max of y coordinates

        """
        self.sort_grid()
        return (self.sorted_ygrid[0],self.sorted_ygrid[-1])

    def get_sorted_x_grid(self):
        """Function used to get sorted grid in x direction

        """
        self.sort_grid()
        return copy.deepcopy(self.sorted_xgrid)

    def get_sorted_y_grid(self):
        """Function used to get sorted grid in y direction

        """
        self.sort_grid()
        return copy.deepcopy(self.sorted_ygrid)
        
        
class SPAXCollection(dict):

    """Class used to identify the Slip Periods over grids

    """
    keychar = '!'
    
    def __init__(self):
        """Initiates an object

        """
        dict.__init__(self)
        self.grid = SPAXGrid()
        
    def insert_spax(self,spax):
        """Function used to insert Slip Periods at a grid point into current collection of Slip Periods

        Args:
            spax (SlipPeriodsAtX): Slip Periods at a grid point

        """
        if spax.gp in self:
            self[spax.gp].extend(spax)
        else:
            self[spax.gp] = spax
            self.grid.add(spax.gp)

    def insert_sp(self,sp,gp):
        """Function used to insert Slip Periods into current collection of Slip Periods for a certain grid point

        Args:
            sp (SlipPeriod): slip period
            gp (tuple): x and y coordinate, (x, y)

        """
        if gp in self:
            self[gp].append(sp)
        else:
            spax = SlipPeriodsAtX(gp)
            spax.append(sp)
            self.insert_spax(spax)
            #self[(x,y)] = spax

    def getSPAXAtX(self,gp):
        """Function used to get Slip Periods at a grid point

        Args:
            gp (tuple): x and y coordinate, (x, y)

        Returns:
            SlipPeriodsAtX: Slip Periods at a grid point

        """
        if gp not in self:
            spax = SlipPeriodsAtX(gp)
            self.insert_spax(spax)
        return self[gp]

    def write_strings(self):
        output = list()
        output.append(SPAXCollection.keychar+'spaxgrid')
        output.append(self.grid.write_string())
        output.append(SPAXCollection.keychar+'spax')
        for value in self.values():
            output.append(value.write_string())
        return output

    def read_strings(self,string_list):
        for string in string_list:
            if string.startswith(SPAXCollection.keychar):
                key=string.strip(SPAXCollection.keychar)
                continue
            
            if key == 'spaxgrid':
                self.grid.read_string(string)

            elif key == 'spax':
                spax = SlipPeriodsAtX(None)
                spax.read_string(string)
                self.insert_spax(spax)

            
    def __repr__(self,spnb=2):
        """Return a string containing a printable representation of an object.

        Args:
            spnb (int): parameter to control the output

        Returns:
            string:

        """
        out_string = list()
        out_string.append("SPAXC:")
        [out_string.append(spax.__repr__(spnb+1)) for spax in self.values()]
        space = "\n" + " ".join(["" for i in range(spnb)])
        return space.join(out_string)


class Rupture(SPAXCollection):
    """Class used to identify ruptures

    """
    _rid = count(0)
    
    def __init__(self):
        """Initiates an object

        """
        SPAXCollection.__init__(self)
        self.index = next(self._rid)

    def set_rupture_index(self,idx):
        """Function used to set rupture index

        Args:
            idx (int): index

        """
        self.index = idx
        for spax in self.values():
            spax.set_rupture_index(idx)

    def insert_spax(self,spax):
        """Function used to insert Slip Periods at a grid point into current collection of Slip Periods

        Args:
            spax (SlipPeriodsAtX): Slip Periods at a grid point

        """
        spax.set_rupture_index(self.index)
        super(Rupture, self).insert_spax(spax)

    def insert_sp(self,sp,gp):
        """Function used to insert Slip Periods into current collection of Slip Periods for a certain grid point

        Args:
            sp (SlipPeriod): slip period
            gp (tuple): x and y coordinate, (x, y)

        """
        sp.rupture_index = self.index
        super(Rupture, self).insert_sp(sp,gp)

    def add_rupture(self,rpt):
        """Function used to add a rupture into current collection of Slip Periods

        Args:
            rpt (SPAXCollection): rupture
        """
        for spax in rpt.values():
            self.insert_spax(spax)

    def get_start(self):
        """Funtion used to get the rupture start time

        Returns:
            float: rupture start time
        """
        start = float("inf")
        for spax in self.values():
            for sp in spax:
                if sp.start < start:
                    start = sp.start
        return start

    def get_nucleation(self):
        """
        get nucleation point and tim

        Returns:
          coordinates (tuple) of point of nucleation, and
          time (float) of nucleation
        """
        start = self.get_start()

        gp = None
        for spax in self.values():
            for sp in spax:
                if sp.start == start:
                    gp = spax.gp

        if gp is None:
            print('could not find nucleation')
            raise RuntimeError

        return gp, start


    def get_front(self):
        """
        Function used to get a list of arrays for fronts of all ruptures
        front = f[x0,x1,t], 
        where x0, x1 are positions in cartesian coordinate system and t time

        Returns:
            list: arrays for fronts of all ruptures fronts
        """
        front = list()
        for spax in self.values():
            gp = spax.gp
            for sp in spax:
                front.append([gp[0],gp[1],sp.start])
        return np.array(front)

    def get_sorted_front(self):
        """ get array for sorted fronts of this rupture

        Returns:
            arrays for sorted fronts of this ruptures
            front = f[x0,x1,t]
        """
        front = self.get_front()

        ind = np.lexsort((front[:,1],front[:,0]))
        sorted_front = np.array([front[i,:] for i in ind])
        
        return sorted_front

    def get_sorted_front_XYT(self):
        """ get sorted 2Darrays for sorted fronts of this rupture
            ideal for colorplot or countour line

        Returns:
            arrays for sorted fronts of this ruptures
            front = X,Y,T
        """
        front = self.get_sorted_front()
        x=front[:,0]  
        y=front[:,1]  
        t=front[:,2]  
        nx=len(np.unique(x))
        ny=len(np.unique(y))
        
        try:
            X=x.reshape(nx,ny)
            Y=y.reshape(nx,ny)
            T=t.reshape(nx,ny)
        except:
            yu=np.unique(y)
            xu=np.unique(x)
            Y,X=np.meshgrid(yu,xu)
            T=np.zeros(X.shape)*np.nan
            for xi,yi,ti in zip(x,y,t):
                ix = np.where(xi==xu)
                iy = np.where(yi==yu)
                T[ix,iy]=ti

        return X,Y,T

    def get_back(self):
        """Function used to get a list of arrays for backs of all ruptures
            back = b[x0,x1,t]

        Returns:
            list: arrays for backs of all ruptures
        """
        back = list()
        for spax in self.values():
            gp = spax.gp
            for sp in spax:
                back.append([gp[0],gp[1],sp.end])
        return np.array(back)

    def get_sorted_back(self):
        """Function used to get a list of arrays for sorted backs of all ruptures
            back = b[x0,x1,t]

        Returns:
            list: arrays for sorted backs of all ruptures

        """
        back = self.get_back()

        ind = np.lexsort((back[:,1],back[:,0]))
        sorted_back = np.array([back[i,:] for i in ind])

        return sorted_back

    def first(self): 
        """
        returns copy of this with spax that have only first slip periods
        """
        rptc = Rupture()
        
        for spax in self.values():
            spaxf = spax.first()
            rptc.insert_spax(spaxf)

        return rptc

    def get_expansion(self):
        """Function used to get a list of expansion

        Returns:
            list: expansion data
        """
        xmm = self.grid.get_x_min_max()
        ymm = self.grid.get_y_min_max()
        return [[xmm[0],ymm[0]], [xmm[1],ymm[1]]]

    def get_tip_coord_at_time(self,time,**kwargs):

        rpt = self.first()
        front = rpt.get_sorted_front()
        nuc_p, nuc_t = rpt.get_nucleation()
        nuc_i = np.argmin(abs(front[:,0]-nuc_p[0]))

        # find crack tip position
        prop_dir = kwargs.get('prop_dir','to_right')
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

        return xtip

    # needed change: get only first front for a given position
    def get_propagation_speed(self,
                              direction=0,
                              avg_dist=0.,
                              mode='central_diff',
                              layer='all',
                              **kwargs):
        """
        compute rupture speed of first front of this rupture

        Args:
            direction (int): 0 for x direction, 1 for y direction
            avg_dist (float): average position for the propagation
            layer='all' (str/float): coordinate perpendicular to crack prop

        Kwargs:
            prop_dir(string): 'both' (default) 'to_right', 'to_left' 
                              only propagation in one direction from nucleation

        Returns:
            np.array: propagation speed [p, t, speed], where
                        p is the position,
                        t is the time,
                        speed is the propagation speed

        """
        propagation_speed_map = dict()
        # eliminate for a given position all later fronts
        sorted_front = self.first().get_sorted_front()
        
        if direction == 0:
            ldir = 1
            layers = self.grid.get_sorted_y_grid()
        elif direction == 1:
            ldir = 0
            layers = self.grid.get_sorted_x_grid()
        if layer != 'all':
            true_layer = layers[np.argmin(np.abs(np.array(layers) - layer))]
            layers=[true_layer]
        
        for l in layers:
            propagation_speed = list()

            # take only the sorted front of this layer
            rel = np.where(sorted_front[:,ldir] == l)[0]
            current_sorted_front = sorted_front[rel]
            
            # for a given time take only the most advanced front position
            # only if they follow each other directly
            # (not if there are two tips at the same time)
            # this is to eliminate same timing because of low dumping frequency
            nb_sort_front = len(current_sorted_front[:,2])
            ind = [nb_sort_front-1]
            for i in range(nb_sort_front-2, -1, -1):
                if current_sorted_front[i,2] != current_sorted_front[i+1,2]:
                    ind.append(i)
            ind = list(reversed(ind))
            current_sorted_front = current_sorted_front[ind]

            position = np.array([csf[direction] for csf in current_sorted_front])
            time = np.array([csf[2] for csf in current_sorted_front])

            # if only for one direction from nucleation
            prop_dirs = ['both','to_right','to_left']
            prop_dir = kwargs.get('prop_dir',prop_dirs[0])
            if not prop_dir in prop_dirs:
                print('Not an appropriate propagation direction "prop_dir"')
                print('choose among: {}'.format(prop_dirs))
                raise RuntimeError
            
            if prop_dir != 'both':
                nuc_pos, nuc_time = self.get_nucleation()
                if prop_dir == 'to_right':
                    fltr = position >= nuc_pos[direction]
                elif prop_dir == 'to_left':
                    fltr = position <= nuc_pos[direction]
                position = position[fltr]
                time = time[fltr]

            if mode=='central_diff':
                for i in range(len(position)-2):
                    # left position
                    lpos = position[i]
                    il = i

                    # right position
                    rpos = lpos + avg_dist
                    dpos = abs(position - rpos)
                    ir = np.argmin(dpos)
                                        
                    ir = max(ir, il+2) # ensure that avg_dist < discretization
                    rpos = position[ir]

                    if ir == len(position)-1:
                        break
                        
                    # center position
                    cpos = lpos + 0.5*avg_dist
                    dpos = abs(position - cpos)
                    ic = np.argmin(dpos)
                    ic = max(ic, il+1) # ensure that avg_dist < discretization
                    cpos = position[ic]
                    
                    # compute speed
                    dx = abs(rpos - lpos)
                    dt = abs(time[ir] - time[il])

                    # [0]-position, [1]-time, [2]-speed
                    if dt > 0:
                        propagation_speed.append([cpos, time[ic], dx/float(dt)])
                   
            elif mode=='forward_diff':
                print('WARNING: only accurate for regular mesh')
                
                step=1
                dx_0 = avg_dist
                
                for il in range(len(position)-1):
                    # left position
                    lpos = position[il]

                    # right position
                    ir=il+step
                    rpos = position[ir]
                    if ir == len(position)-1:
                        break

                    # compute speed
                    dx = rpos - lpos
                    dt = abs(time[ir] - time[il])# for Burridge Andrews transition

                    # [0]-position, [1]-time, [2]-speed
                                        
                    step_i = int(round(dx/dx_0))
                    propagation_speed.append([lpos, time[il], dx/float(dt)])
                    for j in range(1,step_i): # when velocity would be infinite is computed over multiple dx_o
                        propagation_speed.append([lpos+dx/step_i*j, time[il]+dt/step_i*j, dx/float(dt)])
            elif mode=='spline_intp':
                """ use cubic spline to estime derivative
                s : float
                    A smoothing condition. The amount of smoothness is determined by satisfying the conditions: sum((w * (y - g))**2,axis=0) <= s where g(x) is the smoothed interpolation of (x,y). The user can use s to control the tradeoff between closeness and smoothness of fit. Larger s means more smoothing while smaller values of s indicate less smoothing. Recommended values of s depend on the weights, w. If the weights represent the inverse of the standard-deviation of y, then a good s value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m is the number of datapoints in x, y, and w. default : s=m-sqrt(2*m) if weights are supplied. s = 0.0 (interpolating) if no weights are supplied.
                """
                x = position
                y = time
                dymin=np.unique(np.abs(np.diff(y)))[1]
                print('dymin',dymin)
                stdevyerr=dymin*0.18*1.5#(0.145+0.32)/2##*1.25#np.sqrt(2)
                print('stdev time err = dt_dump =',stdevyerr)
                w = np.ones(len(x))*1/stdevyerr
                m = len(x)
                s = (m+np.sqrt(2*m)) # m+np.sqrt(2*m)
                tck = interpolate.splrep(x, y, w=w,s=s)# tck is BSpline object

                xnew = x#np.arange(0, 2*np.pi, np.pi/50)
                ynew = interpolate.splev(xnew, tck, der=0)
                yder = interpolate.splev(xnew, tck, der=1)
                speed=yder**-1
                propagation_speed=np.array([position,time,speed]).T
            else:
                raise RuntimeError('differentiation mode does not exist')
            propagation_speed_map[l] = np.array(propagation_speed)

        if len(layers)==1:
            return propagation_speed_map[l]
        else:
            return propagation_speed_map


    def get_filtered_propagation_speed(self,dx,a,b,
                                       prop_speed=None,
                                       direction=0,
                                       layer='all',
                                       axis=-1,
                                       padtype='odd',
                                       padlen=None):
        """applies filter to propagation speed

        Args:
            a,b, (array): filter input 
                e.g. b,a = scipy.signal.butter(order,cutoff_f)
                    order (int): 2
                    cutoff_f (float): normalized by the Nyquist frequency
            direction (int): 0 for x direction, 1 for y direction      
            layer='all' (str/float): coordinate perpendicular to crack propagation
            axis : int, optional
                The axis of x to which the filter is applied. Default is -1.
            padtype : str or None, optional
                Must be 'odd', 'even', 'constant', or None. This determines the type of extension to use for the padded signal to which the filter is applied. If padtype is None, no padding is used. The default is 'odd'.
            padlen : int or None, optional
                The number of elements by which to extend x at both ends of axis before applying the filter. This value must be less than x.shape[axis]-1. padlen=0 implies no padding. The default value is 3*max(len(a),len(b)).

        Returns:
            np.array: propagation speed [p, t, speed], where
                        p is the position,
                        t is the time,
                        speed is the propagation speed
        """

        if prop_speed==None:
            prop_speed_map = self.get_propagation_speed(direction,dx,mode='forward_diff',layer=layer)
            if type(prop_speed_map) is not dict:
                prop_speed_map={0:prop_speed_map}
        else: 
            prop_speed_map={0:prop_speed}

        layers =prop_speed_map.keys()
        for layer in layers:
            prop_speed = prop_speed_map[layer].T

            # make sure constant space step
            for p in range(len(prop_speed[0])-2):
                dx2=prop_speed[0,p+2]-prop_speed[0,p+1]
                dx1=prop_speed[0,p+1]-prop_speed[0,p]
                err=abs(dx1-dx2)
                if err>1e-16:
                    raise RuntimeError(err,dx1,dx2,p)
                       
            prop_speed[2] = signal.filtfilt(b, a, prop_speed[2]**-1,axis, padtype, padlen)**-1
            
            #crop
            n_bdr=len(b)*len(a)*2
            print(n_bdr)
            print(prop_speed.shape)
            prop_speed=prop_speed[:,n_bdr:-n_bdr]
            print(prop_speed.shape)
            prop_speed_map[layer] = prop_speed.T
        if len(layers)==1:
            return prop_speed_map[layer]
        else:
            return prop_speed_map

    def get_average_propagation_speed(self,
                                      direction=0,
                                      avg_dist=0.,
                                      wrt_time=False):
        """
        rupture speed averaged perpendicular to propagation direction

        Args:
            direction (int): 0 for x direction, 1 for y direction
            avg_dist (float): average position for the propagation
            wrt_time (bool): whether times occur multiple times

        Returns:
            np.array: propagation speed [p, t, speed], where
                        p is the position,
                        t is the time,
                        speed is the propagation speed

        """
        psm = self.get_propagation_speed(direction,avg_dist)

        cumul_speed = dict()
        value_count = dict()

        # set of time moments that occur several times in a single layer
        double_times = set()
        
        for ps in psm.values():
            sh = ps.shape
            times_in_layer = set()
            for i in range(sh[0]):
                pos = ps[i,0]
                speed = ps[i,2]

                # find times that occure multiple times in this layer
                if wrt_time:
                    pos = ps[i,1]
                    if pos in times_in_layer:
                        double_times.add(pos)
                    times_in_layer.add(pos)

                # cumulate speeds 
                if not pos in cumul_speed:
                    cumul_speed[pos] = 0.
                    value_count[pos] = 0
                cumul_speed[pos] = cumul_speed[pos] + speed
                value_count[pos] = value_count[pos] + 1

        average_speed = list()
        for (p,cs) in cumul_speed.items():
            if p not in double_times:
                average_speed.append([p,cs/float(value_count[p])])

        ind = np.argsort(np.array(average_speed)[:,0])
        average_speed = [average_speed[i] for i in ind]

        return np.array(average_speed)


    def write_strings(self):
        output = list()
        output.append(SPAXCollection.keychar+'ruptureindex')
        output.append(str(self.index))
        spaxc = SPAXCollection.write_strings(self)
        output.extend(spaxc)
        return output

    def read_strings(self,string_list):
        for string in string_list:
            if string.startswith(SPAXCollection.keychar):
                key=string.strip(SPAXCollection.keychar)
                continue
            
            if key == 'ruptureindex':
                self.index = int(string.strip())

        SPAXCollection.read_strings(self,string_list)
            
        
    def __repr__(self, spnb=2):
        """Return a string containing a printable representation of an object.

        Args:
            spnb (int): parameter to control the output

        Returns:
            string:

        """
        out_string = list()
        out_string.append("Rupture {}".format(self.index))
        out_string.append(super(Rupture, self).__repr__(spnb))
        return " ".join(out_string)
        

class RuptureHunter(object):
    """Class used to track ruptures from simulation data

    """
    def __init__(self):
        """Initiates an object

        """
        self.ruptures = dict()
        self.spax_collection = SPAXCollection()
        print(" - RUPTURE HUNTER")

    def __repr__(self, spnb=2):
        """Return a string containing a printable representation of an object.

        Args:
            spnb (int): parameter to control the output

        Returns:
            string:

        """
        out_string = list()
        out_string.append("Rupture Hunter:")
        out_string.extend([rpt.__repr__(spnb+1) for rpt in self.ruptures.values()])
        space = "\n" + " ".join(["" for i in range(spnb)])
        return space.join(out_string)

    def load_check(self,xposition,yposition,time,stick):
        """ Function used to check the validity of the input

        Args:
            position (numpy.memmap):   get data using get_field_at_t_index at idxs = 0
            time (numpy.memmap):       get data using get_full_field and the time FieldId
            sticking(numpy.memmap):    get data using get_full_field and the is_sticking FieldId

        """
        xshape = xposition.shape
        yshape = yposition.shape
        tshape = time.shape
        sshape = stick.shape
        if (xshape[1] != yshape[1]):
            raise("not the same number of nodes per time in xposition and yposition")
        if (xshape[1] != sshape[1]):
            raise("Not the same number of nodes per time in position and stick")
        if (tshape[0] != sshape[0]):
            raise("Not the same number of time steps in time and stick")

    def get_rupture_indexes(self):
        """Function used to get the rupture indexes

        Return:
            list: rupture indexes

        """
        return self.get_sorted_rupture_indexes()

    def get_sorted_rupture_indexes(self):
        """Function used to get the sorted rupture indexes

        Return:
            list: sorted rupture indexes

        """
        indexes = sorted(self.ruptures.keys())
        return indexes

    def get_rupture(self,idx):
        """Function used to get a rupture of a given index

        Args:
            idx (int): rupture index

        """
        return self.ruptures.get(idx)

    def sort_fields(self,xposition,yposition,stick):
        print(xposition.shape,yposition.shape,stick.shape)

        ind = np.lexsort((yposition,xposition))#front[:,1],front[:,0]))
        xposition = np.array([xposition[:,i] for i in ind])[0]#[front[i,:] for i in ind])
        yposition = np.array([yposition[:,i] for i in ind])[0]
        stick = np.array([stick[:,i] for i in ind])[0]
        print(xposition.shape,yposition.shape,stick.shape)

    # for backwards compatibility with 1d ruptures
    def load(self,position,time,stick):
        """Function used to load input from the simulation data

        Args:
            position (numpy.memmap):   get data using get_field_at_t_index at idxs = 0
            time (numpy.memmap):       get data using get_full_field and the time FieldId
            sticking(numpy.memmap):    get data using get_full_field and the is_sticking FieldId

        """
        self.load2D(position,np.zeros_like(position),time,stick)

    # fill the spax_collection with the stick matrix
    def load2D(self,xposition,yposition,time,stick):
        """Function used to load input from the 3D simulation data

        Args:
            xposition (numpy.memmap):   get data using get_field_at_t_index at idxs = 0
            yposition (numpy.memmap):   get data using get_field_at_t_index at idxs = 0
            time (numpy.memmap):        get data using get_full_field and the time FieldId
            sticking(numpy.memmap):     get data using get_full_field and the is_sticking FieldId

        """
        print("   * Load")
        self.load_check(xposition,yposition,time,stick)
        
        nb_pos = len(xposition[0])
        nbp = len(str(nb_pos))
        nb_time = len(time)
        
        for p in range(nb_pos):
            print("     * Position {1:{0}d}/{2:{0}d}".format(nbp,p+1,nb_pos), end='\r')
            sys.stdout.flush()

            x = xposition[0,p]
            y = yposition[0,p]
            gp = (x,y)
            
            ts = 0
            search_start = True
            if not stick[0,p]:
                search_start = False
                
            for t in range(1,nb_time):
                si = stick[t,p]

                # find start of slip period
                if search_start and not si:
                    ts = t
                    search_start = False

                # find end of slip period
                if not search_start and si:
                    sp = SlipPeriod(time[ts],time[t])
                    #sp = SlipPeriod(time[ts,0],time[t,0])
                    self.spax_collection.insert_sp(sp,gp)
                    search_start = True

            if not search_start:
                sp = SlipPeriod(time[ts],time[t])
                self.spax_collection.insert_sp(sp,gp)
                
        print("")

        
    def hunt(self):
        """Function used to track the ruptures

        """
        print("   * Hunt")

        grid_points = self.spax_collection.grid.get_grid_point_list()
        
        # slip period starter
        for gp in grid_points:
            spax = self.spax_collection.getSPAXAtX(gp)
            for sp in spax:
                # slip period is already attributed
                if not sp.rupture_index == -1:
                    continue

                # slip period not attributed yet -> new rupture
                rpt = Rupture()
                rpt.insert_sp(sp,spax.gp)

                # collect ruptures that are connected to this rupture 
                # -> unify them afterwards
                connected_rpts = set()
                
                # search all grid points around the current grid point
                gps = [(gp,[sp])]
                while len(gps):
                    new_gps = list()

                    for s in gps:
                        cgp = s[0] # current grid point
                        csps = s[1] # current slip periods
                        ngps = self.spax_collection.grid.get_neighbor_grid_points(cgp) # neighbor grid points
                        
                        # for each neighbor grid point
                        for ngp in ngps:
                            # spax of neighbor grid point
                            nspax = self.spax_collection.getSPAXAtX(ngp)

                            # find overlaping neighbor slip periods
                            nsps = list() 
                            for nsp in csps:
                                nsps.extend(nspax.get_overlaps(nsp))

                            csp = list()
                            for n in nsps:
                                if n.rupture_index == -1: # this is a slip period not attributed yet
                                    rpt.insert_sp(n,ngp)
                                    csp.append(n)
                                elif n.rupture_index == rpt.index: # this belongs already to this rupture
                                    pass
                                else: # this is a connected rupture -> it will become one rupture afterwards
                                    connected_rpts.add(n.rupture_index)
        
                            if len(csp):
                                new_gps.append((ngp,csp))
                                
                    gps = new_gps

                    
                # add this rupture
                if len(connected_rpts) == 0:
                    self.ruptures[rpt.index] = rpt

                # take all connected ruptures and add it to an
                # anker rupture (chose arbitrarilly)
                else:
                    anker = connected_rpts.pop()
                    self.ruptures.get(anker).add_rupture(rpt)
                    while(len(connected_rpts) != 0):
                        orpt = self.ruptures.pop(connected_rpts.pop())
                        self.ruptures.get(anker).add_rupture(orpt)

                        
    # delete all ruptures up to the length of max_nb_spax
    def free_small_ruptures(self,max_nb_spax):
        """Function used to delete all ruptures up to the indicated length

        Args:
            max_nb_spax (set): the maximal length of ruptures that are deleted

        """
        print("   * Free (all ruptures with nb_spax <= {})".format(max_nb_spax))

        # find small ruptures
        to_free = set()
        for idx, rpt in self.ruptures.items():
            if not len(rpt) > max_nb_spax:
                to_free.add(idx)
        print("     * ruptures to free: {}".format(to_free))

        self.free(to_free)
        

    def free(self,rpt_set):
        """Function used to delete all ruptures given by a set

        Args:
            rpt_set (set): set of rupture index

        """
        print("   * Free rupture.")

        # delete ruptures
        while len(rpt_set):
           self.ruptures.pop(rpt_set.pop())


    def renumber(self):
        """Function used to renumber the indexes of the ruptures

        """
        print("   * Renumber")

        # find start times for all ruptures
        starts = list()
        for idx, rpt in self.ruptures.items():
            starts.append([idx, rpt.get_start()])

        # sort w.r.t starting time of ruptures
        starts = np.array(starts)
        p = starts[:,1].argsort()
        p = [[i, i] for i in p]
        i = np.indices(starts.shape)
        starts = starts[p, i[1]]

        # renumber starting with next rupture index
        for idx in starts[:,0]:
            rpt = self.ruptures.pop(idx)
            rpt.set_rupture_index(Rupture._rid.next())
            self.ruptures[rpt.index] = rpt

        # renumber starting with zero
        Rupture._rid = count(0)
        sorted_ids = sorted(self.ruptures.keys())
        for idx in sorted_ids:
            rpt = self.ruptures.pop(idx)
            rpt.set_rupture_index(Rupture._rid.next())
            self.ruptures[rpt.index] = rpt

    # returns list of rupture expansions
    def get_expansions(self):
        """Function used to get a list of expansion data for all ruptures

        Returns:
            list: expansion data for all rupture

        """
        print("   * Expansions")
        exps = list()
        idxs = self.get_sorted_rupture_indexes()
        for idx in idxs:
            rpt = self.get_rupture(idx)
            exps.append(rpt.get_expansion())
        return exps

    # returns list of sorted fronts
    def get_fronts(self):
        """Function used to get a list of arrays for sorted fronts of all ruptures

        Returns:
            list: arrays for sorted fronts of all ruptures

        """
        fronts = list()
        idxs = self.get_sorted_rupture_indexes()
        for idx in idxs:
            rpt = self.get_rupture(idx)
            fronts.append(rpt.get_sorted_front())
        return fronts

    # returns list of sorted backs
    def get_backs(self):
        """Function used to get a list of arrays for sorted backs of all ruptures

        Returns:
            list: arrays for sorted backs of all ruptures

        """
        backs = list()
        idxs = self.get_sorted_rupture_indexes()
        for idx in idxs:
            rpt = self.get_rupture(idx)
            backs.append(rpt.get_sorted_back())
        return backs
