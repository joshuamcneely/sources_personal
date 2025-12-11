#!/usr/bin/env python
#
# fieldcollectionanalysis.py
#
# Code to get data for different criterion
#
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @date     2013/05/21
# @modified 2020/12/22

from __future__ import print_function, division, absolute_import

import numpy as np
import math as mt
import types
import sys
import copy
from collections.abc import Iterable

from .fieldid import FieldId
from .field import Field
from .field_collection import FieldCollection


class FieldCollectionAnalysis(FieldCollection):
    """IO helper extension of FieldCollection.

    FieldCollectionIOHelper inherits FieldCollection

    Attributes:
        print_info
    """

    # used for "casting" of FieldCollection
    def __new__(cls, other):
        """Casts FieldCollection to FieldCollectionAnalysis.

        Args:
            other (FieldCollection): The object to copy from.
        
        Returns:
            FieldCollectionAnalysis: Casted instance.
        """
        if isinstance(other, FieldCollection):
            other = copy.copy(other)
            other.__class__ = FieldCollectionAnalysis
            return other
        return object.__new__(cls)

    def __init__(self, other):
        """Creates FieldCollectionAnalysis object.

        This method is called after __new__(cls, other) and initiates 
        a FieldCollectionAnalysis object.

        Args:
            other (FieldCollection): The object to copy from.
        """
        self.print_info = False

    # simplifies MemMaps, to one dimension if the second dimension is of length 1
    def get_simplified_memmap(self,fldid):
        """Gets memory map with reduced dimension if the second dimension is of length 1.

        Args:
            fldid (FieldId): FieldId of field.

        Returns:
             memmap: Memory map with reduced dimension.
        """
        mmap = self.get_field_memmap(fldid)
        
        fld = self.get_field(fldid)
        if fld.NEG == 'G':
            mmap = mmap[:,0]

        return mmap

    def get_full_field(self,fldid):
        """Gets memory map of a field.

        Args:
            fldid (FieldId): FieldId of field.

        Returns:
             memmap: Memory map of a field at all time.
        """
        return self.get_simplified_memmap(fldid)

    def get_field_at_t(self,fldid,tm,ts):
        """Gets a field at time.

        Args:
            fldid   (FieldId): FieldId of field.
            tm      (FieldId): Time field identity.
            ts (list(double)): Time.

        Returns:
             memmap: Memory map of a field at a time.
        """
        if not isinstance(ts, Iterable):
            ts = list([ts])
        idxs = self.get_index_of_closest_time(tm,ts)
        return self.get_field_at_t_index(fldid,idxs)

    def get_field_at_t_index(self,fldid,idxs):
        if not isinstance(idxs, Iterable):
            idxs = list([idxs])
        fld = self.get_full_field(fldid)[idxs]
        return fld

    def get_field_at_node_index(self,fldid,idxs):
        if not isinstance(idxs, Iterable):
            idxs = list([idxs])
        fld = self.get_full_field(fldid)[:,idxs]
        return fld

    # gets the time/step indexs for a list of indicators
    # last = last plot
    # middle = middle plot
    # first = 0
    def get_t_index(self,indicators):
        """Gets the time/step indexs for a list of indicators.

        Args:
            indicators (str): 'last'   last time index.
                              'middle' middle time index.
                              'first'  first time index (0).
        Returns:
            list(int): Time index.
        """

        is_list = True
        if type(indicators) is str:
            indicators = list([indicators])
            is_list = False

        mm = self.get_field_memmap(FieldId('step'))[:,0]

        index = list()
        for i in indicators:
            if i == 'last':
                index.append(len(mm) - 1)
            elif i == 'middle':
                index.append(int((len(mm) - 1)/2))
            elif i == "first":
                index.append(0)
            else:
                raise("Don't know this T indicator")

        if not is_list:
            index = index[0]

        return index

    def get_indices_of_nodes_on_line(self,fldid,vlu,tolerance=0.0):
        """Returns only indices of nodes that within tolerance of this value.

        Args:
            fldid (FieldId): FieldId of field.
            vlu    (double): Target value.
        KwArgs:
            tolerance=0.0
        Returns:
            list(int): Indices of nodes that are within tolerance of this value.
        """
        
        mm = self.get_field_memmap(fldid)[0,:]

        # find closest value
        vlu = mm[np.argmin(np.abs(mm-vlu))]
        print(vlu)
        
        # get values within tolderance
        indices = np.where(np.abs(mm-vlu)<tolerance)
        
        # flatten and cast to list
        indices = list(np.array(indices).flatten())
        
        return indices

        
    def get_index_of_closest_position(self,fldid,vlu):
        """Gets index of closest position.

        Args:
            fldid    (FieldId): FieldId of field.
            vlu (list(double)): Target position.

        Returns:
             list(int): Indexs with closest position.

        Raises:
            Exception: When lengths of fldid and vlu mismatch.
        """
        if type(fldid) is not list:
            fldid = [fldid]
        if type(vlu) not in [list,type(np.array([]))]:
            vlu = [vlu]
        if not len(vlu) == len(fldid):
            print('vlu',len(vlu),vlu,'fldid',len(fldid),fldid)
            raise RuntimeError("Need the same number of vlu as fldid")

        mms = list()
        nb_nodes = sys.maxsize
        for f in fldid:
            nb_nodes = min(nb_nodes, self.get_field(f).N)
            mms.append(self.get_field_memmap(f)[0,:].tolist())
            
        index = 0
        min_dist = float("inf")
        for i in range(nb_nodes):
            dist = 0
            for d in range(len(mms)):
                dist += (mms[d][i] - vlu[d]) * (mms[d][i] - vlu[d])
            dist = mt.sqrt(dist)
            if dist < min_dist:
                index = i
                min_dist = dist
        if self.print_info:
            print("For {0}, we found closest value: {1}".format(" ".join([str(v) for v in vlu]),
                                                                " ".join([str(m[index]) for m in mms])))
        return index

    # returns the index of the closest value in time_steps or time
    def get_index_of_closest_time(self,fldid,vlus):
        """Returns the index of the closest value in time_steps or time.

        Args:
            fldid     (FieldId): FieldId of time.
            vlus (list(double)): List of time.

        Retruns:
            list(int): Index of the closest value in time_steps or time.
        """
        is_list = True
        if not isinstance(vlus, Iterable):
            vlus = list([vlus])
            is_list = False

        mm = self.get_field_memmap(fldid)[:,0]
        idxs = list()
        for vlu in vlus:
            ix = np.argmin(np.absolute(mm-vlu))
            idxs.append(ix)

            # printing info
            result_info = "For {0}, we found closest value: {1} (".format(vlu, mm[ix])
            if ix > 0 :
                result_info += "{} ".format(mm[ix-1])
            if ix < len(mm)-1:
                result_info += "{}".format(mm[ix+1])
            result_info += ")"
            if self.print_info:
                print(result_info)

        if not is_list:
            idxs = idxs[0]

        return idxs


# -------------------------------------------------------------------------
    def get_sliced_x_sliced_y_plot(self, xfldid, xslice, yfldid, yslice, zfldid, tidx):
        """Gets structured data for xy plotting at time with sliced x and y.

        Args:
            xfldid      (FieldId): FieldId of x position.
            xslice    (list(int)): Sliced indices of x.
            yfldid      (FieldId): FieldId of y position.
            yslice    (list(int)): Sliced indices of y.
            zfldid      (FieldId): FieldId of z position.
            tidx            (int): Time index.
        
        Return:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured Y 2D np.array for plotting.
            np.array: Structured Z 2D np.array for plotting.
        """

        X,Y,Z = self.get_xy_plot(xfldid, 
                                 yfldid,
                                 zfldid,
                                 tidx)
        X = X[xslice,:][:,yslice]
        Y = Y[xslice,:][:,yslice]
        Z = Z[xslice,:][:,yslice]
        return X,Y,Z

    def get_sliced_xy_plot(self, xfldid, xslice, yfldid, zfldid,tidx):
        """Gets structured data for xy plotting at time with sliced x.

        Args:
            xfldid      (FieldId): FieldId of x position.
            xslice    (list(int)): Sliced indices of x.
            yfldid      (FieldId): FieldId of y position.
            zfldid      (FieldId): FieldId of z position.
            tidx            (int): Time index.
        
        Return:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured Y 2D np.array for plotting.
            np.array: Structured Z 2D np.array for plotting.
        """

        X,Y,Z = self.get_xy_plot(xfldid,
                                 yfldid,
                                 zfldid,
                                 tidx)
        X = X[xslice,:][:,:]
        Y = Y[xslice,:][:,:]
        Z = Z[xslice,:][:,:]
        return X,Y,Z

    def get_x_sliced_y_plot(self, xfldid, yfldid, yslice, zfldid, tidx):
        """Gets structured data for xy plotting at time with sliced x.

        Args:
            xfldid      (FieldId): FieldId of x position.
            xslice    (list(int)): Sliced indices of x.
            yfldid      (FieldId): FieldId of y position.
            zfldid      (FieldId): FieldId of z position.
            tidx            (int): Time index.
        
        Return:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured Y 2D np.array for plotting.
            np.array: Structured Z 2D np.array for plotting.
        """

        X,Y,Z = self.get_xy_plot(xfldid,
                                 yfldid,
                                 zfldid,
                                 tidx)
        X = X[:,:][:,yslice]
        Y = Y[:,:][:,yslice]
        Z = Z[:,:][:,yslice]
        return X,Y,Z

    def get_sorted_xy_plot(self,X,Y,F):
        """Gets structured data for xy plot
        Args:
            np.array: Unstructured X 1D np.array.
            np.array: Unstructured Y 1D np.array.
            np.array: Unstructured V 1D np.array.
        Returns:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured Y 2D np.array for plotting.
            np.array: Structured V 2D np.array for plotting.

        """
        print('sort')
        print(X.shape,Y.shape,F.shape)
        p = X.argsort()
        X = X[p]
        Y = Y[p]
        F = F[p]

        Xr = X[::-1]
        Xmin = Xr[-1]
        lng = len(Xr) - np.where(Xr==Xmin)[0][0]
        X.shape = [-1,lng]
        Y.shape = [-1,lng]
        F.shape = [-1,lng]

        soi = np.argsort(Y, axis=1)
        sti = np.indices(Y.shape)
        X = X[sti[0], soi]
        Y = Y[sti[0], soi]
        F = F[sti[0], soi]
        
        return X,Y,F
    
    def get_xy_plot(self,xfldid,yfldid,zfldid,tidx):
        """Gets structured data for xy plotting at time.

        Args:
            xfldid (FieldId): FieldId of x position.
            yfldid (FieldId): FieldId of y position.
            zfldid (FieldId): FieldId of z position.
            tidx       (int): Time index.
        
        Return:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured Y 2D np.array for plotting.
            np.array: Structured Z 2D np.array for plotting.
        """
        if len(self.get_field_memmap(xfldid)[:,0])==1:
            print("position field constant in time")
            X = self.get_field_memmap(xfldid)[0,:]
            Y = self.get_field_memmap(yfldid)[0,:]
        else:
            X = self.get_field_memmap(xfldid)[tidx,:]
            Y = self.get_field_memmap(yfldid)[tidx,:]

        Z = self.get_field_memmap(zfldid)[tidx,:]
        
        return self.get_sorted_xy_plot(X,Y,Z)

    def get_nb_points_in_direction(self,xfldid,tidx=0):
        """Gets number of points in direction.

        Args:
            xfldid (FieldId): FieldId of direction.
            tidx       (int): Index of time. Defaults to 0.
            
        Returns:
            int: Number of points in direction.
        """
        X = self.get_field_memmap(xfldid)[tidx,:]

        lng = len(np.unique(X))

        if lng == 1:
                print(" ** Warning in get_nb_points_in_direction:",
                      "In this field all values are the same.",
                      "Maybe you are looking at a plane in direction!")

        return lng
        

# -------------------------------------------------------------------------
    def prepare_xt_plot(self,X,T,V,tslice=None):
        """Prepares x-t plot.

        Args:
            X     (np.array): Structured X 2D np.array for plotting.
            T     (np.array): Structured T 2D np.array for plotting.
            V     (np.array): Structured V 2D np.array for plotting.
            tslice list(int): Sliced indices of time.

        Returns:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured T 2D np.array for plotting.
            np.array: Structured V 2D np.array for plotting.
        """

        if X.shape[0] == 1:
            X = np.tile(X,(T.shape[0],1))

        if tslice is not None:
            X = X[tslice,:]
            T = T[tslice,:]
            V = V[tslice,:]

        T = T.repeat(X.shape[1],1)

        soi = np.argsort(X, axis=1)
        sti = np.indices(X.shape)
        X = X[sti[0], soi]
        V = V[sti[0], soi]

        return X,T,V
    
    def get_xt_plot(self,xfldid,tfldid,vfldid):
        """Gets structured data for x-t plotting.

        Args:
            xfldid (FieldId): FieldId of x position.
            tfldid (FieldId): FieldId of time.
            vfldid (FieldId): FieldId of value.
        
        Returns:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured T 2D np.array for plotting.
            np.array: Structured V 2D np.array for plotting.
        """
        X = self.get_field_memmap(xfldid)
        T = self.get_field_memmap(tfldid)
        V = self.get_field_memmap(vfldid)
        return self.prepare_xt_plot(X,T,V)

    def get_sliced_x_sliced_t_plot(self, xfldid, xslice, tfldid, tslice, vfldid):
        """Gets structured data for x-t plotting with sliced x and t.

        Args:
            xfldid      (FieldId): FieldId of x position.
            xslice    (list(int)): Sliced indices of x.
            tfldid      (FieldId): FieldId of time.
            tslice    (list(int)): Sliced indices of t.
            vfldid      (FieldId): FieldId of value.
            tidx            (int): Time index.

        Returns:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured T 2D np.array for plotting.
            np.array: Structured V 2D np.array for plotting.
        """
        X = self.get_field_memmap(xfldid)
        T = self.get_field_memmap(tfldid)
        V = self.get_field_memmap(vfldid)

        X,T,V = self.prepare_xt_plot(X,T,V,tslice)

        X = X[:,xslice]
        T = T[:,xslice]
        V = V[:,xslice]
        
        return X,T,V

    def get_x_sliced_t_plot(self, xfldid, tfldid, tslice, vfldid):
        """Gets structured data for x-t plotting with sliced t.

        Args:
            xfldid      (FieldId): FieldId of x position.
            tfldid      (FieldId): FieldId of time.
            tslice    (list(int)): Sliced indices of t.
            vfldid      (FieldId): FieldId of value.
            tidx            (int): Time index.

        Returns:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured T 2D np.array for plotting.
            np.array: Structured V 2D np.array for plotting.
        """
        X = self.get_field_memmap(xfldid)
        T = self.get_field_memmap(tfldid)
        V = self.get_field_memmap(vfldid)

        return self.prepare_xt_plot(X,T,V,tslice)

    def get_sliced_xt_plot(self, xfldid, xslice, tfldid, vfldid):
        """Gets structured data for x-t plotting with sliced x.

        Args:
            xfldid      (FieldId): FieldId of x position.
            xslice    (list(int)): Sliced indices of x.
            tfldid      (FieldId): FieldId of time.
            vfldid      (FieldId): FieldId of value.
            tidx            (int): Time index.

        Returns:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured T 2D np.array for plotting.
            np.array: Structured V 2D np.array for plotting.
        """
        X = self.get_field_memmap(xfldid)[:,:]
        T = self.get_field_memmap(tfldid)[:,:]
        V = self.get_field_memmap(vfldid)[:,:]

        X,T,V = self.prepare_xt_plot(X,T,V)

        X = X[:,xslice]
        T = T[:,xslice]
        V = V[:,xslice]
        
        return X,T,V

    def get_xy_plot_at_node_index(self,
                                  xfldid,
                                  yfldid,
                                  vfldid, tidx, idcs):
        """Gets structured data for x-t plotting with sliced t at a node.

        Args:
            xfldid      (FieldId): FieldId of x position.
            xslice    (list(int)): Sliced indices of x.
            yfldid      (FieldId): FieldId of y position.
            yslice    (list(int)): Sliced indices of y.
            vfldid      (FieldId): FieldId of value.
            tidx            (int): Time indes.
            idcs            (int): Node index.

        Returns:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured Y 2D np.array for plotting.
            np.array: Structured F 2D np.array for plotting.
        """
        X = self.get_field_memmap(xfldid)[tidx,idcs]
        Y = self.get_field_memmap(yfldid)[tidx,idcs]
        F = self.get_field_memmap(vfldid)[tidx,idcs]

        return self.get_sorted_xy_plot(X,Y,F)
    
    def get_x_sliced_t_plot_at_node_index(self, xfldid, tfldid, tslice, vfldid, idcs):
        """Gets structured data for x-t plotting with sliced t at a node.

        Args:
            xfldid      (FieldId): FieldId of x position.
            tfldid      (FieldId): FieldId of time.
            tslice    (list(int)): Sliced indices of t.
            vfldid      (FieldId): FieldId of value.
            idcs            (int): Node index.

        Returns:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured T 2D np.array for plotting.
            np.array: Structured V 2D np.array for plotting.
        """
        X = self.get_field_memmap(xfldid)[:,idcs]
        T = self.get_field_memmap(tfldid)
        V = self.get_field_memmap(vfldid)[:,idcs]

        return self.prepare_xt_plot(X,T,V,tslice)
        
    def get_sliced_x_sliced_t_plot_at_node_index(self, xfldid, xslice, tfldid, tslice, vfldid, idcs):
        """Gets structured data for x-t plotting with sliced t at a node.

        Args:
            xfldid      (FieldId): FieldId of x position.
            xslice    (list(int)): Sliced indices of x.
            tfldid      (FieldId): FieldId of time.
            tslice    (list(int)): Sliced indices of t.
            vfldid      (FieldId): FieldId of value.
            idcs            (int): Node index.

        Returns:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured T 2D np.array for plotting.
            np.array: Structured V 2D np.array for plotting.
        """
        X = self.get_field_memmap(xfldid)[:,idcs]
        T = self.get_field_memmap(tfldid)
        V = self.get_field_memmap(vfldid)[:,idcs]

        X,T,V = self.prepare_xt_plot(X,T,V,tslice)

        X = X[:,xslice]
        T = T[:,xslice]
        V = V[:,xslice]
        
        return X,T,V


    @staticmethod
    def make_pretty(X,Y):
        """Makes plot pretty ???

        Args:
            X (np.array): Structured X 2D np.array for plotting.
            Y (np.array): Structured Y 2D np.array for plotting.

        Returns:
            np.array: Structured X 2D np.array for plotting.
            np.array: Structured Y 2D np.array for plotting.
        """

        if not X.shape == Y.shape:
            print("X and Y do not have the same shape!!.",
                  "make_pretty does not work.")
            
        Xn = np.empty((X.shape[0]+1, X.shape[1]+1))
        Yn = np.empty((X.shape[0]+1, X.shape[1]+1))

        Xn[ 0, :Y.shape[1]] = X[ 0,:]
        Xn[-1, :Y.shape[1]] = X[-1,:]
        Xn[ 0,-1] = X[ 0,-1]
        Xn[-1,-1] = X[-1,-1]

        Yn[:Y.shape[0],  0] = Y[:, 0]
        Yn[:Y.shape[0], -1] = Y[:,-1]
        Yn[-1, 0] = Y[-1, 0]
        Yn[-1,-1] = Y[-1,-1]

        for i in range(1,Xn.shape[0]-1):
            for j in range(Xn.shape[1]):
                Xn[i,j] = 0.5 * (X[i-1,0] + X[i,0])

        for i in range(Yn.shape[0]):
            for j in range(1,Yn.shape[1]-1):
                Yn[i,j] = 0.5 * (Y[0,j-1] + Y[0,j])
                
        return Xn,Yn
