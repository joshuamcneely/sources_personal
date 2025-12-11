#!/usr/bin/env python
#
# field.py
#
# information of a given field
# 
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @date     2015/09/01
# @modified 2020/12/21

from __future__ import print_function, division, absolute_import

import numpy as np
import os.path

from .utilities import sys_test_dir, path_exists
from .fieldid import FieldId

class Field:
    """Data structure of field data.

    Field stores and manages data in memory-map (np.memmap) file format.

    Attributes:
        identity (FieldId): Identity of field.
        N            (int): Number of points.
        nbts         (int): Number of time steps
        data_type    (str): Data type: int, double, bool, ...
        NEG          (str): Data type: 'N' nodal, 'E' elemental, 'G' global.
        mmap         (str): File name of memory-map of field.
    """

    # Field is either created with all infromation and creates  it mmap name itself
    # or it is created without any information and load_string is used afterwards
    def __init__(self, fldid=None, N=None, nbts=None, tp=None, neg=None):
        """Creates Field object.

        Note: This method only accepts all-none or all-given arguments.

        Args:
            fldid (FieldId): Field identity. Defaults to None.
            N         (int): Number of points. Defaults to None.
            nbts      (int): Number of time steps Defaults to None.
            tp        (str): Data type: int, double, bool, ... . Defaults to None.
            neg       (str): Data type: 'N' nodal, 'E' elemental, 'G' global. Defaults to None.
        
        Raises:
            RuntimeError: When arguments are not all-given or none-given.
        """

        # either all information is provided or none
        if fldid is None and N == nbts == tp == neg == None:
            pass
        else:
            if fldid is None or N == None or nbts == None or tp == None or neg == None:
                print('Field cannot be created with partial information. Either you give all info or none.')
                raise RuntimeError

        self.identity = fldid

        self.N         = N    # number of points (nodes, quads, elements, ...)
        self.nbts      = nbts # number time steps
        self.data_type = tp   # data type: int, double, bool, ...
        self.NEG       = neg  # data type: N nodal, E elemental, G global
        self.mmap      = None # file_name without any path

        # temporary information about mmap location
        self.path  = None # path to MemMap

        if fldid is not None:
            self.create_file_name()

        self.sep    = " "

    def load_string(self,sname):
        """Loads field data from a string.

        Args:
            sname (str): Field data in string format.
        """
        line = sname
        line = line.strip()
        line = line.split(self.sep)
        self.identity = FieldId()
        self.identity.load_string(str(line[0]))
        self.N         = int(line[1])
        self.nbts      = int(line[2])
        self.data_type = str(line[3])
        self.NEG       = str(line[4])
        self.mmap      = str(line[5])

    def get_string(self):
        """Returns field data in string format.

        Returns:
            str: Field data in string format.
        """
        return self.sep.join([self.identity.get_string(),
                              str(self.N),
                              str(self.nbts),
                              str(self.data_type),
                              str(self.NEG),
                              str(self.mmap)])

    def get_id_string(self):
        """Returns identity in string format.

        Retruns:
            str: Identity in string format.
        """
        return self.identity.get_string()

    def create_file_name(self):
        """Creates file name of memory-map.

        This method creates file name of memory-map and stores it.
        """
        self.mmap = self.get_id_string() + ".mmp"

    def set_path(self,path):
        """Sets path of field.

        Args:
            path (str): Path of field.
        """
        self.path = sys_test_dir(path)

    # Once path is set, this is used to get the full path (with file name) to the MemMap
    def get_memmap_path(self):
        """Returns path to memory-map.

        Retruns:
            str: Path to memory-map.
        """
        if self.path is None:
            print('Path of Field has not been set yet. Use first the "set_path" function')
        return self.path + self.mmap

    def memmap_exists(self):
        """Determines if memory-map exists.

        Retruns:
            bool: True is memory-map exists.
        """
        return path_exists(self.mmap,self.path)

    def get_memmap(self,mode='r'):
        """Returns memory-map of field.

        Args:
            mode (str): 'r'  Open existing file for reading only.
                        'r+' Open existing file for reading and writing.
                        'w+' Create or overwrite existing file for reading and writing.
                        'c'  Copy-on-write: assignments affect data in memory, but changes are not saved to disk. The file on disk is read-only.
                        Defaults to 'r'.

        Returns:
            mmap: Memory-map of field.
        """
        return np.memmap(self.get_memmap_path(),
                         dtype=self.data_type,
                         mode=mode,
                         shape=(self.nbts, self.N))

    def get_shape(self):
        """Returns number of points and number of time steps.

        Returns:
            int: Number of points.
            int: Number of time steps.
        """
        return (self.nbts, self.N)

    def change_identity(self,new_id):
        """Changes identity of field.

        This method changes the identity of field and the file name of memory-map.

        Args:
            new_id (FieldId): New identity.

        Raises:
            RuntimeError: When the memory-map with new identity is already exists.
        """
        old_mmap = self.mmap
        old_path = self.get_memmap_path()

        self.identity = new_id
        self.create_file_name()
        
        # check that we do not override existing MemMap
        if self.memmap_exists():
            print('Want to rename MemMap ',old_mmap,' to ',self.mmap,
                  ' but it exists already!')
            raise RuntimeError

        os.rename(old_path,
                  self.get_memmap_path())

    def crop_at_time_step(self,cts): 
        """Crops at time step. All time steps after cts will be deleted.

        Args:
            cts (int): time step at which to crop
        """
        if not self.memmap_exists():
            self.nbts = min(cts,self.nbts)

        else:
            # rename old memmap file to temporary file
            tmp_path = self.get_memmap_path() + ".tmp"
            os.rename(self.get_memmap_path(),
                      tmp_path)

            # get access to old memmap
            old_nbts = self.nbts
            old_mmap = np.memmap(tmp_path,
                                 dtype=self.data_type,
                                 mode='r',
                                 shape=(old_nbts, self.N))

            # change number of time steps
            self.nbts = min(cts,self.nbts)
            mmap = self.get_memmap('w+')
            mmap[:,:] = old_mmap[:self.nbts,:]
            os.remove(tmp_path)

    def filter(self,fltr):
        """filter data based on a filter. applied to all time steps

        Args:
            fltr (array): booleans where to crop: True is kept

        """
        if self.memmap_exists() and self.NEG != 'G':
            # rename old memmap file to temporary file
            tmp_path = self.get_memmap_path() + ".tmp"
            os.rename(self.get_memmap_path(),
                      tmp_path)

            # get access to old memmap
            old_mmap = np.memmap(tmp_path,
                                 dtype=self.data_type,
                                 mode='r',
                                 shape=(self.nbts, self.N))

            # change number of nodes
            self.N = np.sum(fltr)
            mmap = self.get_memmap('w+')
            mmap[:,:] = old_mmap[:,fltr]
            os.remove(tmp_path)

    def destroy_memmap(self):
        """Removes memory-map of field.
        
        This method removes memory-map of field if it exists.
        """
        if self.memmap_exists():
            os.remove(self.get_memmap_path())

    def __repr__(self):
        """Returns a printable representation of field.

        Returns:
            str: String with information of field.
        """
        return "Field: id={0} N={1} nbts={2} type={3} neg={4} file={5}".format(self.identity.__repr__(),
                                                                               self.N,
                                                                               self.nbts,
                                                                               self.data_type,
                                                                               self.NEG,
                                                                               self.mmap)

