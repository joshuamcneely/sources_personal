#!/usr/bin/env python
#
# datamanager.py
#
# Code to manage data from simulations
#
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @date     2015/09/01
# @modified 2020/12/22

from __future__ import print_function, division, absolute_import

import os.path
import shutil as shtl

from .utilities import sys_test_dir, path_exists, string_is_int
from .info_folder_handler import InfoFolderHandler
from .fieldid import FieldId
from .field import Field
from .field_collection import FieldCollection


class DataManager(InfoFolderHandler):
    """Manages data from simulations

    DataManager inherits InfoFolderHandler to handle simulation data.

    Attributes:
        field_collections (set(FieldCollection)): Set of FieldCollection.
        supplementary     (set(str))            : Set of supplementary file names.
    """
    
    def __init__(self,name,wdir='.',create=False):
        """Creates DataManager object.

        Args:
            name   (str) : Job name.
            wdir   (str) : Path to postprocessed data.
            create (bool): Whether or not to create new files. Defaults to False.
        """

        self.field_collections = set()
        self.supplementary     = set()

        InfoFolderHandler.__init__(self,name,wdir,create)

        self.check_data_completeness()

    # short access
    def __call__(self, fcname):
        return self.get_field_collection(fcname)

    def check_data_completeness(self,**kwargs):
        """Checks data completeness.

        This method checks the existance of field collections and supplementary files
        and prints out warning if somthing is missing.

        Note: This method does not check completeness if data is packed.
        """
        # cannot check if packed
        if self.is_packed():
            if kwargs.get('verbose',False):
                print('Do not check if FieldCollections and Supplementary exist because this data is packed.')
            return

        # check if field collections are there
        for fcn in self.field_collections:
            try:
                fc = FieldCollection(fcn,self.get_data_folder_path(),False)
            except IOError:
                print('WARNING: FieldCollection named "'+fcn
                      +'" is missing!')

        # check if supplementary files are there
        for sp in self.supplementary:
            if not path_exists(sp,self.get_data_folder_path()):
                print('WARNING: supplementary file named "'+sp
                      +'" is missing!')

    def __contains__(self,item):
        """Allows the usage of the {in} operator.

        Returns: 
            Items in {field_collections} or {supplementary}. False if both are empty.
        """
        return item in self.field_collections or item in self.supplementary

    def destroy(self):
        """Destroys object and associated files.
        
        This method removes info file and data folder, then cleans up object.
        """
        InfoFolderHandler.destroy(self)
        
        # clean itself
        self.field_collections.clear()
        self.supplementary.clear()


    def get_field_collection(self,fcname):
        """Returns field collection of requested field collection name.

        Args:
            fcname (str): Name of field collection.

        Returns:
            FieldCollection: Requested field collection.

        Raises:
            RuntimeError: When the requested name does not exist in DataManager.
        """
        self.stop_here_if_packed()
        if fcname not in self.field_collections:
            print('FieldCollection named "{}" does not exist in DataManager {}'.format(fcname,
                                                                                       self.name))
            raise RuntimeError

        return FieldCollection(fcname,self.get_data_folder_path(),False)

    def get_all_field_collections(self):
        """Returns all field collections.

        Args:
            None

        Returns:
            list[FieldCollection]: list of field collection
        """
        fcs = list()
        for fcn in self.field_collections:
            fcs.append(FieldCollection(fcn,self.get_data_folder_path(),False))
        return fcs
        

    def get_new_field_collection(self,fcname):
        """Adds new field collection.

        Args:
            fcname (str): Name of field collection.

        Returns:
            FieldCollection: Requested field collection.

        Raises:
            RuntimeError: When the requested name is already exists in DataManager.
        """
        self.stop_here_if_packed()
        if fcname in self.field_collections:
            print('FieldCollection named "{}" exists already in DataManager {}'.format(fcname,
                                                                                       self.name))
            raise RuntimeError

        fc = FieldCollection(fcname,self.get_data_folder_path(),True)
        fc.destroy() # clean if a FieldCollection of this name existed in the data folder
        fc = FieldCollection(fcname,self.get_data_folder_path(),True)
        self.field_collections.add(fcname)
        self.write()

        return fc
        
    def remove_field_collection(self,fcname):
        """Removes a field collection.

        Args:
            fcname (str): Name of field collection.

        Raises:
            IOError: When the requested name does not exist in DataManager.
        """
        self.stop_here_if_packed()
        if fcname in self.field_collections:
            try:
                fc = FieldCollection(fcname,self.get_data_folder_path())
            except IOError:
                print('INFO: No data for FieldCollection named "{}" in DataManager "{}".'.format(fcname,
                                                                                                 self.name))
            else:
                fc.destroy()
            self.field_collections.discard(fcname)
            self.write()
        

    # add supplementary file. replace it if exists already?
    def add_supplementary(self,sname,fpath,replace=False):
        """Adds a supplementary file.

        Args:
            sname    (str): Name of supplementary file.
            fpath    (str): Path to supplementary file.
            replace (bool): Whether or not to replace if it already exists. Defaults to False.
        """
        self.stop_here_if_packed()
        # check if sname exist already
        if sname in self.supplementary and not replace:
            print('Supplementary with name '+sname+' exist already. '
                  +'If you want to replace it, use: add_supplementary(name,fpath,True)')
        # add and/or replace it
        else:
            shtl.copyfile(fpath,self.get_data_folder_path()+sname)
            self.supplementary.add(sname)
            self.write()

    # returns path to supplementary file
    def get_supplementary(self,sname):
        """Returns path of supplementary file of requested supplementary file name.

        Args:
            sname (str): Name of supplementary file.

        Returns:
            str: Content of requested supplementary file.

        Raises:
            RuntimeError: When the requested name does not exist in DataManager.
        """
        self.stop_here_if_packed()
        if not sname in self.supplementary:
            print('No supplementary with name '+sname+' in datamanager '+self.name)
            raise RuntimeError
        return self.get_data_folder_path()+sname

    # content of supplementary file is returned as (single) string
    def get_supplementary_content(self,sname):
        """Returns supplementary file of requested supplementary file name.

        Args:
            sname (str): Name of supplementary file.

        Returns:
            str: Content of requested supplementary file.

        Raises:
            RuntimeError: When the requested name does not exist in DataManager.
        """
        with open(self.get_supplementary(sname),'r') as fl:
            content = fl.read()
        return content

    def remove_supplementary(self,sname):
        """Removes a supplementary file.

        This method removes a supplementary file from DataManager and from data folder.
        Does nothing if the requested supplementary file does not exist in the data folder.

        Args:
            sname (str): Name of supplementary file.
        """
        self.stop_here_if_packed()
        try:
            os.remove(self.get_data_folder_path()+sname)
        except OSError:
            pass # if file does not exist, do nothing
        self.supplementary.discard(sname)
        self.write()


    def write_to_file(self,fl):
        """Writes information of DataManager into file.
        
        This method writes names of field collections, supplementary files, 
        and the path to data folder into file.

        Args:
            fl (file): File to be written.
        """
        print(InfoFolderHandler.keychar+'field-collections', file=fl)
        for gr in self.field_collections:
            print(gr, file=fl)
            
        print(InfoFolderHandler.keychar+'supplementary', file=fl)
        for sp in self.supplementary:
            print(sp, file=fl)

        # for multiple inheritance
        InfoFolderHandler.write_to_file(self,fl)
        

    def read_from_file(self,key,line):
        """Stores value if the key string is valid.

        Args:
            key  (str): Key string.
            line (str): Value string.

        Returns:
            bool: True if successful.
        """
        knows_key = False

        if key == 'field-collections':
            self.field_collections.add(line)
            knows_key = True
        elif key == 'supplementary':
            self.supplementary.add(line)
            knows_key = True

        # for multiple inheritance
        else:
            knows_key = InfoFolderHandler.read_from_file(self,key,line)

        return knows_key

    def create_info_file_name(self,name):
        """Returns info file name.

        Returns:
            str: Info file name.
        """
        return name + '.datamanager.info'

    def create_data_folder_name(self,name):
        """Returns data folder name.

        Returns:
            str: Data folder name.
        """
        return name + '-datamanager-files'

