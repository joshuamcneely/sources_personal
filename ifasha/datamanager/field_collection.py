#!/usr/bin/env python
#
# fieldcollection.py
#
# collection of fields
#
# WARNING: if a field is accessed directly, it should not be modified, 
# as the FieldCollection would not know about it and get corrupted.
#
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @date     2015/09/01
# @modified 2020/12/22

from __future__ import print_function, division, absolute_import

import shutil as shtl

from .info_folder_handler import InfoFolderHandler
from .fieldid import FieldId
from .field import Field


class FieldCollection(InfoFolderHandler):
    """Collection of Field.

    FieldCollection inherits InfoFolderHandler 

    Attributes:
        fields (dict(Field)): 
        sim_info       (str):
    """

    def __init__(self,name,wdir='.',create=False):
        """Creates FieldCollection object.

        Args:
            name   (str) : Job name.
            wdir   (str) : Path to postprocessed data. Defaults to '.'.
            create (bool): Whether or not to create new files. Defaults to False.
        """
        # maps field-id-string to field
        self.fields = dict()

        # information about simulation
        self.sim_info = ''

        InfoFolderHandler.__init__(self,name,wdir,create)
        
        self.load_fields()
#        self.check_data_completeness()

#    def check_data_completeness(self):
#        for fld in self.fields.itervalues():
#            if not fld.memmap_exists():
#                print('WARNING: MemMap of Field named "'+fld.get_id_string()
#                      +'" does not exist. Is it missing?')
                

    def destroy(self):
        """Destroys object and associated files.
        
        This method removes info file and data folder, then cleans up object.
        """
        InfoFolderHandler.destroy(self)

        # no nead to destroy memmaps as folder is deleted with every thing in it
        self.fields.clear()

    def write_to_file(self,fl):
        """Writes information of FieldCollection into file.
        
        This method writes information of simulation, information of fields, 
        and the path to data folder into file.

        Args:
            fl (file): File to be written.
        """
        # write information about simulation
        print(InfoFolderHandler.keychar+'simulation_info', file=fl)
        print('{}'.format(self.sim_info), file=fl)

        # write Fields information
        print(InfoFolderHandler.keychar+'fields', file=fl)
        for fs in self.fields.values():
            print(fs.get_string(), file=fl)

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

        # read Fields
        if key == 'fields':
            fld = Field()
            fld.load_string(line)
            self.fields[fld.get_id_string()] = fld
            knows_key = True

        # read simulation information
        elif key == 'simulation_info':
            self.sim_info = line
            knows_key = True

        # for multiple inheritance
        else:
            knows_key = InfoFolderHandler.read_from_file(self,key,line)

        return knows_key

    def load_fields(self,**kwargs):
        """Sets path to fields and checks if the corresponding memory-map exists.

        This method sets path to fields. And if the corresponding memory-map exists, it prints warning message.
        """
        if kwargs.get('verbose',False) and self.is_packed():
            print('Do not check if MemMaps exist because this data is packed.')
        for fs in self.fields.values():
            fs.set_path(self.get_data_folder_path())
            if not self.is_packed() and not fs.memmap_exists():
                print('WARNING: MemMap "{}" for Field "{}" does not exist!'.format(fs.mmap,
                                                                                   fs.identity))


    def get_field_memmap(self,field_id,mode='r'):
        """Returns memory-map of a field.

        Args:
            field_id (FieldId): Identity of field.
            mode         (str): 'r'  Open existing file for reading only.
                                'r+' Open existing file for reading and writing.
                                'w+' Create or overwrite existing file for reading and writing.
                                'c'  Copy-on-write: assignments affect data in memory, but changes are not saved to disk. The file on disk is read-only.
                                Defaults to 'r'.

        Returns:
            mmap: Memory-map of field.

        Raises:
            RuntimeError: If data is packed into archive.
        """
        self.stop_here_if_packed()
        fld = self.get_field(field_id)
        return fld.get_memmap(mode)

    def has_field(self,field_id):
        """Determines if a field exists .

        Args:
            field_id (FieldId): Identity of field.

        Returns:
            bool: True if the field exists.
        """
        fid = field_id.get_string()
        return fid in self.fields

    def get_field_shape(self,field_id):
        """Returns number of points and number of time steps of a field.
        
        Args:
            field_id (FieldId): Identity of field.

        Returns:
            int: Number of points.
            int: Number of time steps.
        """
        fld = self.get_field(field_id)
        return fld.get_shape()

    # Field should not be modified as the FieldCollection would not know about it
    # hence it would be corrupted
    def get_field(self,field_id):
        """Returns a field.

        Note: Field should not be modified as the FieldCollection would not know about it
              hence it would be corrupted.

        Args:
            field_id (FieldId): Identity of field.

        Returns:
            Field: The requested field.

        Raises:
            RuntimeError: If data is packed into archive.
        """
        self.stop_here_if_packed()
        fid = field_id.get_string()
        return self.fields[fid]

    def get_all_fields(self):
        """Gets all fields.

        Args:
            None

        Returns:
            list[Fields]: list of fields

        Raises:
            RuntimeError: If data is packed into archive.
        """
        self.stop_here_if_packed()
        flds = list()
        for fs in self.fields.keys():
            fid = FieldId()
            fid.load_string(fs)
            flds.append(self.get_field(fid))
        return flds

    def get_new_field(self,fldid,N,nbts,tp,neg):
        """Adds a new field.
        
        Note: Field should not be modified as the FieldCollection would not know about it
              hence it would be corrupted.

        Args:
            fldid (FieldId): Field identity.
            N         (int): Number of points.
            nbts      (int): Number of time steps
            tp        (str): Data type: int, double, bool, ... .
            neg       (str): Data type: 'N' nodal, 'E' elemental, 'G' global.

        Returns:
            Field: The requested field.

        Raises:
            RuntimeError: If data is packed into archive or the field is already exists.
        """
        self.stop_here_if_packed()
        fld = Field(fldid,N,nbts,tp,neg)

        if fld.get_id_string() in self.fields:
            print('Field with id ({}) exists already in FieldCollection!'.format(fld.get_id_string()))
            raise RuntimeError

        fld.set_path(self.get_data_folder_path())
        self.add_field_info(fld)

        return fld

    def add_field(self,fld):
        """Adds an existing field.

        This method adds an existing field and moves its memory-map to data folder.

        Args:
            fld (Field): Existing field.

        Raises:
            RuntimeError: If data is packed into archive or the field is already exists.
        """
        self.stop_here_if_packed()
        # add new field (this checks if it exists already)
        new_fld = self.get_new_field(fld.identity,
                                     fld.N,fld.nbts,
                                     fld.data_type,fld.NEG)
        if new_fld.memmap_exists(): # check if I am overwriting a memmap
            print('MemMap with name "{}" exists already. Cannot add Field'.format(new_fld.mmap))
            self.remove_field_info(new_fld.identity) # take out added info from info file
            raise RuntimeError

        # move MemMap
        shtl.move(fld.get_memmap_path(),
                  new_fld.get_memmap_path())

    # add field info of field
    def add_field_info(self,fld):
        self.stop_here_if_packed()

        if fld.get_id_string() in self.fields:
            print('Field with id ({}) exists already in FieldCollection!'.format(fld.get_id_string()))
            raise RuntimeError

        self.fields[fld.get_id_string()] = fld
        self.write()

    # crop all fields at time step
    def crop_fields_at_time_step(self,cts):
        """Crops all fields at time step.

        Args:
            cts (int): time step at which to crop
        """
        self.stop_here_if_packed()

        for fld in self.fields.values():
            fld.crop_at_time_step(cts)
        self.write()

    # filter all fields (not in time)
    def filter_fields(self,fltr):
        """Filters all fields based on filter (not in time).

        Args:
             fltr (array): array of boolean with true being what is kept
        """
        self.stop_here_if_packed()

        for fld in self.fields.values():
            fld.filter(fltr)
        self.write()
        
    # change FieldId of a field
    def change_field_identity(self,old_fldid,new_fldid):
        self.stop_here_if_packed()
        
        if new_fldid.get_string() in self.fields:
            print('Field with id ({}) exists already in FieldCollection!'.format(new_fldid.get_string()))
            raise RuntimeError

        fld = self.get_field(old_fldid)
        fld.change_identity(new_fldid)
        self.add_field_info(fld)
        self.remove_field_info(old_fldid)
        
    # removes Field from FieldCollection
    # this destroys the MemMap
    def remove_field(self,fldid):
        """Removes a field from collection and destroys its memory-map.
        
        Args:
            fldid (FieldId): Identity of field.

        Raises:
            RuntimeError: If data is packed into archive.
        """
        self.stop_here_if_packed()
        fld = self.get_field(fldid)
        fld.destroy_memmap()
        self.remove_field_info(fldid)

    def remove_field_info(self,fldid):
        """Removes a field from collection

        Args:
            fldid (FieldId): Identity of field.
        """
        idstr = fldid.get_string()
        if idstr in self.fields:
            del self.fields[idstr]
        self.write()

    def create_info_file_name(self,name):
        """Returns info file name.

        Returns:
            str: Info file name.
        """
        return name + '.fieldcollection.info'

    def create_data_folder_name(self,name):
        """Returns data folder name.

        Returns:
            str: Data folder name.
        """
        return name + '-fieldcollection-files'

    def __repr__(self):
        """Returns a printable representation of field collection.

        Returns:
            str: String with information of field collection.
        """
        field_string = list()
        idspc = max(0,max([len(fld.get_id_string()) for fld in self.fields.values()]))
        [field_string.append("  {} {}".format(str(fld.get_id_string()).ljust(idspc),
                                                   " ".join(fld.__repr__().split()[2:]))) 
         for fld in self.fields.values()]
        return "FieldCollection '{}' with fields:\n{}".format(InfoFolderHandler.__repr__(self),
                                                              "\n".join(field_string))
