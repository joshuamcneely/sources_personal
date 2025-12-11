#!/usr/bin/env python
#
# infofolderhandler.py
#
# This handles an info file and a corresponding folder
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
import zipfile, tarfile

from .utilities import sys_test_dir, path_exists

class InfoFolderHandler(object):
    """Handles an info file and its corresponding folder.

    InfoFolderHandler uses the following file structure to handle data.

    {wdir}/{name}.info        <- info file
    {wdir}/{name}-files       <- data folder
    {wdir}/{name}.{zip_ext}   <- archive (if enabled)

    Info file consists of lines starting with {keychar} as key string
    and the respective following line as value.

    Static Attributes:
        keychar     (str): Key string identifier.
        commentchar (str): Comment string identifier.
        zip_format  (str): Format of archive.
        zip_ext     (str): File extension of archive.

    Attributes:
        name     (str): Job name.
        wdir     (str): Path to working directory.
        data_dir (str): Path to corresponding data.
        info_fn  (str): Name of info file.
        zip_fn   (str): Name of archive.
    """

    # static class variable
    keychar = '!'
    commentchar = '#'

    # zip format does not work with files >2GB
    zip_format='gztar' #'zip'
    zip_ext='.tar.gz' #'.zip'

    def __init__(self,name,wdir,create=False):
        """Creates InfoFolderHandler object.
        
        Reads or creates an info file and data folder in the working directory.

        Args:
            name   (str) : Job name.
            wdir   (str) : Path to working directory.
            create (bool): Whether or not to create new files. Defaults to False.
        """

        # data given
        self.name = name
        self.wdir = sys_test_dir(wdir)

        # data in the info file
        self.data_dir = None

        # tmp data
        self.info_fn = self.create_info_file_name(self.name)

        # name of zipped file
        self.zip_fn = None

        # get or create information
        if self.exists_already(self.name):
            self.read()
            if not self.zip_fn and not self.folder_exists_already(self.name):
                self.create_data_folder()
        elif create:
            self.create()
        else:
            raise IOError("File "+self.info_fn+" does not exist in folder: "+self.wdir
                          +" and I was not allowed to create it!")

    def change_name(self,new_name):
        """Changes job name.

        If duplicate file/directory names were found, raise RunTimeError.
        Otherwise it saves/moves files with new_name then delete old files.

        Args:
            new_name (str): New job name.
        
        Raises:
            RuntimeError: When duplicate file/directory names were found.
        """
        # check if something with the new name exists already
        # if so, do nothing
        if self.exists_already(new_name):
            print('Data with new name '+new_name+' exists already!')
            raise RuntimeError
        if self.folder_exists_already(new_name):
            print('Folder '+self.create_data_folder_name(new_name)+' exists already!')
            raise RuntimeError
        if self.is_packed() and self.zip_exists_already(new_name):
            print('Zip file '+self.zip_fn+' exists already!')
            raise RuntimeError

        # keep old names to move or delete
        old_info_fn  = self.info_fn
        old_data_dir = self.data_dir
        if self.is_packed():
            old_zip_fn   = self.zip_fn

        # create new information
        self.name = new_name
        self.info_fn  = self.create_info_file_name(self.name)
        self.data_dir = self.create_data_folder_name(self.name)
        if self.is_packed():
            self.zip_fn = self.create_zip_file_name(self.name)
        self.write()

        # delete old information
        os.remove(self.wdir+old_info_fn)

        if self.is_packed():
            # move zip file
            os.rename(self.wdir+old_zip_fn,
                      self.wdir+self.zip_fn)
        else:
            # move the folder
            os.rename(self.wdir+old_data_dir,
                      self.get_data_folder_path())

    def destroy(self):
        """Destroys object and associated files.
        
        This method removes info file and data folder, then cleans up object.
        """
        # remove files and folder
        if self.info_fn and path_exists(self.get_info_file_path()):
            os.remove(self.get_info_file_path())
        if self.data_dir and path_exists(self.get_data_folder_path()):
            shtl.rmtree(self.get_data_folder_path())
        if self.is_packed() and path_exists(self.get_zip_file_path()):
            os.remove(self.get_zip_file_path())

        # clean itself
        self.name = None
        self.info_fn = None
        self.data_dir = None
        self.zip_fn = None

    # function that can be used to stop if packed
    def stop_here_if_packed(self):
        """Stops here if data is packed into archive.
        
        This method raises RuntimeError if data is packed into archive.

        Raises:
            RuntimeError: If data is packed into archive.
        """
        if self.is_packed():
            print('This data is packed and cannot be accessed directly. You need to unpack it.')
            raise RuntimeError

    # check if is packed
    def is_packed(self):
        """Determines if data is packed into archive.

        Returns:
            bool: True when data is packed into archive.
        """
        return bool(self.zip_fn)

    # compress folder
    def pack(self):
        """Packs data into archive.

        This method packs files in data folder into archive then removes data folder.

        Raises:
            RuntimeError: When archive already exists or failed to save archive.
        """

        # check if already packed
        if self.is_packed():
            print('data is already packed. I do nothing.')
            return

        # check if there is actually something to pack
        if not self.folder_exists_already(self.name):
            print('No data folder to pack.')
            return

        # check if zip file with this name exists already
        if self.zip_exists_already(self.name):
            print('Zip file '+self.create_zip_file_name(self.name)+' exists already!')
            raise RuntimeError

        # archive it now
        full_zip_path = shtl.make_archive(self.wdir + self.data_dir,
                                          InfoFolderHandler.zip_format,
                                          self.get_data_folder_path())
        self.zip_fn = os.path.basename(full_zip_path)
        print('packed to file: {}'.format(self.zip_fn))

        # check if zip file name does actually correspond to what is expected
        # if shutil changes, this could cause problems
        if not self.zip_fn == self.create_zip_file_name(self.name):
            print('zip file naming does not work anymore. Potentially shutil has changed.')
            print('expected: {}'.format(self.create_zip_file_name(self.name)))
            raise RuntimeError

        # check if zip file exists
        if not self.zip_exists_already(self.name):
            print('writing of packed file failed. it does not exist.')
            raise RuntimeError

        self.write()

        # delete data folder
        if self.data_dir and path_exists(self.get_data_folder_path()):
            shtl.rmtree(self.get_data_folder_path())
                
    def unpack(self):
        """Unpacks archive into data directory.

        This methid unpacks archive to data folder then removes archive.

        Raises:
            RuntimeError: When missing archive or data folder already exists.
        """

        # check it was packed
        if not self.is_packed():
            print('data was not packed. do nothing.')
            return

        # check if a zip file exists
        if not self.zip_exists_already(self.name):
            print('the zip file is missing. cannot unpack it.')
            raise RuntimeError

        # check if a data folder exists already
        if self.folder_exists_already(self.name):
            print('a data folder exists already. cannot unpack and overwrite it.')
            raise RuntimeError

        # unpack it now
        self.create_data_folder()
        # in python 3, this could be replaced by shutil.unpack_archive
        tar = tarfile.open(self.get_zip_file_path())
        tar.extractall(self.get_data_folder_path())
        tar.close()
        # this does not work, because cannot pack zip files > 2GB with shutil.make_archive (at least for now)
        #zipfile.ZipFile(self.get_zip_file_path()).extractall(self.get_data_folder_path())
        # this fails with a header error. Thus replaced with the 3 lines above
        #tarfile.TarFile(self.get_zip_file_path()).extractall(self.get_data_folder_path())

        # delete zip file
        os.remove(self.get_zip_file_path())
        print('unpacked from file: {}'.format(self.zip_fn))
        self.zip_fn = None
        self.write()


    def exists_already(self,name):
        """Checks the existance of info file of requested job name.

        Args:
            name (str): Requested job name.
        
        Returns:
            bool: True if info file exists.
        """
        return path_exists(self.create_info_file_name(name),self.wdir)

    def folder_exists_already(self,name):
        """Checks the existance of data folder of requested job name.

        Args:
            name (str): Requested job name.

        Returns:
            bool: True if data folder exists.
        """
        return path_exists(self.create_data_folder_name(name),self.wdir)

    def zip_exists_already(self,name):
        """Checks the existance of archive of requested job name.

        Args:
            name (str): Requested job name.

        Returns:
            bool: True if data folder exists.
        """
        return path_exists(self.create_zip_file_name(name),self.wdir)

    def create(self):
        """Creates info file and data folder.

        This method creates info file and data folder in the working directory.
        """
        self.info_fn  = self.create_info_file_name(self.name)
        self.data_dir = self.create_data_folder_name(self.name)
        self.write()
        self.create_data_folder()

    def create_data_folder(self):
        """Creates data folder.

        This method creates data folder in the working directory.
        """
        if not os.path.exists(self.get_data_folder_path()):
            os.makedirs(self.get_data_folder_path())

    def write(self):
        """Writes info file.

        This method writes path of info file into the info file.
        """
        with open(self.get_info_file_path(), 'w') as fl:
            self.write_to_file(fl)

    # used to give the daugther class possibility of writing
    def write_to_file(self,fl):
        """Writes path of data folder (and archive) into file.

        This method is used to enable inherited classes to write data.

        Args:
            fl (file): File to be written.
        """
        print(InfoFolderHandler.keychar+'datafolder', file=fl)
        print('{0}'.format(self.data_dir), file=fl)
        
        if self.is_packed():
            print(InfoFolderHandler.keychar+'compressed-datafolder', file=fl)
            print('{0}'.format(self.zip_fn), file=fl)

    def read(self):
        """Reads info file.

        This method finds key strings in info file and reads corresponding values.
        """
        key = None
        try:
            with open(self.get_info_file_path(), 'r') as fl:
                for line in fl:
                    line = str(line.strip())
                    #line = line.split()
                    knows_key = False
                    if not line or line[0] == InfoFolderHandler.commentchar:
                        knows_key = True
                    elif line[0] == InfoFolderHandler.keychar:
                        key = line[1:]
                        knows_key = True
                    else:
                        knows_key = self.read_from_file(key,line)

                    if not knows_key:
                        print("Don't know what to do with: {0} (key = {1})".format(line,key))
        except IOError:
            print("File "+self.get_info_file_path()+" does not exist")

    # used to give daugther class possibility of writing
    # returns True if it was successful
    def read_from_file(self,key,line):
        """Stores value if the key string is valid.

        This method is used to enable inherited classes to write data.

        Args:
            key  (str): Key string.
            line (str): Value string.

        Returns:
            bool: True if successful.
        """

        knows_key = False
    
        # data about datafolder
        if key == 'datafolder':
            self.data_dir = line
            knows_key = True

        # data about compressed datafolder
        elif key == 'compressed-datafolder':
            self.zip_fn = line
            knows_key = True
        
        return knows_key

    def get_info_file_path(self):
        """Returns path of info file.

        Returns:
            str: Path of info file.
        """
        return self.wdir + self.info_fn

    def get_data_folder_path(self):
        """Returns path of data folder.

        Returns:
            str: Path of data folder.
        """
        return sys_test_dir(self.wdir + self.data_dir)

    def get_zip_file_path(self):
        """Returns path of archive.

        Returns:
            str: Path of archive.
        """
        return self.wdir + self.zip_fn

    def create_info_file_name(self,name):
        """Returns name of info file (including file extension).

        Returns:
            str: Name of info file (including file extension).
        """
        return name + '.info'

    def create_data_folder_name(self,name):
        """Returns name of data folder.

        Returns:
            str: Name of data folder.
        """
        return name + '-files'

    # according to the shutil.make_archive function
    def create_zip_file_name(self,name):
        """Returns name of archive.

        Returns:
            str: Name of archive.
        """
        return self.create_data_folder_name(name) + InfoFolderHandler.zip_ext

    def __repr__(self):
        """Returns a string containing a printable representation (job name) of an object.

        Retruns:
            str: Job name.
        """
        return str(self.name)
