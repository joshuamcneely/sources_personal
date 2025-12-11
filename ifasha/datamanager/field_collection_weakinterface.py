#!/usr/bin/env python
#
# fieldcollectionweakinterface.py
#
# Code to read files dumped by the weak-interface library
#
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @date     2016/04/09
# @modified 2020/12/22

from __future__ import print_function, division, absolute_import

from time import time
import numpy as np
import copy

from .utilities import sys_test_dir

from .fieldid import FieldId
from .field import Field
from .field_collection import FieldCollection

def convert_to_type(dtype,string):
    """Converts number in string to specific format.

    Args:
        dtype  (str): Target format, 'float', 'int', 'uint', or 'bool'.
        string (str): Number in string.

    Raises:
        TypeError: When target format is not supported.
    """
    dtype.lower()
    if dtype.startswith('float'):
        dt = float(string)
    elif dtype.startswith('int') or dtype.startswith('uint'):
        dt = int(string)
    elif dtype == 'bool':
        dt = bool(int(string))
    else:
        raise TypeError("Invalid dtype: {}".format(dtype))
    return dt

# ----------------------------------------------------------------------------------------------

class FieldCollectionWeakInterface(FieldCollection):
    """WeakInterface extension of FieldCollection.

    FieldCollectionWeakInterface inherits FieldCollection

    Attributes:
        source_base_name    (str): Base name of source.
        source_dir          (str): Path to source directory.
        time_fn             (str): Time description.
        coord_fn            (str): Coord description.
        fields_fn           (str): Fields description.
        fields_folder       (str): Path to the folder containing files of fields.
        separator           (str): String separator, ' '.
        extension           (str): Simulation output file extension, '.out'.
        infosep             (str): Info string separator, ' '.
        comment             (str): Comment string identifier ,"#".
        nodal_fi     (list(dict)): Collection of nodal field information.
        nbts                (int): Number of time steps
        N                   (int): Number of points.
    """

    # used for "casting" of FieldCollection
    def __new__(cls, other):
        """Casts FieldCollection to FieldCollectionWeakInterface.

        Args:
            other (FieldCollection): The object to copy from.
        
        Returns:
            FieldCollectionWeakInterface: Casted instance.
        """
        if isinstance(other, FieldCollection):
            other = copy.copy(other)
            other.__class__ = FieldCollectionWeakInterface
            return other
        return object.__new__(cls)

    def __init__(self, other):
        """Creates FieldCollectionWeakInterface object.

        This method is called after __new__(cls, other) and initiates 
        a FieldCollectionWeakInterface object.

        Args:
            other (FieldCollection): The object to copy from.
        """
        self.source_base_name = None
        self.source_dir       = None

        self.time_fn = None
        self.coord_fn = None
        self.fields_fn = None
        self.fields_folder = None

        self.separator = ' '
        self.extension = '.out'

        self.infosep = ' '
        self.comment = "#"

        # field informations
        self.nodal_fi     = None

        # temporary simulation information
        self.nbts = 1 # assume one time step if no time-file
        self.N    = 0
        self.binary = False
       
    def read_simulation_output(self,fname,sdir):
        """Reads simulation output file.

        Args:
            fname (str): Simulation output file name.
            sdir  (str): Path to source directory. 
        """
        print("START: Reading Simulation Output")
        start = time() # timing of post processing

        self.source_dir = sys_test_dir(sdir)

        print("     * Loading Information File")
        self.read_info_file(self.source_dir + fname)
        print("       * Binary: {0}".format(self.binary))

        print("     * Loading Time Data")
        self.read_time_file()

        print("     * Loading Field Data")
        self.read_field_file()

        self.read_nb_nodes()
        print("       * Nb Nodes: {0}".format(self.N))

        # coords
        print("       * Loading Coord Data")
        self.read_coord_file()
        

        # nodal fields
        print("       * Nb Nodal Fields: {0}".format(len(self.nodal_fi)))
        self.load_nodal_fields()

        print("DONE! This took {}".format(time() - start))
        

    # ------------------------------------
    def read_info_file(self,fpath):
        """Reads info file information from simulation output file.

        Args:
            fpath (str): Full path to simulation output file.
        """
        try:
            with open(fpath) as fl:
                for line in fl:
                    line = line.strip()
                    if line.startswith(self.comment) or line == "":
                        continue
                    line = line.split(self.infosep)
                    
                    if str(line[0]) == "field_description":
                        self.fields_fn = str(line[1])
                    elif str(line[0]) == "time_description":
                        self.time_fn = str(line[1])
                    elif str(line[0]) == "coord_description":
                        self.coord_fn = str(line[1])
                    elif str(line[0]) == "folder_name":
                        self.fields_folder = sys_test_dir(str(line[1]))
                    elif str(line[0]) == "output_format":
                        self.binary = str(line[1]) == "binary"
                    else:
                        print("In the info file: do not understand key word: {0}".format(line[0]))
        except IOError:
            print("File "+fpath+" does not exist.")


    # ------------------------------------
    def read_time_file(self):
        """Reads time information from simulation output file.
        """
        if self.time_fn:
            self.nbts = self.count_nb_lines(self.source_dir + self.time_fn)

        step = self.get_new_field(FieldId('step'),1,self.nbts,'int32','G')
        time = self.get_new_field(FieldId('time'),1,self.nbts,'float64','G')

        step_map = step.get_memmap('w+')
        time_map = time.get_memmap('w+')

        if self.time_fn:
            try:
                with open(self.source_dir + self.time_fn) as fl:
                    count=0
                    for line in fl:
                        line = line.strip()
                        if line.startswith(self.comment) or line == "":
                            continue
                        line = line.split(self.separator)
                        stp = line[0].lstrip('0')
                        if stp == '':
                            stp = 0
                        stp = np.int32(stp)
                        tm = float(line[1])
                        step_map[count] = stp
                        time_map[count] = tm
                        count += 1
            except IOError:
                print("File "+self.source_dir+self.time_fn+" does not exist.")
        else:
            step_map[0] = 0
            time_map[0] = 0

    # ------------------------------------
    def read_coord_file(self):
        """Reads coord file.
        """
        if self.coord_fn:
            nb_coord_dim = self.count_nb_entries_in_first_line(self.source_dir + self.coord_fn)

            # load MemMaps for different components
            mmaps = list()
            for i in range(nb_coord_dim):
                field = self.get_new_field(FieldId('coord',i),self.N,1,
                                           'float64', 'N')
                mmaps.append(field.get_memmap('w+'))

            try:
                with open(self.source_dir + self.coord_fn) as fl:
                    ncount=0
                    for line in fl:
                        line = line.strip()
                        if line.startswith(self.comment) or line == "":
                            continue
                        line = line.split(self.separator)
                        
                        for mm, l in zip(mmaps,line):
                            vl = convert_to_type('float64',l)
                            mm[0,ncount] = vl
                        ncount += 1
            except IOError:
                print("File "+self.source_dir+self.coord_fn+" does not exist.")


    # ------------------------------------
    def read_field_file(self):
        """Reads field file.
        """
        field_infos = list()

        try:
            with open(self.source_dir + self.fields_fn) as fl:
                for line in fl:
                    line = line.strip()
                    if line.startswith(self.comment) or line == "":
                        continue
                    line = line.split()

                    fi = dict()
                    fi['name'] = str(line[0])
                    fi['fname'] = str(line[1])

                    field_infos.append(fi)

        except IOError:
            print("File "+self.source_dir+self.fields_fn+" does not exist.")

        self.nodal_fi     = [fi for fi in field_infos] #[fi for fi in field_infos if fi['neg']=='N']

    # ------------------------------------------------------------------------------------------
    def load_nodal_fields(self):
        """Loads nodal fields.
        """

        if len(self.nodal_fi) == 0:
            return

        self.read_node_single_file_serial(self.nodal_fi)

    def read_node_single_file_serial(self,field_infos):
        """Reads nodal file in serial.

        Args:
            field_infos list(dict()): Information of fields.
        """

        for fi in field_infos:
            print("       * Field: {0}".format(fi['name']))

            # load MemMaps for different components
            con = '_'
            name_parts = fi['name'].split(con)
            finame = con.join(name_parts[:-1])
            
            try:
                comp = int(name_parts[-1]) 
                fid = FieldId(finame,comp)
            except ValueError:
                fid = FieldId(fi['name'])

            fname = self.file_path(fi)
            
            # Auto-detect actual data size per timestep BEFORE creating field
            # (handles cases where N from coords doesn't match data, e.g., free_surface=True)
            if self.binary:
                with open(fname, 'rb') as fl:
                    data_peek = np.fromfile(fl, dtype=np.float32)
                    total_values = len(data_peek)
                    
                if total_values % self.nbts == 0:
                    # Use time file to determine values per timestep
                    values_per_step = total_values // self.nbts
                    n_time_steps = self.nbts
                else:
                    # Fallback to original calculation
                    n_time_steps = int(total_values / self.N)
                    values_per_step = self.N
                
                if values_per_step != self.N:
                    print(f'       * Detected field size: {values_per_step} (differs from coord file: {self.N})')
            else:
                values_per_step = self.N

            # Create field with correct size
            field = self.get_new_field(fid,
                                       values_per_step, self.nbts,
                                       'float64', 'N')
            mmap = field.get_memmap('w+')

            nb_steps = self.nbts
            nbs = len(str(nb_steps))
            tcount = 0
            try:
                with open(fname) as fl:
                    if self.binary:
                        data = np.fromfile(fl, dtype=np.float32)
                        
                        try:
                            data = np.reshape(data, [n_time_steps, values_per_step])
                        except:
                            data=data[:n_time_steps * values_per_step]
                            data = np.reshape(data, [n_time_steps, values_per_step])
                        try:
                            mmap[:,:] = data[:,:]
                        
                        except: # incomplete file 
                            for t in range(n_time_steps):
                                try:
                                    mmap[t,:] = data[t,:]
                                except:
                                    print('Warning: incomplete file - {}, timestep {}/{}'.format(fname, t, n_time_steps))
                    else:
                        for line in fl:
                            line = line.strip()
                            if tcount >= nb_steps: # stop here (important for ongoing simulation)
                                break
                            if line.startswith(self.comment): # just a comment
                                continue
                            if line == "": # empty line
                                continue
                            line = line.split(self.separator)
                            # loop over fields to add data from this line
                            # Only read up to self.N values (number of nodes allocated)
                            for ncount in range(min(len(line), mmap.shape[1])):
                                mmap[tcount,ncount] = convert_to_type('float64',line[ncount])
                            tcount += 1
                            print("         "
                                  +"* Step {1:{0}d}/{2:{0}d}".format(nbs,tcount,nb_steps),
                                  end="\r")
            except IOError:
                    print("File "+fname+" does not exist.")



    # ------------------------------------------------------------------------------------------

    def read_nb_nodes(self):
        """Reads number of nodes.
        """

        self.N = self.read_nb_nodes_serial()

    def read_nb_nodes_serial(self):
        """Reads number of nodes in serial.

        Args:
            step          (int): Step index.

        Returns:
            int: Number of nodes.
        """
        return self.count_nb_lines(self.source_dir + self.coord_fn)


    # ------------------------------------------------------------------------------------------

    # counts number of lines that are not empty and not commented
    def count_nb_lines(self,fpath):
        """Counts number of lines that are not empty and not commented.

        Args:
            fpath (str): Full path to file.

        Returns:
            int: Number of lines that are not empty and not commented.
        """
        count = 0
        try:
            with open(fpath) as fl:
                for line in fl:
                    line = line.strip()
                    if line.startswith(self.comment) or line == "":
                        continue
                    count += 1
        except IOError:
            print("File "+fpath+" does not exist.")
        return count

    # returns file path (path + name of file) for source files
    def file_path(self,field_info):
        """Returns full path to source files.

        Args:
            field_info (dict()): Information of field.
            step          (int): Step index. Defaults to None.
            proc          (int): Process index. Defaults to None.

        """
        fp  = self.source_dir
        fp += self.fields_folder
        fp += field_info['fname']
        return fp

    # counts lines until first empty line
    def count_nb_entries_in_first_line(self,fpath):
        """Counts number of entries in the first line.

        Args:
            fpath (str): Full path to file.

        Returns:
            int: Number of entries in the first line.
        """
        count = 0
        try:
            with open(fpath) as fl:
                for line in fl:
                    line = line.strip()
                    if line.startswith(self.comment):
                        continue
                    elif line == "":
                        continue
                    else:
                        line = line.split()
                        count = len(line)
                        break
        except IOError:
            print("File "+fpath+" does not exist.")
        return count



if __name__ == "__main__":
    """Does nothing if the python interpreter is running this module as the main program.
    """
    print('Do nothing')
