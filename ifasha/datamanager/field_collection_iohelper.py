#!/usr/bin/env python
#
# fieldcollectioniohelper.py
#
# Code to read files dumped by the IOHelper TextDumper
#
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @date     2013/05/16
# @modified 2020/12/22

from __future__ import print_function, division, absolute_import

from time import time
import numpy as np
import itertools
import os.path
import sys
import copy

from .utilities import sys_test_dir

from .fieldid import FieldId
from .field import Field
from .field_collection import FieldCollection


def get_mode_info(mode):
    """Returns the corresponding separator of mode.

    Returns:
        str: Separator.
    """
    mode.lower()
    if mode == "csv":
        return (',', '.csv')
    elif mode == "space":
        return (' ', '.out')
    else:
        return None


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
    elif dtype == 'ntype':
        dt = int(string)
        if dt == -1 or dt == -2:
            dt = bool(1)
        else:
            dt = bool(0)
    else:
        raise TypeError("Invalid dtype: {}".format(dtype))
    return dt


# ----------------------------------------------------------------------------------------------

class FieldCollectionIOHelper(FieldCollection):
    """IO helper extension of FieldCollection.

    FieldCollectionIOHelper inherits FieldCollection

    Attributes:
        source_base_name    (str): Base name of source.
        source_dir          (str): Path to source directory.
        time_fn             (str): Time description.
        fields_fn           (str): Fields description.
        nbproc              (int): Number of processes.
        procs              (list): Collection of processes.
        file_p_ts          (bool): Whether in parallel.
        stepwidth           (int): Width of steps.
        procwidth           (int): Width of processes.
        version             (str): Version.
        separator           (str): String separator, ' '.
        extension           (str): Simulation output file extension, '.out'.
        infosep             (str): Info string separator, ' '.
        comment             (str): Comment string identifier ,"#".
        nodal_fi     (list(dict)): Collection of nodal field information.
        elemental_fi (list(dict)): Collection of elemental field information.
        global_fi    (list(dict)): Collection of global field information.
        ntype         (list(list): Nodal value in defined type.
        nbts                (int): Number of time steps
        N                   (int): Number of points.
        nodetypeid          (str): Identifier for determining number of nodes in serial or parallel, 'nodes_type'.
        positionid          (str): Identifier for determining number of nodes in serial or parallel, 'position'.
    """

    # used for "casting" of FieldCollection
    def __new__(cls, other):
        """Casts FieldCollection to FieldCollectionIOHelper.

        Args:
            other (FieldCollection): The object to copy from.
        
        Returns:
            FieldCollectionIOHelper: Casted instance.
        """
        if isinstance(other, FieldCollection):
            other = copy.copy(other)
            other.__class__ = FieldCollectionIOHelper
            return other
        return object.__new__(cls)

    def __init__(self, other):
        """Creates FieldCollectionIOHelper object.

        This method is called after __new__(cls, other) and initiates 
        a FieldCollectionIOHelper object.

        Args:
            other (FieldCollection): The object to copy from.
        """
      
        self.source_base_name = None
        self.source_dir       = None

        self.time_fn = None
        self.fields_fn = None

        self.nbproc   = 1
        self.procs = list()
        self.file_p_ts = False

        self.stepwidth = 4
        self.procwidth = 3

        self.version = None
        self.separator = ' '
        self.extension = '.out'

        self.infosep = ' '
        self.comment = "#"

        # field informations
        self.nodal_fi     = None
        self.elemental_fi = None
        self.global_fi    = None

        # nodes that should be read
        self.ntype = None

        # temporary simulation information
        self.nbts = 1 # assume one time step if no time-file
        self.N    = 0

        # identifier for determining number of nodes in serial or parallel
        self.nodetypeid = 'nodes_type'
        self.positionid = 'position'

       
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

        print("     * Loading Time Data")
        self.read_time_file()

        print("     * Loading Field Data")
        self.read_field_file()
        self.check_for_needed_fields()

        self.read_proc_list()
        print("       * Proc List: {0}".format(self.procs))

        self.read_nb_nodes()
        print("       * Nb Nodes: {0}".format(self.N))

        # nodal fields
        print("       * Nb Nodal Fields: {0}".format(len(self.nodal_fi)))
        self.load_nodal_fields()

        # elemental fields
        print("       * Nb Elemental Fields: {0}".format(len(self.elemental_fi)))
        if len(self.elemental_fi) > 0:
            print('WARNING: loadElementalFields not yet implemented')

        # global fields
        print("       * Nb Global Fields: {0}".format(len(self.global_fi)))
        self.load_global_fields()

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
                    if str(line[0]) == "0-1":
                        self.version    = str(line[0])
                        self.stepwidth  = int(line[1])
                        self.nbproc     = int(line[2])
                        self.procwidth  = int(line[3])
                        self.file_p_ts  = convert_to_type("bool",line[4])
                    elif str(line[0]) == "base_name":
                        self.source_base_name = str(line[1])
                    elif str(line[0]) == "field_description":
                        self.fields_fn = str(line[1])
                    elif str(line[0]) == "time_description":
                        self.time_fn = str(line[1])
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
        time = self.get_new_field(FieldId('time'),1,self.nbts,'float32','G')

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
                print("File "+fpath+" does not exist.")
        else:
            step_map[0] = 0
            time_map[0] = 0


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

                    # [0]-field_name [1]-nodal_element_global_data [2]-data_type
                    # [3]-dump_mode [4]-nb_components [5]-rel_file_path
                    fi = dict()
                    fi['name'] = str(line[0])
                    fi['neg']  = str(line[1])
                    fi['dtype'] = str(line[2])
                    fi['dmode'] = str(line[3])
                    fi['nbc'] = int(str(line[4]))
                    fi['path'] = str(line[5])

                    field_infos.append(fi)

        except IOError:
            print("File "+fpath+" does not exist.")

        self.nodal_fi     = [fi for fi in field_infos if fi['neg']=='N']
        self.elemental_fi = [fi for fi in field_infos if fi['neg']=='E']
        self.global_fi    = [fi for fi in field_infos if fi['neg']=='G']

            
    # ------------------------------------
    def load_global_fields(self):
        """Loads global fields.

        Raises:
            IOError: When simulation output file does not exist.
        """

        for fi in self.global_fi:

            print("       * Field: {0}".format(fi['name']))

            # load MemMaps for different components
            mmaps = list()
            for i in range(fi['nbc']):
                field = self.get_new_field(FieldId(fi['name'],i),1,self.nbts,
                                           fi['dtype'], fi['neg'])
                mmaps.append(field.get_memmap('w+'))
                   
            fname = self.file_path(fi)
            sep = get_mode_info(fi['dmode'])[0]

            nb_lines = self.nbts
            nbl = len(str(nb_lines))
            try:
                with open(fname) as fl:
                    lcount = 0
                    for line in fl:
                        # In case of ongoing simulation
                        if lcount >= nb_lines:
                            break
                        line = line.strip()
                        if line.startswith(self.comment) or line == "":
                            continue
                        line = line.split(sep)
                        print("         * Line {1:{0}d}/{2:{0}d}".format(nbl,
                                                                         lcount+1,
                                                                         nb_lines), end="\r")
                        sys.stdout.flush()
                        for mm, l in zip(mmaps,line):
                            vl = convert_to_type(fi['dtype'],l)
                            mm[lcount] = vl
                        lcount += 1
                print("")
            except IOError:
                print("File "+fname+" does not exist.")


    # ------------------------------------------------------------------------------------------

    def load_nodal_fields(self):
        """Loads nodal fields.
        """

        if len(self.nodal_fi) == 0:
            return

        if self.nbproc > 1:
            self.read_node_type()
            if self.file_p_ts:
                # without loading the node type field
                self.read_node_file_parallel([fi for fi in self.nodal_fi if not fi['name'] == self.nodetypeid])
            else:
                # without loading the node type field
                self.read_node_single_file_parallel([fi for fi in self.nodal_fi if not fi['name'] == self.nodetypeid])
        else:
            if self.file_p_ts:
                self.read_node_file_serial(self.nodal_fi)
            else:
                self.read_node_single_file_serial(self.nodal_fi)


    def read_node_file_serial(self,field_infos):
        """Reads nodal file in serial.

        Args:
            field_infos list(dict()): Information of fields.
        """

        for fi in field_infos:

            print("       * Field: {0}".format(fi['name']))

            # load MemMaps for different components
            mmaps = list()
            for i in range(fi['nbc']):
                field = self.get_new_field(FieldId(fi['name'],i),self.N,self.nbts,
                                           fi['dtype'], fi['neg'])
                mmaps.append(field.get_memmap('w+'))

            ts = self.get_field_memmap(FieldId('step'))[:,0].tolist()

            sep = get_mode_info(fi['dmode'])[0]
            nbs = len(str(self.nbts))

            # loop over all time steps
            for t, tcount in zip(ts,range(len(ts))):
                fname = self.file_path(fi,t)
                # for each time step open file and go through it line by line
                try:
                    with open(fname) as fl:
                        ncount = 0
                        for line in fl:
                            line = line.strip()
                            if line.startswith(self.comment) or line == "":
                                continue
                            line = line.split(sep)

                            # loop over fields to add data from this line
                            for mm, l in zip(mmaps,line):
                                vl = convert_to_type(fi['dtype'],l)
                                mm[tcount,ncount] = vl
                            ncount += 1
                except IOError:
                    print("File "+fname+" does not exist.")
                    print("         "
                          +"* Step {1:{0}d}/{2:{0}d}".format(nbs,tcount+1,self.nbts),
                          end="\r")
            print("")



    def read_node_single_file_serial(self,field_infos):
        """Reads single nodal file in serial.

        Args:
            field_infos list(dict()): Information of fields.
        """

        for fi in field_infos:
                    
            print("       * Field: {0}".format(fi['name']))

            # load MemMaps for different components
            mmaps = list()
            for i in range(fi['nbc']):
                field = self.get_new_field(FieldId(fi['name'],i),self.N,self.nbts,
                                           fi['dtype'], fi['neg'])
                mmaps.append(field.get_memmap('w+'))
            
            sep = get_mode_info(fi['dmode'])[0]

            fname = self.file_path(fi)

            nb_steps = self.nbts
            nbs = len(str(nb_steps))
            tcount = 0
            ncount = 0
            
            try:
                with open(fname) as fl:
                    for line in fl:
                        line = line.strip()
                        if tcount >= nb_steps: # stop here (important for ongoing simulation)
                            break
                        if line.startswith(self.comment): # just a comment
                            continue
                        if line == "": # new block = new time step
                            tcount += 1
                            ncount = 0
                            print("         "
                                  +"* Step {1:{0}d}/{2:{0}d}".format(nbs,tcount,nb_steps),
                                  end="\r")
                            continue

                        line = line.split(sep)
                        # loop over fields to add data from this line
                        for mm, l in zip(mmaps,line):
                            vl = convert_to_type(fi['dtype'],l)
                            mm[tcount,ncount] = vl
                        ncount += 1

            except IOError:
                print("File "+fname+" does not exist.")
            print("")





    def read_node_file_parallel(self,field_infos):
        """Reads nodal file in parallel.

        Args:
            field_infos list(dict()): Information of fields.
        """

        for fi in field_infos:

            print("       * Field: {0}".format(fi['name']))

            # load MemMaps for different components
            mmaps = list()
            for i in range(fi['nbc']):
                field = self.get_new_field(FieldId(fi['name'],i),self.N,self.nbts,
                                           fi['dtype'], fi['neg'])
                mmaps.append(field.get_memmap('w+'))

            ts = self.get_field_memmap(FieldId('step'))[:,0].tolist()

            sep = get_mode_info(fi['dmode'])[0]
            nbs = len(str(self.nbts))

            # loop over all time steps
            for t, tcount in zip(ts,range(len(ts))):
                ncount = 0
                pcount = 0
                for p in self.procs:
                    fname = self.file_path(fi,t,p)
                    # for each time step open file and go through it line by line
                    try:
                        with open(fname) as fl:
                            lcount =  0
                            for line in fl:
                                line = line.strip()
                                if line.startswith(self.comment) or line == "":
                                    continue
                                if not ntype[pcount][lcount]:
                                    lcount += 1
                                    continue

                                line = line.split(sep)
                                # loop over fields to add data from this line
                                for mm, l in zip(mmaps,line):
                                    vl = convert_to_type(fi['dtype'],l)
                                    mm[tcount,ncount] = vl
                                lcount += 1
                                ncount += 1

                    except IOError:
                        print("File "+fname+" does not exist.")
                    pcount += 1
                print("         "
                      +"* Step {1:{0}d}/{2:{0}d}".format(nbs,tcount+1,self.nbts), 
                      end="\r")
            print("")


    def read_node_single_file_parallel(self, field_infos):
        """Reads single nodal file in parallel.

        Args:
            field_infos list(dict()): Information of fields.
        """

        for fi in field_infos:

            print("       * Field: {0}".format(fi['name']))

            # load MemMaps for different components
            mmaps = list()
            for i in range(fi['nbc']):
                field = self.get_new_field(FieldId(fi['name'],i),self.N,self.nbts,
                                           fi['dtype'], fi['neg'])
                mmaps.append(field.get_memmap('w+'))

            
            sep = get_mode_info(fi['dmode'])[0]
 
            nb_steps = self.nbts
            nbs = len(str(nb_steps))
            nb_files = len(self.procs)
            nbf =len(str(nb_files))

            ncount_previous_proc = 0
            ncount_current_proc = 0
            pcount = 0
            for p in self.procs:
                fname = self.file_path(fi,None,p)
                tcount = 0
                lcount = 0
                ncount = ncount_previous_proc

                try:
                    with open(fname) as fl:
                        for line in fl:
                            line = line.strip()
                            if tcount >= nb_steps: # stop here (important for ongoing simulation)
                                break
                            if line.startswith(self.comment): # just a comment
                                continue
                            if line == "": # new block = new time step
                                tcount += 1
                                lcount = 0
                                ncount = ncount_previous_proc
                                print("         "
                                      +"* Proc {1:{0}d}/{2:{0}d}".format(nbf,pcount+1,nb_files)
                                      +" Step {1:{0}d}/{2:{0}d}".format(nbs,tcount,nb_steps),
                                      end="\r")
                                continue

                            if not self.ntype[pcount][lcount]:
                                lcount += 1
                                continue

                            line = line.split(sep)
                            # loop over fields to add data from this line
                            for mm, l in zip(mmaps,line):
                                vl = convert_to_type(fi['dtype'],l)
                                mm[tcount,ncount] = vl
                            lcount += 1
                            ncount += 1
                            ncount_current_proc = ncount

                except IOError:
                    print("File "+fname+" does not exist.")
                pcount += 1
                ncount_previous_proc = ncount_current_proc
            print("")


    def read_node_type(self):
        """Reads nodal value in defined type.
        """
        
        nt_fi = list(filter(lambda fi: fi['name'] == self.nodetypeid, self.nodal_fi))[0]
        sep = get_mode_info(nt_fi['dmode'])[0]

        # first step (in any case)
        step = self.get_field_memmap(FieldId("step"))[0] if self.file_p_ts else None

        self.ntype = list()
        for p in self.procs:
            fname = self.file_path(nt_fi,step,p)
            ntypep = list()
            try:
                with open(fname) as fl:
                    for line in fl:
                        line = line.strip()
                        if line.startswith(self.comment):
                            continue
                        if line == "":
                            break
                        line = line.split(sep)
                        ntypep.append(convert_to_type('ntype',line[0]))
            except IOError:
                print("File "+fname+" does not exist.")
            self.ntype.append(ntypep)

    # ------------------------------------------------------------------------------------------

    def check_for_needed_fields(self):
        """Checks whether needed fields are provided.

        Raises:
            RuntimeError: When needed fields are not provided.
        """

        # check nodal fields
        if len(self.nodal_fi) == 0:
            return

        if self.nbproc > 1 and len(list(filter(lambda fi: fi['name'] == self.nodetypeid, self.nodal_fi))) == 0:
            print('This is parallel simulation output. Need Field: '+self.nodetypeid)
            raise RuntimeError

        if self.nbproc == 1 and len(list(filter(lambda fi: fi['name'] == self.positionid, self.nodal_fi))) == 0:
            print('This is serial simulation output. Need Field: '+self.positionid)
            raise RuntimeError
            
        # check elemental fields

        # check global fields

    # ------------------------------------------------------------------------------------------

    def read_nb_nodes(self):
        """Reads number of nodes.
        """

        if len(self.nodal_fi) == 0:
            return
        
        # first step (in any case)
        step = self.get_field_memmap(FieldId("step"))[0] if self.file_p_ts else None

        nb=0
        if self.nbproc > 1:
            nt_fi = list(filter(lambda fi: fi['name'] == self.nodetypeid, self.nodal_fi))[0]
            nb = self.read_nb_nodes_parallel(step,nt_fi)
        else:
            po_fi = list(filter(lambda fi: fi['name'] == self.positionid, self.nodal_fi))[0]
            nb = self.read_nb_nodes_serial(step,po_fi)

        self.N = nb

    def read_nb_nodes_serial(self,step,field_info):
        """Reads number of nodes in serial.

        Args:
            step          (int): Step index.
            field_info (dict()): Information of field.

        Returns:
            int: Number of nodes.
        """
        return self.count_nb_lines_in_first_block(self.file_path(field_info,step))

    def read_nb_nodes_parallel(self,step,field_info):
        """Reads number of nodes in parallel.

        Args:
            step          (int): Step index.
            field_info (dict()): Information of field.

        Returns:
            int: Number of nodes.
        """
        sep = get_mode_info(field_info['dmode'])[0]

        count = 0
        for i in self.procs:
            fname = self.file_path(field_info,step,i)
            try:
                with open(fname) as fl:
                    for line in fl:
                        line = line.strip()
                        if line.startswith(self.comment):
                            continue
                        if not self.file_p_ts and line == "":
                            break
                        line = line.split(sep)
                        if (convert_to_type('ntype',line[0])):
                            count += 1
            except IOError:
                print("File "+fname+" does not exist.")
        return count


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
    def file_path(self,field_info,step=None,proc=None):
        """Returns full path to source files.

        Args:
            field_info (dict()): Information of field.
            step          (int): Step index. Defaults to None.
            proc          (int): Process index. Defaults to None.

        """
        fp  = self.source_dir
        fp += sys_test_dir(field_info['path'])
        fp += self.source_base_name
        fp += "_" + field_info['name']
        if not step == None:
            fp += "_" + str(step).zfill(self.stepwidth)
        if not proc == None:
            fp += "".join([".proc",str(proc).zfill(self.procwidth)])
        fp += get_mode_info(field_info['dmode'])[1]
        return fp

    # gets the procs that contain information
    def read_proc_list(self):
        """Reads the procs that contain information
        """

        if len(self.nodal_fi) == 0 and len(self.elemental_fi) == 0:
            return
        
        # first step (in any case)
        step = self.get_field_memmap(FieldId("step"))[0] if self.file_p_ts else None

        if self.nbproc > 1:
            fi = list(filter(lambda fi: fi['name'] == self.nodetypeid, self.nodal_fi))[0]
            for i in range(self.nbproc):
                fname = self.file_path(fi,step,i)
                if os.path.isfile(fname) and self.count_nb_lines_in_first_block(fname) > 0:
                    self.procs.append(i)

    # counts lines until first empty line
    def count_nb_lines_in_first_block(self,fpath):
        """Counts lines until first empty line.

        Args:
            fpath (str): Full path to file.
        Returns:
            int: Lines until first empty line.
        """
        count = 0
        try:
            with open(fpath) as fl:
                for line in fl:
                    line = line.strip()
                    if line.startswith(self.comment):
                        continue
                    elif line == "":
                        break
                    count += 1
        except IOError:
            print("File "+fpath+" does not exist.")
        return count


if __name__ == "__main__":
    """Does nothing if the python interpreter is running this module as the main program.
    """
    print('Do nothing')
