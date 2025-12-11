#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import itertools
import os.path

# SimulationData.py
#
# Code to read files dumped by the IOHelper TextDumper
# There is no warranty for this code
#
# @version 0.2
# @author David Kammer <davekammer@gmail.com>
# @date     2013/05/16
# @modified 2013/09/13

sysv_separator = '/'

def sys_testDir(dr):
    if not dr.endswith(sysv_separator) and not len(dr) == 0:
        dr += sysv_separator
    return dr

class BasicSimInfo:
    simdataversion='0.2'
    
    def __init__(self):
        self.version = None

        self.base_name = None # basename of data
        self.N = None # number of nodes
        self.nbts = None # number of time steps

        self.info_fn   = None
        self.fields_fn = None # file with information about fields
        self.mmap_dir  = None # subdirectory for memmaps

    def setBaseName(self,name):
        self.base_name = name
        self.makeFileNames()
        self.makeMapDir()

    def makeFileNames(self):
        self.info_fn   = self.base_name + ".mmap" + ".info"
        self.fields_fn = self.base_name + ".mmap" + ".fields"

    def makeMapDir(self):
        self.mmap_dir = self.base_name + "-" + "MemMaps" + sysv_separator

    def checkExistenceOfInfoFile(self,path,fname=None):
        path = sys_testDir(path)
        if fname == None:
            fname = self.info_fn
        return os.path.exists(path+fname)

    def write(self,folder):
        self.makeFileNames()
        with open(folder+self.info_fn, 'w') as fl:
            print("{0}".format(self.version), file=fl)
            print("{0} {1} {2}".format(self.base_name,
                                       self.N,
                                       self.nbts), file=fl)
            print("{0}".format(self.fields_fn), file=fl)
            print("{0}".format(self.mmap_dir), file=fl)

    def read(self,fname,folder):
        try:
            with open(folder+fname, 'r') as fl:
                for line in fl:
                    line = line.strip()
                    line = line.split()
                    for l in line:
                        if self.version == None:
                            self.version = str(l)
                        elif self.base_name == None:
                            self.base_name = str(l)
                        elif self.N == None:
                            self.N = int(l)
                        elif self.nbts == None:
                            self.nbts = int(l)
                        elif self.fields_fn == None:
                            self.fields_fn = str(l)
                        elif self.mmap_dir == None:
                            self.mmap_dir = str(l)
                        else:
                            print("too much information in info file")

        except IOError:
            print("File "+fname+" does not exist in folder: "+folder)

        self.info_fn = fname

    def __repr__(self):
        return "BasicSimInfo {0}".format(self.base_name)


class FieldId():
    def __init__(self,nm,dr=0):
        self.name = nm
        self.dir  = dr

    def getString(self):
        return str(self.name) + "_" + str(self.dir)

    def __repr__(self):
        return self.getString()

class Field:
    def __init__(self,N=None,dr=None,nbts=None,tp=None,nm=None,neg=None):
        self.id = FieldId(nm,dr)

        self.N     = N # number of nodes
        self.nbts  = nbts # number time steps
        self.type  = tp # data type: int, double, bool, ...
        self.NEG = neg # data type: N nodal, E elemental, G global

        self.file  = None # file_name with relativ path from working directory
        self.path  = None # file_name with complete path including working directory

        self.sep    = " "

    def initValues(self,N,dr,nbts,tp,nm,neg):
        self.id = FieldId(nm,dr)
        self.N     = N # number of nodes
        self.nbts  = nbts # number time steps
        self.type  = tp # data type: int, double, bool
        self.NEG   = neg # data type: N nodal, E elemental, G global

    def readString(self,sname):
        line = sname
        line = line.strip()
        line = line.split(self.sep)
        self.id.name = str(line[0])
        self.N  = int(line[1])
        self.nbts = int(line[2])
        self.id.dir = int(line[3])
        self.type = str(line[4])
        self.NEG  = str(line[5])
        self.file = str(line[6])

    def makeString(self):
        return self.sep.join([self.id.name,
                              str(self.N),
                              str(self.nbts),
                              str(self.id.dir),
                              self.type,
                              str(self.NEG),
                              self.file])

    def getMyIdString(self):
        return self.id.getString()

    def createFileName(self,base_name,mmap_dir):
        self.file = mmap_dir + base_name + "_" + self.getMyIdString() + ".mmp"

    def findMemMap(self,folder):
        self.path = folder + self.file

    def getMemMap(self,mode='r'):
        return np.memmap(self.path,
                         dtype=self.type,
                         mode=mode,
                         shape=(self.nbts, self.N))

    def __repr__(self):
        return "Field: id={0} N={1} nbts={2} type={3} neg={4} file={5}".format(self.id,
                                                                               self.N,
                                                                               self.nbts,
                                                                               self.type,
                                                                               self.NEG,
                                                                               self.file)


class FieldCollection(dict):
    def __init__(self):
        dict.__init__(self)
        self.comment = "#"

    # returns the memmap of appended field
    def append(self,field):
        self[field.getMyIdString()] = field

    def getMemMap(self,field_id,mode='r'):
        fld = self.getField(field_id)
        return fld.getMemMap(mode)

    def getField(self,field_id):
        fid = field_id.getString()
        return self[fid]

    def write(self,fpath):
        with open(fpath, 'w') as fl:
            for val in self.values():
                print(val.makeString(), file=fl)

    def read(self,fname,folder):
        fpath = folder + fname
        try:
            with open(fpath) as fl:
                for line in fl:
                    line = line.strip()
                    if line.startswith(self.comment) or line == "":
                        continue
                    fld = Field()
                    fld.readString(line)
                    fld.findMemMap(folder)
                    self.append(fld)
        except IOError:
            print("File "+fname+" does not exist.")

    def __repr__(self):
        field_string = list()
        [field_string.append(" {0}".format(fld)) for fld in self]
        return "FieldCollection with fields: \n{1}".format("\n".join(field_string))



class SimulationData(object):
    def __init__(self):
        self.work_dir = "."+sysv_separator

        self.siminfo = BasicSimInfo()
        self.fields  = FieldCollection()

    def __contains__(self,field_id):
        return field_id.getString() in self.fields
        
    def setDestination(self,name,path):
        self.setBaseName(name)
        self.setWorkDir(path)

    def setWorkDir(self,dr):
        self.work_dir = sys_testDir(dr)

    def setBaseName(self,name):
        self.siminfo.base_name = name

    # returns memory map to appended field
    def appendField(self,field,mode='w+'):
        mmap_path = self.work_dir + self.siminfo.mmap_dir
        if not os.path.exists(mmap_path):
            os.makedirs(mmap_path)

        field.createFileName(self.siminfo.base_name,
                             self.siminfo.mmap_dir)
        field.findMemMap(self.work_dir)
        self.fields.append(field)

        return field.getMemMap(mode)

    def getMemMap(self,field_id,mode='r'):
        return self.fields.getMemMap(field_id,mode)

    def getField(self,field_id):
        return self.fields.getField(field_id)

    def save(self):
        self.siminfo.write(self.work_dir)
        self.fields.write(self.work_dir+self.siminfo.fields_fn)

        access_info = tuple([self.work_dir,self.siminfo.info_fn])
        print("Path and info file name:")
        print(access_info)

        return access_info

    def load(self,info_file,wrk_dr=None):
        if not wrk_dr == None:
            self.setWorkDir(wrk_dr)

        self.siminfo.read(info_file, self.work_dir)
        self.fields.read(self.siminfo.fields_fn, self.work_dir)

    def cleanUp(self):
        # delete MemMap folder with everything inside
        os.rmdir(self.work_dir + self.siminfo.mmap_dir)
        # delete mmap.fields file
        os.remove(self.work_dir + self.siminfo.fields_fn)
        # delete mmap.info file
        os.remove(self.work_dir + self.siminfo.info_fn)
        print('Clean up the simulation data of: {}'.format(self.siminfo.base_name))

    def changeBaseName(self,new_name):
        # do only something if it is not the same name
        if not new_name == self.siminfo.base_name:
            new_siminfo = BasicSimInfo()
            new_siminfo.read(self.siminfo.info_fn, self.work_dir)
            new_siminfo.setBaseName(new_name)
            new_mmap_dir = self.work_dir + new_siminfo.mmap_dir
            if not os.path.exists(new_mmap_dir):
                os.makedirs(new_mmap_dir)
                
            new_fields = FieldCollection()
            
            while len(self.fields) > 0:
                fld = self.fields.popitem()[1]
                original_file = ''.join(fld.path) # real copy of path
                
                fld.createFileName(new_siminfo.base_name,
                                   new_siminfo.mmap_dir)
                fld.findMemMap(self.work_dir)
                
                new_file = ''.join(fld.path)
                new_fields.append(fld)
                
                # move file
                os.rename(original_file, new_file)

            self.cleanUp()
            self.siminfo = new_siminfo
            self.fields = new_fields

            self.save()

    def __repr__(self):
        return "Simulation Data:\n" + self.siminfo.__repr__() + self.fields.__repr__()


