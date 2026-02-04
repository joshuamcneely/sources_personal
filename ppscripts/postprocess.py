#!/usr/bin/env python

# postprocess.py
#
# Code to postprocess data from simulations
#
# WARNING:
# important comment: For this to work, 
# sim-ids have to be pure digits, and 
# bnames cannot be pure digits
#
# There is no warranty for this code
#
# @version 0.3
# @author David Kammer <dkammer@ethz.ch>
# @date     2015/11/18
# @modified 2020/03/13
from __future__ import print_function, division, absolute_import

import sys
import time
import glob
import os.path
import shutil as shtl
import filecmp
import itertools as ittools

import ifasha.datamanager as idm

# basename for simulation names
def get_basename():
    return [line.strip() for line in open('basename.txt')][0]

# turn sname to simid
def sname_to_simid(sname):
    parts = sname.split('_')
    bname = get_basename()
    if not bname == '_'.join(parts[0:-1]):
        raise RuntimeError('simulation name {} does not contain basename {}'.format(sname,bname))
    return int(parts[-1])

# turn simid to sname
def simid_to_sname(simid):
    return get_basename()+'_'+str(simid)

# make sure that sname is actually a sname
def sname_to_sname(sname):
    if isinstance(sname, int) or sname.isdigit():
        sname = simid_to_sname(sname)
    return sname

# read a string with , and - to a list of simids
def string_to_simidlist(simidstring):
    simidlist = simidstring.split(',')
    simids = []
    for simid in simidlist:
        nbdashes = simid.count('-')
        if nbdashes == 0:
            simids.append(simid)
        elif nbdashes == 1:
            bounds = simid.split('-')
            simids.extend([str(i) for i in range(int(bounds[0]), int(bounds[1])+1)])
        else:
            print('do not understand syntax for {}'.format(simid))
            raise RuntimeError
    return simids

# get code used for simulation
def get_code_name(sname, **kwargs):

    sname = sname_to_sname(sname)
    
    # input file is stored with sim-id (see below)
    wdir = kwargs.get('wdir','./data')
    dm = idm.DataManager(sname,wdir)
    
    ofile = dm.get_supplementary_content(sname+'.out').split('\n')
    
    sim_codes = list()
    for line in ofile:
        line = line.strip().split()
        if len(line) == 0:
            continue
        if line[0] == 'simulation_code':
            sim_codes.append(line[-1])
            break

    sim_codes = set(sim_codes)
    if len(sim_codes) > 1:
        print(sim_codes)
        raise RuntimeError("multiple codes detected")
    elif len(sim_codes) == 1:
        return sim_codes.pop()
    else:
        return 'unkown'

# input file name
def get_input_fname(sname):
    sname = sname_to_sname(sname)
    return '{}.in'.format(sname)

# get input data from input file
def get_input_data(sname, **kwargs):

    sname = sname_to_sname(sname)

    wdir = kwargs.get('wdir','./data')
    code = kwargs.get('code',None)

    # input file is stored with sim-id (see below)
    dm = idm.DataManager(sname,wdir)
    input_file = dm.get_supplementary_content(get_input_fname(sname))
    input_file = input_file.split('\n')

    if code == None:
        code = get_code_name(sname, **kwargs)

    if code == 'weak-interface':
        comment_char = '#'
        set_char = '='

        input_data = dict()
        for line in input_file:
            line = line.strip().split(comment_char)[0]
            if not line:
                continue
            line = line.split(set_char)

            line = line[0].strip().split()+ [line[1]]
            if len(line)==3:
                _, key,value = line
            else:
                key,value = line
            try: value = float(value) # to avoid numbers as string
            except: pass
            input_data[key] = value
    elif code == 'akantu':
        comment_char = '#'
        sec_s_char = '['
        sec_e_char = ']'
        sep_char = ' '
        set_char = '='

        section_1 = None
        section_2 = None
        input_data = dict()
        for line in input_file:
            line = line.strip().split(comment_char)[0]
            if not line:
                continue
            line = line.split(sep_char)
            line = [l for l in line if l != '']
            if line[-1][-1]==sec_s_char: line.append(sec_s_char)# to fix a bug with a missing space 
            # start of new section
            if sec_s_char in line and set_char not in line:
                section_1 = line[0]
                section_2 = line[1]
                tmp_data = dict()
            # end of section
            elif sec_e_char in line and set_char not in line:
                # add temp dict to input_data
                if section_1 == 'user' and section_2 == 'parameters': # these are the general value
                    input_data.update(tmp_data)
                else:
                    key='{}-{}'
                    nb=0
                    while key.format(section_1,nb) in input_data:
                        nb += 1
                    tmp_data['type']=section_2
                    input_data[key.format(section_1,nb)] = tmp_data
                section_1 = None
                section_2 = None
            else:
                key = line[0].strip()
                idx = line.index(set_char)
                if idx == 1 and len(line) == 3:
                    value = line[2].strip()
                elif idx == 1:
                    value = ' '.join(line[idx+1:])
                else:
                    print("don't understand: {}".format(' '.join(line)))
                    print(line)
                    continue
                try: value = float(value) # to avoid numbers as string
                except: pass
                tmp_data[key] = value
        for key in input_data:
            # string value to vector if possible
            if isinstance(input_data[key], basestring) \
               and '[' in input_data[key] and ']' in input_data[key]:
                values = input_data[key].split('[')[1].split(']')[0]
                values = [i.strip() for i in values.split(',')]
                try: values = [float(i) for i in values]
                except: pass
                input_data[key] = values
            # string to boolean
            if input_data[key]=='true':
                input_data[key]=True
            if input_data[key]=='false':
                input_data[key]=False

            
    else:
        raise RuntimeError('Do not know the used code'
                           +' and thus do not know how to read input file')

    return input_data



def postprocess(sim_id,mode,new_sname=None):

    modes = ['interactive', 'save', 'forced']
    if mode not in modes:
        print('Choose a possible mode:',modes)
        raise RuntimeError

    # -------------------------------------------------------------------------
    # potential source directories
    sdirs = [line.strip() for line in open('source_directories.txt')]
    print(sdirs)

    # basename for simulation names
    basename = get_basename()
    print('basename={}'.format(basename))
    
    # destination of postprocessing
    wdir = './data/'

    # destination for input files for backup
    bdir = './idata/'
    
    # -------------------------------------------------------------------------
    print('sim-id={}'.format(sim_id))

    # find o-files
    ofiles = list()
    for s in sdirs:
        if basename:
            ofpattern = '{}{}_{}.progress'.format(s+os.path.sep,basename,sim_id)
        else:
            ofpattern = '{}{}.progress'.format(s+os.path.sep,sim_id)
        #ofpattern = '{}{}.ubwonko.progress'.format(s+os.path.sep,sim_id) # please do not commit this!
        print('search for o-files with: {}'.format(ofpattern))
        ofile = glob.glob(ofpattern)
        ofiles.extend(ofile)
        if len(ofile):
            sdir = s
            
    # is there a unique ofile?
    if not len(ofiles):
        print('No o-files found!')
        return False,None
    if len(ofiles) > 1:
        print('multiple o-files with same sim-id found! check if all are the same')
        for o in ofiles: print(o)
        for comb in ittools.combinations(ofiles,2):
            if not filecmp.cmp(comb[0], comb[1], shallow=False):
                if mode in ['save', 'interactive']:
                    return False,None

    # try to find at least one that opens
    for ofile in ofiles:
        try:
            open(ofile)
        except IOError:
            time.sleep(5) # maybe the files are still being written: give it a bit time
            pass
        else:
            break
    print('ofile={}'.format(ofile))

    # get from the o-file: path to output-files, basename, groups that are dumped
    sname = ''
    bname_sep = ''
    outputfolder = ''
    groups = set()
    with open(ofile) as fl:
        for line in fl:
            line = line.strip().split()
            if not len(line):
                continue
            key = line[0]
            if key == 'dumper_bname':
                sname = line[2]
            elif key == 'bname_sep':
                bname_sep = line[2]
            elif key == 'dumper_group':
                groups.add(line[2])
            elif key == 'output_folder':
                outputfolder = line[2]
            elif key == 'simulation_code':
                code = line[2]
            else:
                pass
    print('sname={}'.format(sname))
    print('bname_sep={}'.format(bname_sep))
    print('outputfolder={}'.format(outputfolder))
    print('groups={}'.format(groups))

    # check if outputfolder is relative or absolut path
    pot_dumppaths = [sdir + os.path.sep + outputfolder,
                     outputfolder]
    good_dumppaths = [dp for dp in pot_dumppaths if os.path.exists(dp)]
    if len(good_dumppaths) > 0:
        # choose one: first
        dumppath = good_dumppaths[0]
    else:
        # test local path
        print('Could not find dumppath. Check folder of output file.')
        dumppath = os.path.dirname(ofile)
        pot_info_file = dumppath+os.path.sep+sname+bname_sep+list(groups)[0]+'.info'
        if not os.path.exists(pot_info_file):
            raise RuntimeError('None of my potential dumppaths seems to exist.')
    if not dumppath.endswith(os.path.sep):
        dumppath = dumppath + os.path.sep
    print('dumppath={}'.format(dumppath))

    # find input file
    input_file = os.path.join(os.path.dirname(ofile),sname+'.in') #original
    if not os.path.exists(input_file):
        input_file = dumppath + sname + '.in' #copied
    if not os.path.exists(input_file):
        print('Could not find input file: {}'.format(input_file))
        input_file = None
        if mode == 'save':
            return False,None
    print('input_file={}'.format(input_file))

    # FORCE new_sname to be consistent with external numbering if not explicitly provided
    if new_sname is None:
        new_sname = "{}_{}".format(basename, sim_id)
    
    print('new_sname={}'.format(new_sname))

    # new basename if needed
    if new_sname == None:
        new_sname = sname
    print('new_sname={}'.format(new_sname))


    # --------------------------------------------------------------------------
    # do the actual post-processing
    # --------------------------------------------------------------------------

    # verify it does not yet exist
    uidentify=''
    try:
        dmioh = idm.DataManager(new_sname,wdir,False)
    except IOError: # it does not yet exist
        dmioh = idm.DataManager(new_sname,wdir,True)
    else:
        if mode == 'save':
            return False,None
        elif mode == 'interactive':
                answer = input('DataManager named "{}" exists already.\n'.format(dmioh.name)
                       + 'Want to: [m] modify it, [r] replace it, [s] stop here? ')
                answer = answer.strip().lower()
                if answer == 's':
                    sys.exit('you stopped here.')
                elif answer == 'r':
                    dmioh.destroy()
                    dmioh = idm.DataManager(new_sname,wdir,True)
                elif answer == 'm':
                    from datetime import datetime
                    now=datetime.now()
                    uidentify=now.strftime("%Y%m%d%H%M%S")
                    #pass
                else:
                    sys.exit('Incorrect answer!')
        elif mode == 'forced':
            dmioh.destroy()
            dmioh = idm.DataManager(new_sname,wdir,True)


    # add input file to datamanager
    if not input_file == None:
        dmioh.add_supplementary(get_input_fname(new_sname)+uidentify,input_file,True)

    # add o-file to datamanager (replace it)
    dmioh.add_supplementary(new_sname+'.out'+uidentify,ofile,True)

    # find supp files to add
    supfiles = list()
    # .random and .*.random for legacy purposes
    for ft in ['.sub','.run','.*.in','.*.out']:
        supfname = input_file.strip('.in')+ft
        supfile = glob.glob(supfname)
        supfiles.extend(supfile)
    print('supfiles',supfiles)

    # add supp files
    for supfile in supfiles:
        newsupname = new_sname+supfile.split(input_file.strip('.in'))[-1]
        print('add supp file: {} as {}'.format(supfile,newsupname))
        dmioh.add_supplementary(newsupname+uidentify,supfile,True)        

    # copy input file to backup folder
    #shtl.copyfile(input_file,bdir+get_input_fname(new_sname)+uidentify)
        
    # postprocessing each group
    for group in groups:

        # check if FieldCollection exist already
        if group in dmioh:
            answer = input('FieldCollection named "{}" exists already.\n'.format(group)
                               #+ 'Want to: [m] modify it, [r] replace it, [s] stop here? ')
                                + 'Want to: [p] pass it, [r] replace it, [s] stop here? ')
            answer = answer.strip().lower()

            if answer == 's':
                sys.exit('you stopped here')
            elif answer == 'p':
                print('passed!')
                continue
            elif answer == 'r':
                dmioh.remove_field_collection(group)
                fc = dmioh.get_new_field_collection(group)
            # does not work:
            #elif answer == 'm':
            #    fc = dmioh.get_field_collection(group)
            else:
                sys.exit('Incorrect answer!')

        else:
            fc = dmioh.get_new_field_collection(group)

        # this format is used in simid_bname_data as well. 
        # don't forget to change there as well in case you change it here
        fc.sim_info = 'sim-id={}'.format(sim_id)

        # "cast" FieldCollection 
        if code == 'weak-interface':
            fcioh = idm.FieldCollectionWeakInterface(fc)
        # to IOHelperReader
        else:
            fcioh = idm.FieldCollectionIOHelper(fc)
        fcioh.read_simulation_output(sname+bname_sep+group+'.info', dumppath)


    return True,new_sname


if __name__ == "__main__":

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.exit('Missing argument! usage: ./postprocess.py sim-id [mode/new-name]')

    # simid can be 22,24,30-33 -> [22,24,30,31,32,33]      
    if len(sys.argv) == 2:
        simids = string_to_simidlist(str(sys.argv[1]))
        for simid in simids:
            print('postprocess simid={}'.format(simid))
            postprocess(simid,'interactive')
    else:
        # Check if the second argument is a mode
        arg2 = sys.argv[2]
        if arg2 in ['interactive', 'save', 'forced']:
            simids = string_to_simidlist(str(sys.argv[1]))
            for simid in simids:
                print('postprocess simid={} mode={}'.format(simid, arg2))
                postprocess(simid, arg2)
        else:
            # Assume it is a new name (legacy behavior)
            postprocess(sys.argv[1],'interactive',arg2)


    

