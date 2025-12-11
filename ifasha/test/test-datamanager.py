#!/usr/bin/env python3

from __future__ import print_function, division, absolute_import

import numpy as np
import os.path
import shutil as shtl
import sys

import ifasha.datamanager as idm
from ifasha.datamanager.utilities import path_exists
from ifasha.datamanager.info_folder_handler import InfoFolderHandler

# -----------------------------------------------------
    
sbfld = './subfolder'
if os.path.exists(sbfld):
    shtl.rmtree(sbfld)
os.makedirs(sbfld)

# TEST INFOFOLDERHANDLER

# raise exception because it does not exist and cannot create
try:
    ifh3 = InfoFolderHandler('ifhtest3','.')
except IOError:
    print("Catched the 'InfoFolderHandler does not exist - do not create' error")
else:
    print("should have raised an error for the 'does not exist - do not create' error")
    raise RuntimeError

# test creating and adding groups
name2 = 'ifhtest2'
name2a = 'ifhtest2a'
ifh2 = InfoFolderHandler(name2,sbfld,True) # test creation
if not os.path.exists(sbfld+'/'+ifh2.create_info_file_name(name2)):
    print('Did not create the info file')
    raise RuntimeError
if not os.path.exists(sbfld+'/'+ifh2.create_data_folder_name(name2)):
    print('Did not create data folder')
    raise RuntimeError
ifh2.change_name(name2a) # test changing name
if not os.path.exists(sbfld+'/'+ifh2.create_info_file_name(name2a)):
    print('Did not change name of info file')
    raise RuntimeError
if not os.path.exists(sbfld+'/'+ifh2.create_data_folder_name(name2a)):
    print('Did not move data folder')
    raise RuntimeError
ifh2.destroy() # test destroying InfoFolderHandler
if os.path.exists(sbfld+'/'+ifh2.create_info_file_name(name2a)):
    print('Did not erase info file')
    raise RuntimeError
if os.path.exists(sbfld+'/'+ifh2.create_data_folder_name(name2a)):
    print('Did not erase data folder')
    raise RuntimeError

# test reading
name1 = 'ifhtest1'
file1 = """
# test comment

!datafolder
difffoldername-datamanager-files

#test comment"""
with open(sbfld+'/'+name1+'.info', "w") as inp:
    print(file1,file=inp)

ifh1 = InfoFolderHandler(name1,sbfld)
if not os.path.exists(sbfld+'/'+ifh1.data_dir):
    print('Did not create missing data folder')
    raise RuntimeError
ifh1.destroy()
ifh1.destroy() # check double destruction

# test compressing
compfilename = 'file-to-comp.txt'
compfilecont = 'abc 432 ust'
ifh1 = InfoFolderHandler(name1,sbfld,True)
ifh1path = sbfld+'/'+ifh1.data_dir
with open(ifh1path+'/'+compfilename,'w') as inp:
    print(compfilecont,file=inp)
ifh1.pack()
# check if pack twice nothing goes wrong
ifh1.pack()
# check if change name with zip file works
ifh1.change_name(name2)

ifh2 = InfoFolderHandler(name2,sbfld,False)
ifh2path = sbfld+'/'+ifh2.data_dir
# unpack and compare input
ifh2.unpack()
if not compfilecont.strip() == open(ifh2path+'/'+compfilename,'r').read().strip():
    print('file does not contain the original content after packing and unpacking it')
    print('before: {}'.format(compfilecont))
    print('after:  {}'.format(open(ifh2path+'/'+compfilename,'r').read()))
    raise RuntimeError
ifh2.destroy()

ifh1 = InfoFolderHandler(name1,sbfld,True)
ifh1path = sbfld+'/'+ifh1.data_dir
with open(ifh1path+'/'+compfilename,'w') as inp:
    print(compfilecont,file=inp)
ifh1.pack()
# check if destroy of compressed folder works
ifh1.destroy()
if os.path.exists(sbfld+'/'+ifh1.create_info_file_name(name1)):
    print('Did not erase info file')
    raise RuntimeError
if os.path.exists(sbfld+'/'+ifh1.create_zip_file_name(name1)):
    print('Did not erase compressed data folder')
    raise RuntimeError

print('Test of InfoFolderHandler: SUCCESSFULL')
print('--------------------------------------------------------------------------------------')


# TEST FIELDID

# test creation of FieldId
fldn1 = 'fieldt1'
fldid1 = idm.FieldId(fldn1)
print(fldid1)
fldid1 = idm.FieldId(fldn1,0)
print(fldid1)
fldid1 = idm.FieldId(fldn1,2,1)
print(fldid1)
fldid2 = idm.FieldId()
fldid2.load_string(fldid1.get_string())
if not fldid1 == fldid2:
    print('get_string and then load_string does not result in the same FieldId')
    raise RuntimeError
# test loading a string for FieldId
strbnm = 'abcd_1_efg'
comp1 = '4'
comp2 = '2'
for ext in ['','_'+comp1,'_'+comp1+'_'+comp2]:
    fldid1.load_string(strbnm+ext)
    if not fldid1.name == strbnm:
        raise RuntimeError
try:
    fldid1.load_string('abcd_1_efg_2_4_5')
except RuntimeError:
    print("Caught the 'too many components' error")
else:
    print("should have raised an error for the 'too many components' error")
    raise RuntimeError

print('Test of FieldId: SUCCESSFULL')
print('--------------------------------------------------------------------------------------')


# TEST FIELD

strn1 = 'fld1_3_2 22 62 float64 N diff_name.mmp' # file name is different
fld = idm.Field()
# check string creation and loading
fld.load_string(strn1)
if not fld.get_string() == strn1:
    print('load_string and get_string do not work in the same way!')
    print(strn1)
    print(fld.get_string())
    raise RuntimeError
# check MemMap creation
fld.set_path(sbfld)
fld.get_memmap('w+')
if not fld.memmap_exists():
    print('MemMap creation failed!')
    raise RuntimeError
# check creation with partial information
fldid2 = idm.FieldId('fld2',4,5)
try:
    fld2 = idm.Field(fldid2)
except RuntimeError:
    print('caught the "partial information" error in Field creation')
else:
    print('did not catch the "partial information" error during Field creation')
    raise RuntimeError
# check creation with information
fld2 = idm.Field(fldid2,34,56,'float64','N')
fld2.set_path(sbfld)
fld2.get_memmap('w+')
# check renaming field failure because other MemMap exists already
try:
    fld.change_identity(fldid2)
except RuntimeError:
    print('caught the "MemMap exists error" while changing identity of Field')
    fld.load_string(strn1) # restore previous state
else:
    print('did not catch the "MemMap exists error" while changing identity of Field')
    raise RuntimeError
# check renaming field
nnm1 = 'renamed'
fld.change_identity(idm.FieldId(nnm1,4,5))
if nnm1 not in fld.mmap or nnm1 not in fld.get_id_string():
    print('MemMap renaming failed!')
    raise RuntimeError
if not fld.memmap_exists():
    print('MemMap renaming failed!')
    raise RuntimeError    
# check print of Field
print(fld) 
# check filling memmap
mmap_to_fill = fld.get_memmap('r+')
mmap_shp = fld.get_shape()
for t in range(mmap_shp[0]):
    for n in range(mmap_shp[1]):
        mmap_to_fill[t,n] = t*n
if not mmap_to_fill[-1,-1] == 1281:
    print('did not fill memmap correct')
    raise RuntimeError
# check croping memmap in time
crop_tstep = int(mmap_shp[0] / 2)
fld.crop_at_time_step(crop_tstep)
cropped_mmap = fld.get_memmap('r')
if not cropped_mmap[-1,-1] == 630:
    print("did not time crop memmap correctly")
    raise RuntimeError
# check croping memmap in space
Nold = fld.get_shape()[1]
crp = [True for i in range(Nold)]
crp[0] = False
crp[-1] = False
fld.filter(crp)
cropped_mmap = fld.get_memmap('r')
if not cropped_mmap[-1,0] == 30 or not cropped_mmap[-1,-1] == 600:
    print("did not crop memmap correctly")
    raise RuntimeError
# check destruction of MemMap
fld.destroy_memmap()
if fld.memmap_exists():
    print('MemMap was not destoyed!')
    raise RuntimeError
fld.destroy_memmap() # check what happens if MemMap does not exist
fld2.destroy_memmap() # clean

print('Test of Field: SUCCESSFULL')
print('--------------------------------------------------------------------------------------')


# TEST FIELDCOLLECTION

fcn1 = 'fc-test1'
fldn1 = 'fld1-fc1'
i1 = 3
j1 = 2
N1 = 56
nbts1 = 123
vl1 = 33.64
# check creating new FieldCollection
fc1 = idm.FieldCollection(fcn1,sbfld,True)
# check creating new Field in FC
fld1 = fc1.get_new_field(idm.FieldId(fldn1,i1,j1),N1,nbts1,'float64','N')
mmp1 = fld1.get_memmap('w+')
mmp1[nbts1-1,N1-1] = vl1

# check adding existing Field into FieldCollection
strn3 = 'fld3-fc1_7 2 17 float64 N fld3-fc1_7.mmp'
fld3 = idm.Field()
fld3.load_string(strn3)
fld3.set_path(sbfld)
fld3.get_memmap('w+')
fc1.add_field(fld3)
if fld3.get_id_string() not in fc1.fields:
    print('Field was not added to the FieldCollection')
    raise RuntimeError
if not path_exists(strn3.split()[-1],fc1.get_data_folder_path()):
    print('MemMap was not created or created in a wrong folder')
    raise RuntimeError

# check loading existing FieldCollection
fc2 = idm.FieldCollection(fcn1,sbfld)
# check getting MemMap of Field
mmp2 = fc2.get_field_memmap(idm.FieldId(fldn1,i1,j1))
if not mmp1[nbts1-1,N1-1] == mmp2[nbts1-1,N1-1]:
    print('The reloaded MemMaps does not contain the value as assigned')
    raise RuntimeError

# check changing name of field
fldn1a = 'fld1a-fc1'
i1a = 4
j1a = 5
fc2.change_field_identity(idm.FieldId(fldn1,i1,j1),idm.FieldId(fldn1a,i1a,j1a))
if fc2.has_field(idm.FieldId(fldn1,i1,j1)):
    print("Field collection should not have this field anymore as it has changed identity")
    raise RuntimeError
if not fc2.has_field(idm.FieldId(fldn1a,i1a,j1a)):
    print("Field collection should have a field that had its identity changed")
    raise RuntimeError
# check that you don't change it to something that exist already
try:
    fc2.change_field_identity(idm.FieldId(fldn1a,i1a,j1a),idm.FieldId(fldn1a,i1a,j1a))
except RuntimeError:
    print('caught the changing identity of Field to existing field error')
else:
    print('did not catch the changing identity of Field to existing field error')
    raise RuntimeError

# check croping of all field
crop_value = 3
fc1.crop_fields_at_time_step(crop_value)
for fld in fc1.fields.values():
    if fld.get_shape()[0] != crop_value:
        print('did not crop all fields correctly')
        raise RuntimeError
#check that it saved it in file as well
fc4 = idm.FieldCollection(fcn1,sbfld)
for fld in fc4.fields.values():
    if fld.get_shape()[0] != crop_value:
        print('did not save after crop')
        raise RuntimeError

# check print
print(fc2)

# check that you cannot add Field into packed FieldCollection
fc1.pack()
fc3 = idm.FieldCollection(fcn1,sbfld)
fldn1p = 'fld1-fc1p'
try:
    fld1p = fc1.get_new_field(idm.FieldId(fldn1p,i1,j1),N1,nbts1,'float64','N')
except RuntimeError:
    print('caught the error for trying to get a new Field in a packed FieldCollection')
else:
    print('did not catch the error for trying to get a new Field in a packed FieldCollection')
    raise RuntimeError
strn3p = 'fld3-fc1p_7 2 17 float64 N fld3-fc1_7.mmp'
fld3p = idm.Field()
fld3p.load_string(strn3p)
fld3p.set_path(sbfld)
fld3p.get_memmap('w+')
try:
    fc1.add_field(fld3p)
except RuntimeError:
    print('caught the error for trying to add a Field into a packed FieldCollection')
else:
    print('did not catch the error for trying to add a Field into a packed FieldCollection')
    raise RuntimeError
fc1.unpack()

# check destruction of FieldCollection
fc1.destroy()
# check destruction of FieldCollection that does not exist
fc2.destroy()

print('Test of FieldCollection: SUCCESSFULL')
print('--------------------------------------------------------------------------------------')


# TEST DATAMANAGER

dmn1 = 'dm-test1'
flnm2 = '{}/{}.datamanager.info'.format(sbfld,dmn1)
fold2 = '{}/{}-datamanager-files'.format(sbfld,dmn1)
missfc = 'missing-field-collection'
file2 = """!datafolder
{}

!field-collections
{}

!supplementary
missing-supplementary""".format(fold2.split('/')[-1],
                                missfc)
with open(flnm2, 'w') as inp:
    print(file2,file=inp)
os.makedirs(fold2)

# check loading DataManager
dm1 = idm.DataManager(dmn1,sbfld,True)

# check adding existing FieldCollection
try:
    dm1.get_new_field_collection(missfc)
except RuntimeError:
    print('Caught the "FieldCollection exists already" error')
else:
    print('Did not catch the "FieldCollection exists already" error')
    raise RuntimeError

# check adding new FieldCollection
fc1n = 'nfc1'
fc1 = dm1.get_new_field_collection(fc1n)
fc1t = idm.FieldCollection(fc1n,dm1.get_data_folder_path()) # check if FieldCollection exists

# check adding new Fields to these FieldCollections
fldid = idm.FieldId('fld56')
fc1.get_new_field(fldid,12,34,'float64','N')

# test accessing FieldCollection
fc1tt = dm1.get_field_collection(fc1n) 

# load new DataManager to erase FieldCollection before removing it from other DataManager
dm2 = idm.DataManager(dmn1,sbfld)
dm2.remove_field_collection(fc1n) # erases all data of FieldCollection
dm1.remove_field_collection(fc1n) # test removal of FieldCollection which has not data anymore
dm1.remove_field_collection(fc1n) # test removal of inexisting FieldCollection


# check adding supplementary files
flnm1 = 'input.in'
file1 = """This is an input file"""
with open(flnm1, "w") as inp:
    print(file1,file=inp)
snm1 = 'input.txt'
dm1.add_supplementary(snm1,flnm1,True)
if not snm1 in open(dm1.get_info_file_path()).read():
    print('Did not add supplementary file to info file')
    raise RuntimeError
if not os.path.exists(dm1.get_data_folder_path()+snm1):
    print('Did not copy supplementary file to data folder')
    raise RuntimeError

# check removing supplementary
dm1.remove_supplementary('bla') # check removing a non-existing supplementary
dm1.remove_supplementary(snm1)
if snm1 in open(dm1.get_info_file_path()).read():
    print('Did not remove supplementary file from info file')
    raise RuntimeError
if os.path.exists(dm1.get_data_folder_path()+snm1):
    print('Did not delete supplementary file from data folder')
    raise RuntimeError

# check if packed DataManager is not used for things it cannot be used
dm1.pack()
fc1np = 'nfc1p'
try:
    fc1p = dm1.get_new_field_collection(fc1np)
except RuntimeError:
    print('Caught the get new FieldCollection error for packed DataManager.')
else:
    print('Did not catch the get new FieldCollection error for packed DataManager.')
    raise RuntimeError

try:
    fc1tt = dm1.get_field_collection(fc1np) 
except RuntimeError:
    print('Caught the get FieldCollection error for packed DataManager.')
else:
    print('Did not catch the get FieldCollection error for packed DataManager.')
    raise RuntimeError
dm1.unpack()

# test destruction of DataManager
dm1.destroy()
dm1.destroy() # check destruction of destroyed DataManager

print('Test of DataManager: SUCCESSFULL')
print('--------------------------------------------------------------------------------------')


# TEST IOHELPERTEXTDUMPERREADER

test_sdir = sbfld
test_sdir = './test-datamanager-sourcefolder'
test_bname = 'iohelper-given-name'
test_bname = 'supershear_dynamic_nucD2.0e-04_taui4.50e+06_sigi5.00e+06_E5.65e+09nu0.33rho1180psss1_lswnhs1.06k0.74d1.40e-06_L0.2H0.1top'#_elem3200'
test_bname = 'verify_data'
groups = ['gr1','gr2']
groups = ['elem3200']
groups = ['interface']

new_bname = 'dm-given-name'

#dmioh = DataManager(new_bname,sbfld,True)

# verify it does not yet exist
try:
    dmioh = idm.DataManager(new_bname,sbfld,False)
except IOError: # it does not yet exist
    dmioh = idm.DataManager(new_bname,sbfld,True)
else:
    answer = raw_input('DataManager named "{}" exists already.\n' 
                       + 'Want to: [m] modify it, [r] replace it, [s] stop here? '.format(dmioh.name))
    answer = answer.strip().lower()
    if answer == 's':
        sys.exit()
    elif answer == 'r':
        dmioh.destroy()
        dmioh = idm.DataManager(new_bname,sbfld,True)
    elif answer == 'm':
        pass
    else:
        print('Incorrect answer!')
        sys.exit()

for group in groups:

    # check if FieldCollection exist already
    if group in dmioh:
        answer = raw_input('FieldCollection named "{}" exists already.\n'
                           + 'Want to: [m] modify it, [r] replace it, [s] stop here? '.format(group))
        answer = answer.strip().lower()

        if answer == 's':
            sys.exit()
        elif answer == 'r':
            dmioh.remove_field_collection(group)
            fc = dmioh.get_new_field_collection(group)
        elif answer == 'm':
            fc = dmioh.get_field_collection(group)
        else:
            print('Incorrect answer!')
            sys.exit()

    else:
        fc = dmioh.get_new_field_collection(group)

    # "cast" FieldCollection to IOHelperReader
    fcioh = idm.FieldCollectionIOHelper(fc)
    fcioh.read_simulation_output(test_bname+'-'+group+'.info', test_sdir)

    # --------------------------------------
    # check data
    print(fcioh)
    iv = 8
    jv = -1
    vls = list()

    fldids_verify = [idm.FieldId('position',0), 
                     idm.FieldId('friction_traction',0), 
                     idm.FieldId('nb_slip_nodes',0)]
    fc_verify = dmioh.get_field_collection(group)
    for fldid_verify in fldids_verify:
        fld_verify = fc_verify.get_field(fldid_verify)
        mmap_verify = fld_verify.get_memmap()

        fld_verified = idm.Field()
        fld_verified.load_string(fld_verify.get_string())
        fld_verified.mmap = 'verified_' + fld_verify.mmap # normally not to be done (here only for check)
        fld_verified.set_path(test_sdir)
        mmap_verified = fld_verified.get_memmap()

        vls.append(mmap_verified[iv,jv]) # used to check afterwards the DataManagerAnalysis

        for (i,j), value in np.ndenumerate(mmap_verify):
            if not value == mmap_verified[i,j]:
                print('wrong result: ({},{})={} not = {}'.format(i,j,value,mmap_verified[i,j]))
                raise RuntimeError


print('Test of DataManagerIOHelper: SUCCESSFULL')
print('--------------------------------------------------------------------------------------')


# TEST DATAMANAGERANALYSIS
vlsc = list()

dma = idm.DataManagerAnalysis(new_bname,sbfld)
fca = dma(groups[0])
for fldid_verify in fldids_verify:
    tmp = fca.get_field_at_t_index(fldid_verify,iv)[0]
    if not fca.get_field(fldid_verify).NEG == 'G':
        tmp = tmp[-1]
    vlsc.append(tmp)
    
if not vlsc == vls:
    print('DataManagerAnalysis does not find correct values')
    print('correct:',vls)
    print('but has:',vlsc)
    raise RuntimeError

print('Test of DataManagerAnalysis: SUCCESSFULL')
print('--------------------------------------------------------------------------------------')
