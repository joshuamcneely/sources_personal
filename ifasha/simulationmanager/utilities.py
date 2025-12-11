#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import os.path
import sys
import getopt
import fnmatch

# utilities.py
#
# useful functions
#
# There is no warranty for this code
#
# @version 0.1
# @author David Kammer <davekammer@gmail.com>
# @author Gabriele Albertini <galberti819@gmail.com>
# @date     2015/09/01
# @modified 2015/09/29

def sys_test_dir(dr):
    if not dr.endswith(os.path.sep) and not len(dr) == 0:
        dr += os.path.sep
    return dr

def path_exists(fname,path=''):
    path = sys_test_dir(path)
    return os.path.exists(path+fname)

def string_is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def find_all(pattern, path):
    """Finds all pattern in a path, including sub-folders. Folders are not found
    Arguments:
        pattern (str): file name or part of it e.g. '*.txt'
        path (str): '/path/to/dir'
    
    Returns:
        list of strings with '/path/to/file'
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def find(name, path):
    """Finds file in a path, including sub-folders. Folders are not found
    Arguments:
        pattern (str): file name e.g. 'text.txt'
        path (str): '/path/to/dir'

    Returns:
        (str): '/path/to/file'
    """
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def exist(name, path):
    """Asses whether a file in a specific path exists
        Arguments:
        pattern (str): file name e.g. 'text.txt'
        path (str): '/path/to/dir'

    Returns:
        (bool)
    """
    if find(name, path) == os.path.join(path,name):
        return True
    else:
        return False
        
def execute(args):
    try:
        sys.argv = [str(args[0])] 
        for a in args:
            sys.argv.append(str(a))
        execfile(str(args[0]))
    except:
        print('Could not execute {}'.format(args))

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
