#!/usr/bin/env python
#
# utilities.py
#
# useful functions
#
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @date     2015/09/01
# @modified 2020/12/22

from __future__ import print_function, division, absolute_import

import os.path

def sys_test_dir(dr):
    """Retruns the path of directory ends with os.path.sep ('/') if it does not.

    Args:
        dr (str): Path to a directory.

    Returns:
        str: Path of directory ends with os.path.sep ('/').
    """
    if not dr.endswith(os.path.sep) and not len(dr) == 0:
        dr += os.path.sep
    return dr

def path_exists(fname,path=''):
    """Determines whether a file exists in a directory.

    Args:
        fname (str): File name.
        path  (str): Path to a directory.
    
    Returns:
        bool: True if the file {fname} exists in the directory of {path}.
    """
    path = sys_test_dir(path)
    return os.path.exists(path+fname)

def string_is_int(s):
    """Determines whether a string can be parsed to an integer.

    Args:
        s (str): String to be determined.
    
    Returns:
        bool: True if {s} can be parsed to an integer.
    """
    try: 
        int(s)
        return True
    except ValueError:
        return False
