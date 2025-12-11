#!/usr/bin/env python

# change_name.py
#
# Code to change name of postprocessed data from simulations
#
# There is no warranty for this code
#
# @version 0.1
# @author David Kammer <kammer@cornell.edu>
# @date     2016/07/11
# @modified 2017/05/25
from __future__ import print_function, division, absolute_import

import sys

from ifasha.datamanager import DataManager

def change_name(old_bname, new_bname, mode, **kwargs):

    modes = ['interactive', 'save']
    if mode not in modes:
        print('Choose a possible mode:',modes)
        raise RuntimeError

    wdir = kwargs.get('wdir','./data')

    if mode == 'interactive':
        print('old_bname={}'.format(old_bname))
        print('new_bname={}'.format(new_bname))

        # check with user if that is what he/she wants to do
        answer = input('Do you want to change the name of data according to the above mentioned names?\n'
                           + 'Answer with: [y] yes or [n] no? ')
        answer = answer.strip().lower()
        if answer == 'y':
            pass
        elif answer == 'n':
            sys.exit('you said NO.')
        else:
            sys.exit('Incorrect answer!')
            
    # load datamanager
    dm = DataManager(old_bname,wdir)

    # change name: will fail if something with new_bname exists already
    dm.change_name(new_bname)



if __name__ == "__main__":

    if len(sys.argv) != 3:
        sys.exit('Missing argument! usage: ./change_name.py old-basename new-basename')

    change_name(sys.argv[1], sys.argv[2], 'interactive')
