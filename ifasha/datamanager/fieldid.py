#!/usr/bin/env python
#
# fieldid.py
#
# identifies a field with i and j component
# 
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @date     2015/09/01
# @modified 2020/12/22

from __future__ import print_function, division, absolute_import

from .utilities import string_is_int

class FieldId():
    """Identity of Field

    Attributes:
        sep  (str): String separator.
        name (str): Name of FieldId.
        _i     (*): Component i.
        _j     (*): Component j.
    """
    def __init__(self,nm=None,i=None,j=None):
        """Creates FieldId object.

        Args:
            nm (str): Name of FieldId
            i    (*): Component i.
            j    (*): Component j.
        
        Raises:
            RuntimeError: When name contains string separator, '_'.
        """
        self.sep = '_'

        if len(self.get_string_components(nm)) > 1:
            print('FieldId cannot be named as: '+nm
                  +' it looks like it has components.')
            raise RuntimeError
        self.name = nm

        # components i,j
        # should not be modified
        self._i    = i
        self._j    = j

    # define comparison operators
    # use string to compare
    def __eq__(self, other):
        """Allows the usage of comparison operator, '=='.

        Args:
            other (FieldId): The FieldId to be compared.

        Returns: 
            bool: True if the string representations of self and the other are identical.
        """
        return self.get_string() == other.get_string()

    def __ne__(self, other):
        """Allows the usage of comparison operator, '!='.

        Args:
            other (FieldId): The FieldId to be compared.

        Returns: 
            bool: False if the string representations of self and the other are identical.
        """
        return not self.__eq__(other)

    # make a list of strings become a list of FieldIds
    # can also receive just one string
    @staticmethod
    def string_to_fieldid(strings):

        was_iterable = True
        if isinstance(strings, str):
            strings = [strings]
            was_iterable = False

        fids = list()
        for string in strings:
            if isinstance(string, str):
                fid = FieldId()
                fid.load_string(string)
            else:
                fid = string
            fids.append(fid)

        if not was_iterable and len(fids)!=0:
            return fids[0]
        else:
            return fids


    # create a string with all the information of this FieldId
    def get_string(self):
        """Returns a string with all the information of this FieldId.

        Returns:
            str: A string with all the information of this FieldId.
        """
        output = str(self.name)
        if not self._i == None:
            output = output + self.sep + str(self._i)
        if not self._j == None:
            output = output + self.sep + str(self._j)
        return output
    
    # load a string that was created by the get_string function
    def load_string(self,string):
        """Load information from a string.

        Args:
            string (str): A string with information of FieldId.
        
        Raises:
            RuntimeError: When string is empty or has more than 2 components.
        """
        if not string:
            print('string is empty')
            raise RuntimeError

        self._i = None
        self._j = None

        comps = self.get_string_components(string)
        if len(comps) > 3: # first is name
            print('string describing FieldId is corrupted. '
                  +'It cannot have more than 2 components: '+string)
            raise RuntimeError

        self.name = comps[0]
        if len(comps) > 1:
            self._i = comps[1]
        if len(comps) > 2:
            self._j = comps[2]

    # returns list with first element being the name 
    # and the following ones being the components
    def get_string_components(self,nm):
        """Returns a list with first element being the name and the following ones being the components.

        Returns:
            list(string): A list with first element being the name and the following ones being the components.
        """
        if nm == None:
            return list()

        all_comps = str(nm).strip().split(self.sep)

        # find number components
        comps = list()
        for comp in reversed(all_comps):
            if string_is_int(comp):
                comps.insert(0,comp)
            else:
                break

        if len(comps):
            del all_comps[-len(comps):]

        # add the other part as entire name
        comps.insert(0,self.sep.join(all_comps))

        return comps


    def __repr__(self):
        """Returns a printable representation of field identity.

        Returns:
            str: String with information of field identity.
        """
        return self.get_string()
