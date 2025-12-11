#!/usr/bin/env python
#
# datamanageranalysis.py
#
# code to get access to FieldCollectionAnalysis
#
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @date     2013/05/21
# @modified 2020/12/22

from __future__ import print_function, division, absolute_import

from .data_manager import DataManager
from .field_collection_analysis import FieldCollectionAnalysis


# only to get faster and easier access to FieldCollectionAnalysis
class DataManagerAnalysis(DataManager):
    """Manages data from simulations with access to FieldCollectionAnalysis.

    DataManagerAnalysis inherits DataManager and adds access to FieldCollectionAnalysis.
    """

    # cannot create new data structure
    def __init__(self,name,wdir='.'):
        """Creates DataManagerAnalysis object.

        Args:
            name   (str) : Job name.
            wdir   (str) : Path to postprocessed data. Defaults to '.'.
        """
        DataManager.__init__(self,name,wdir,False)

    # access to field collections used for analysis
    def get_field_collection_analysis(self,fcname):
        """Returns field collection analysis of requested name.

        Note: Code is not save when the requested object does not exist.

        Args:
            fcname (str): Name of field collection analysis.

        Returns:
            FieldCollectionAnalysis: Requested field collection analysis.
        """

        fc = DataManager.get_field_collection(self,fcname)
        fca = FieldCollectionAnalysis(fc)
        return fca

    # short access
    def __call__(self, fcname):
        """Allows the usage of the () operator.

        This method is a shortcut to {get_field_collection_analysis}.

        Returns:
            FieldCollectionAnalysis: Requested field collection analysis.
        """
        return self.get_field_collection_analysis(fcname)
        
