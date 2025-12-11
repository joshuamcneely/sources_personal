import sys
if not '-m' in sys.argv:
    from .fieldid import FieldId

    from .field import Field

    from .field_collection import FieldCollection
    from .field_collection_iohelper import FieldCollectionIOHelper
    from .field_collection_weakinterface import FieldCollectionWeakInterface
    from .field_collection_analysis import FieldCollectionAnalysis

    from .data_manager import DataManager
    from .data_manager_analysis import DataManagerAnalysis
