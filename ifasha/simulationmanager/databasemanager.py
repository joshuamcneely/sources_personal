#!/usr/bin/env python

"""
databasemanager.py

Code to manage simulations input and outputs

There is no warranty for this code

@version 0.1
@author Gabriele Albertini <galberti819@gmail.com>
@date     2016/03/04
@modified 2016/05/05

"""
from __future__ import print_function
from __future__ import division

import sqlite3
import numpy as np
import os
from ifasha.simulationmanager.utilities import exist

class DatabaseManager:
    """Creates the interface between the SimulationManager and the SQLite database engine.

    Attributes:
        conn (sqlite3.connect): connects to database my_database.sqlite
    
        conn.row_factory (sqlite3.Row): 
            sqlite3.Row class designed to be used as a row factory
            Rows wrapped with this class can be accessed both by 
            index (like tuples) and case-insensitively by name
    
        c (sqlite3.connect.cursor): cursor to execute queries

    Note:
        The best way to enter new rows in the database is by passing the information as dictionaries.
        Dictionary keys corresponds to the column names (fields). This allow more flexibility to the 
        user compare with using lists. The usage of lists for passing information is highly discouraged
        because the user would have to remember the order of the columns, as they were inserted in the 
        database. Many errors can be avoided by using dictionaries instead!

    Advices:
        A graphic interface for viewing the content of SQLite database is available for 
        free as a firefox plugin. It is particularly useful for doing ad-hoc changes, such as adding
        new column, changing single cells content, changing column names etc.
        https://addons.mozilla.org/en-US/firefox/addon/sqlite-manager/
    """
    
    def __init__(self, database_name, source_dir='.', memory=False):
        """Constructor
        creates DatabaseManager object
        connects to database my_database.sqlite
        creates cursor to execute commands
        activates foreign keys
        """
        self.database_name = database_name
        self.source_dir = source_dir
        self.memory = memory
        self.is_new = False
        if self.database_name[-7:] != '.sqlite':
            self.database_name += '.sqlite'
        # if it does not exist it will be crated
        if exist(self.database_name, self.source_dir):
            # loads database form .sqlite file
            self.conn = sqlite3.connect(os.path.join(self.source_dir, self.database_name))
        elif memory:
            # creates database in the ram
            self.conn = sqlite3.connect(':memory:')
        else:
            print('\nWARNING: Database file not found')
            print(os.path.join(self.source_dir, self.database_name))
            create = raw_input('Do you want to create a new database file? yes [y], no [n]')
            if create == 'y':
                self.conn = sqlite3.connect(os.path.join(self.source_dir, self.database_name))
                self.c = self.conn.cursor()
                self.create_database()

        self.conn.row_factory = sqlite3.Row
        # sqlite3.Row class designed to be used as a row factory
        # Rows wrapped with this class can be accessed both by 
        # index (like tuples) and case-insensitively by name

        # creates cursor to execute commands
        self.c = self.conn.cursor()

        # activate foreign keys for table 'linking'
        self.c.execute('PRAGMA foreign_keys = ON;')

    def __del__(self):
        self.conn.close()

    
    # Methods manipulating the structure of the database
    
    def create_database(self):
        """This is method is abstract! 
        It is implemented in the project-specific child class

        It creates a database file my_database.sqlite  
        For each table (input, options, simulation, analysis) the fields (columns) are defined.
        Each field has to be assigned a type (NULL / REAL / INTEGER / TEXT / BLOB). 
        Further attributes are possible (NON NULL / PRIMARY KEY ...)
        
        Examples:

            CREATE TABLE input (
                id INTEGER PRIMARY KEY, -- unique
                
                -- geometry
                lgth  REAL NON NULL,
                hgth  REAL NON NULL,

                -- material properties
                e REAL NON NULL,
                nu REAL NON NULL,
                rho REAL NON NULL
                )
        
        more comprehensive examples can be found in the child classes
        ifasha.simulationmanager.examples.sm_dynamicsimulation.py
        ifasha.simulationmanager.examples.sm_staticsimulation.py
        
        for more informations about sqlite3 visit
        https://www.sqlite.org/datatype3.html
        """    
        pass

    def check_arguments(self, argument):
        """check input arguments to assure database integrity
        to avoid attack to database 'drop table'
        """
        if 'DROP' or 'drop' or ';' in argument:
            self.conn.rollback()
            self.conn.close()
            print('\nERROR: Database Attack Attempted')
            raise RuntimeError

    def create_new_table(self, table_name, first_column, column_type):
        """Create table with 1 column and set it as PRIMARY KEY
        note that PRIMARY KEY column must consist of unique values!
        """
        self.c.execute('CREATE TABLE {tn} ({cn} {ct} PRIMARY KEY)'\
            .format(tn = table_name, cn = first_column, ct = column_type))
        self.conn.commit()

    # # Create Foreign key
    # def create_foreign_key(self, child_table, link_parent_id, parent_table, parent_id):
    #     self.c.execute('ALTER TABLE {ct} ADD CONSTRAINT fk --_{ct}_{pt}\
    #               FOREIGN KEY ({lpid})\
    #               REFERENCES {pt}({pid})'\
    #         .format(ct = child_table,pt=parent_table, lpid=link_parent_id, pid=parent_id))
    #     self.conn.commit()

    def create_new_table_foreign_key(self, table_name, first_column, column_type,\
        link_parent_id, parent_id_type, parent_table, parent_id):
        """
        A foreign key is needed to link the child table to the parent table
        where the parent id appears in the child table and needs to exist in
        the parent table as well!
        """
        self.c.execute('CREATE TABLE {tn} (\
            {cn} {ct} PRIMARY KEY,\
            {lpid} {pidt},\
            FOREIGN KEY({lpid}) REFERENCES {pt}({pid}))'\
            .format(tn=table_name, cn=first_column, ct=column_type, lpid=link_parent_id, \
                pidt=parent_id_type, pt=parent_table, pid=parent_id))
        self.conn.commit()


    def add_new_column(self, table_name, new_column, column_type):
        """Adding a new column without a row values

        Args:
            table_name (int)
            new_column (int)
            column_type (int)
        """
        self.c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct}"\
            .format(tn = table_name, cn = new_column, ct = column_type))
        self.conn.commit()

    def add_new_column_default(self, table_name, new_column, column_type, defaultval):
        """Adding a new column with a default row value
        """
        self.c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct} DEFAULT '{df}'"\
            .format(tn = table_name, cn = new_column, ct = column_type, df = defaultval))
        self.conn.commit()
    
    def create_unique_index(self, table_name, index_name, column):
        """Creating an unique index
        """
        self.c.execute('CREATE INDEX {ix} ON {tn}({c})'\
            .format(ix = index_name, tn = table_name, c = column))
        self.conn.commit()

    def drop_unique_index(self, index_name):
        """Dropping the unique index
        e.g., to avoid future conflicts with update/insert functions
        """
        self.c.execute('DROP INDEX {ix}'.format(ix = index_name))
        self.conn.commit()
    

    # Update & insert & find records:
    
    def insert_record(self, table, values, print_out=False):
        """Insert new record in specific table, values are passed as a dictionary or as list

        Args:
            table (str): as defined in definitions.py
            values (dict): keys are table columns (the way to go)
            values (list): order must be the same as column in table

        Returns:
            int: id e.g row number

        check if row already exist
        assign index id
        insert data in database
        """
        # get fields in table 
        if type(values) == dict:
            fields = self.get_fields(table)
            val = []      
            for f in fields:
                val.append(str(values[f]))
            values = val

        condition = self.assemble_condition_tbname(table, values)
        if self.get_total_nb_rows_condition(table, condition) == 0:
            self.is_new = True
            # assign index
            idx = int(self.get_total_nb_rows(table)) + 1
            if print_out:
                print('Entry in table {0} is unique: new id = {1}'.format(table, idx))
            values[0] = str(idx)
            # insert data
            self.insert_record_row(table, values)
            if print_out:
                print('Entry inserted!')
            self.get_entry(table, idx, print_out)
            return int(idx)
        else:
            self.is_new = False
            idx = self.get_unique_entry_id(table, condition)
            if print_out:
                print('Entry in table {0} already exist: id = {1}'.format(table, idx))
            return int(idx)

    def insert_record_row_string(self, table_name, values):
        """insert entire row input values in a single string
        """
        try:
            self.c.execute('INSERT INTO {tn} VALUES ({vs})'\
                .format(tn=table_name, vs=values))
            self.conn.commit()
        except sqlite3.IntegrityError:
            print('\nWARNING: ID already exists in PRIMARY KEY column {}'.format(id_column))


    def insert_record_row(self, table_name, values):
        """insert entire row input values as a list of strings
        """
        values = "'"+"', '".join(values)+"'"
        #print(values)
        #print(table_name)
        try:
            self.c.execute('INSERT INTO {tn} VALUES ({vs})'\
                .format(tn=table_name, vs=values))
            self.conn.commit()
        except sqlite3.IntegrityError:
            print('\nWARNING: ID already exists in {} PRIMARY KEY column id row = {}'\
                .format(table_name, values))


    def insert_empty_row(self, table_name, id):
        """ insert new empty row and assign id primary key
        """
        try:
            self.c.execute('INSERT INTO {tn} (id) VALUES ({vs})'\
                .format(tn=table_name, vs=id))
            self.conn.commit()
        except sqlite3.IntegrityError:
            print('\nWARNING: ID already exists in PRIMARY KEY column {}'.format(id_column))


    def update_record(self, table_name, primary_key, column, value, print_out=False):
        """Updates the value of a specific cell in a table

        Args:
            table_name (str): as defined in definitions.py
            primary_key (int): e.g. row nb
            column (str): key or field
            value (varies)

        """
        primary_key = int(primary_key)
        if print_out:
            print('Record to be updated:')
            self.get_entry(table_name, primary_key, print_out)
        self.c.execute('UPDATE {tn} SET {cn}=("{vs}") WHERE id={pk}'\
            .format(tn=table_name, cn=column, vs=value, pk=primary_key))
        self.conn.commit()
        if print_out:
            print('Updated record:')
            self.get_entry(table_name, primary_key, print_out)


    # DATE AND TIME - time_stamp is preferable than date + time
    
    def insert_row_date_time(self, table_name, id, date_col, time_col):
        """insert a new row with the current date and time

        e.g.

            2014-03-06, 16:26:37
        """
        try:
            self.c.execute("INSERT INTO {tn} (id, {d}, {t}) VALUES({vs}, DATE('now'), TIME('now'))"\
                .format(tn=table_name, d=date_col, t=time_col, vs=id))
            self.conn.commit()
        except sqlite3.IntegrityError:
            print('\nWARNING: ID already exists in PRIMARY KEY column {}'.format(id_column))

    def insert_row_time_stamp(self,table_name, id, date_time_col):
        """insert a new row with the current timestamp

        e.g. 

            2014-03-06 16:26:37
        """
        try:
            self.c.execute("INSERT INTO {tn} (id, {t}) VALUES({vs}, CURRENT_TIMESTAMP)"\
                .format(tn=table_name, t=date_time_col, vs=id))
            self.conn.commit()
        except sqlite3.IntegrityError:
            print('\nWARNING: ID already exists in PRIMARY KEY column {}'.format(id_column))

    def update_date_time(self, table_name, id, date_col, time_col):
        """update row for the new current date and time column

        e.g. 

            2014-03-06, 16:26:37
        """
        self.c.execute("UPDATE {tn} SET {d}=DATE('now') WHERE id={vs}"\
            .format(tn=table_name, d=date_col, t=time_col, vs=id))
        self.c.execute("UPDATE {tn} SET {t}=TIME('now') WHERE id={vs}"\
            .format(tn=table_name, d=date_col, t=time_col, vs=id))
        self.conn.commit()

    def update_time_stamp(self, table_name, id, date_time_col):
        """update row for the new current date and time column

        e.g.

            2014-03-06 16:26:37
        """
        self.c.execute("UPDATE {tn} SET {cn}=(CURRENT_TIMESTAMP) WHERE id={vs}"\
            .format(tn=table_name, vs=id, cn=date_time_col))
        self.conn.commit()
        print('Timestamp updated')


    # Accessors

    def get_fields(self, table_name, print_out=False):
        """returns column names / keys / fields in a list
        """
        self.c.execute('PRAGMA TABLE_INFO({})'\
            .format(table_name))
        names = [tup[1] for tup in self.c.fetchall()]
        if print_out:
            print('Columns in table: {tn} \n{cl}'\
                .format(tn=table_name,cl=names))
        if len(names) == 0:
            print('\nWARNING: No columns found!')
        return names

    def get_total_nb_rows(self, table_name, print_out=False):
        """Returns the total number of rows in the database 
        """
        self.c.execute('SELECT COUNT(*) FROM {}'\
            .format(table_name))
        count = self.c.fetchall()
        if print_out:
            print('Total rows: {}'.format(count[0][0]))
        return int(count[0][0])

    def get_table_col_info(self, table_name, print_out=False):
        """Returns a list of tuples with column informations

            (id, name, type, notnull, default_value, primary_key)
        """
        self.c.execute('PRAGMA TABLE_INFO({})'.format(table_name))
        info = self.c.fetchall()
        if print_out:
            print("Table {}: Column Info:\nID, Name, Type, NotNull, DefaultVal, PrimaryKey"\
                .format(table_name))
            for col in info:
                print(col)
        return info

    def get_number_entries(self, table_name, print_out=True):
        """Returns a dictionary with columns as keys and the number of not-null 
        entries as associated values.
        """
        self.c.execute('PRAGMA TABLE_INFO({})'.format(table_name))
        info = self.c.fetchall()
        col_dict = dict()
        for col in info:
            col_dict[col[1]] = 0
        for col in col_dict:
            self.c.execute('SELECT ({0}) FROM {1} WHERE {0} IS NOT NULL'\
                .format(col, table_name))
            # In my case this approach resulted in a better performance than using COUNT
            number_rows = len(self.c.fetchall())
            col_dict[col] = number_rows
        if print_out:
            print("Number of entries per column:")
            for i in col_dict.items():
                print('{}: {}'.format(i[0], i[1]))
        return col_dict

    # Utility
    def assemble_condition_dict(self, values_dict, print_out=False):
        """enter the dictionary with the values you want to construct the condition
        """
        fields = []
        values = []
        for k in values_dict.keys():
            fields.append(k)
            values.append(values_dict[k])
        condition = self.assemble(fields, values, print_out)
        return condition

    def assemble_condition_list(self, cond_list, print_out=False):
        """enter the dictionary with the values you want to construct the condition
        """
        cond_list = np.asarray(cond_list).T.tolist()
        try:
            fields,logic_op, values = cond_list
        except:
            fields   = []
            logic_op = []
            values   = []
        condition = self.assemble(fields, values, print_out,logic_operator=logic_op)
        return condition
        
    def assemble_condition_tbname(self, table_name, entire_row, print_out=False):
        """enter the entire row and put a ? for the values you do no want to consider
        """
        fields = self.get_fields(table_name)
        values = entire_row
        condition = self.assemble(fields,values, print_out)
        return condition

    def assemble_condition_fld_vl(self, fields_values):
        """enter a np.array with column 0 = fields and column 1 = values
        if empty or ? 
        

        > DO NOT USE: 

        > use dictionaries instead
        """
        fields = fields_values[:,0]
        values = fields_values[:,1]
        condition = self.assemble(fields,values)
        return condition
    
    def assemble(self, fields, values, print_out=False,logic_operator='=='):
        if print_out:
            print("fields")
            print(fields)
            print("values")
            print(values)
        cdt=[]
        if logic_operator=='==':
            logic_operator=['==' for i in range(len(fields))]
        for (f,v,lop) in zip(fields, values,logic_operator):
            if v != '?':
                cdt.append(f + lop +'"' + str(v) + '"' )
        condition = ' AND '.join(cdt)
        return condition    

    # Selecting entries

    def get_entry(self, table_name, primary_key, print_out=False):
        """Get entry as sqlite3.Row identified by primary key
        """
        primary_key = int(primary_key)
        if primary_key == None:
            print('\nERROR: No primary key provided: \nprimary_key = {}'.format(primary_key))
            raise RuntimeError
        if print_out:
            print('get entry in {} id = "{}"'.format(table_name,primary_key))
        self.c.execute('SELECT * FROM {tn} WHERE id={pk}'\
            .format(tn=table_name, pk=primary_key))
        row = self.c.fetchone()
        if print_out:
            print(row)
        if row['id'] != primary_key:
            print('No entry with id = {}'.format(primary_key))
        else:
            return row

    def get_entries_id_dict(self, table, values_dict, print_out=False):
        """Get entries id from specific table, which values correspond to values_dict.
        values_dict is a dictionary.

        Returns:
            array with the ids found
        """
        # get fields in table 
        nvaldct = dict() 
        fields = self.get_fields(table)
        values = []        
        for k in values_dict.keys():
            if k in fields:
                nvaldct[k] = values_dict[k]
        condition = self.assemble_condition_dict(nvaldct)
        return self.get_entries_id(table, condition)

    def get_entries_id_str(self, table, cond_str, print_out=False):
        """Get entries id from specific table, which values correspond to cond_str with C++ logic operator

        Returns:
            array with the ids found
        """            
        assert '||' not in cond_str
        fields = self.get_fields(table)
        cond_list = []

        cond_str = cond_str.replace('&&','AND')
        cond_str = cond_str.replace('and','AND')
        #cond_str = cond_str.replace('||','OR')

        logic_operators = ['==','!=','<','>']#,'<=','>=']

        for cond in cond_str.split('AND'):
            for lop in logic_operators:
                if lop in cond:
                    key, value = [ string.strip() for string in  cond.split(lop)]
                    if key in fields:
                        cond_list.append([key,lop,value])
                        print(key,lop,value)

        condition = self.assemble_condition_list(cond_list)

        return self.get_entries_id(table, condition)

    def get_entries_id(self, table_name, condition):
        """Return array with all entries_id (primary key) satisfying the condition
        """
        if condition != '':
            self.c.execute('SELECT (id) FROM {tn} WHERE {cd}'\
                .format(tn=table_name, cd=condition))
            entries_id = self.c.fetchall()
            n_entries_id = []            
            for i in entries_id:
                i = int(i[0])
                n_entries_id.append(i)
            return np.array(n_entries_id)
        else:
            return []#np.array(self.get_all_entries_id(table_name))
            
    def get_all_entries_id(self, table_name):
        """Return array with all entries_id (primary keys)
        """
        self.c.execute('SELECT (id) FROM {tn} '\
            .format(tn=table_name,))
        entries_id = self.c.fetchall()
        n_entries_id = []            
        for i in entries_id:
            i = int(i[0])
            n_entries_id.append(i)
        entries_id = np.array(n_entries_id)
        if len(entries_id)==0:
            print('\nWARNING: No entries found, table is empty!')
        else:
            return entries_id

    def get_unique_entry_id(self, table_name, condition, print_out=False):
        """Returns entry id for unique condition
        """
        self.c.execute('SELECT (id) FROM {tn} WHERE {cd}'\
            .format(tn=table_name, cd=condition))
        entries_id = np.array(self.c.fetchall())
        if len(entries_id)==1:
            return entries_id[0][0]
        elif len(entries_id)==0:
            if print_out:
                print('No entries found!')
            return False
        else:
            print('Condition non unique:\n {} entries found!'\
            .format(len(entries_id)))
            return False


    def get_total_nb_rows_condition(self, table_name, condition, print_out=False):
        """Returns the total number of rows in the database 
        """
        if condition == None:
            print('\nERROR: no condition given :(' )
            raise RuntimeError
        else:   
            #print('table_name',table_name,'\ncondition',condition)
            self.c.execute('SELECT COUNT(*) FROM {0} WHERE {1}'\
                .format(table_name, condition))
            count = self.c.fetchall()
            if print_out:
                print('Total rows: {}'.format(count[0][0]))
            return count[0][0]        

    # Get Date and Time
    
    def get_ids_between_time_stamps(self, table_name, date_time_col, start_time, end_time):
        """Returns all IDs of entries between 2 date_times as time stamp 
        
        Examples:

            2014-03-06 16:26:37
        """
        self.c.execute("SELECT id FROM {tn} WHERE {cn} BETWEEN '{s}' AND '{e}'"\
            .format(tn=table_name, cn=date_time_col, s=start_time, e=end_time))
        return np.array(self.c.fetchall())

    def get_ids_from_time(self, table_name, date_time_col, timestamp):
        """Returns all IDs of entries from specific date and time til now
        """
        self.c.execute("SELECT id FROM {tn} WHERE {cn} BETWEEN '{t}' AND CURRENT_TIMESTAMP"\
            .format(tn=table_name, cn=date_time_col, t=timestamp))
        return np.array(self.c.fetchall())

    def get_ids_last_days(self, table_name, date_time_col, nb_days):
        """Returns all IDs of entries of the last amount of time e.g. 1 day
        """
        self.c.execute("SELECT id FROM {tn} WHERE DATE('now') - {dc} <= {n}"\
            .format(tn=table_name, dc=date_time_col, n=nb_days))
        return np.array(self.c.fetchall())
