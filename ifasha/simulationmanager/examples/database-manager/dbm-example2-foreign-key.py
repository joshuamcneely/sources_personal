#!/usr/bin/env python

# test2-foreign-key.py
#
# Code to manage simulations input and outputs
#
# There is no warranty for this code
#
# @version 0.1
# @author Gabriele Albertini <galberti819@gmail.com>
# @date     2016/03/06
# @modified 2016/03/06


from ifasha.simulationmanager import DatabaseManager


project_name = 'test2'
wdir = '.'
memory = True # store db in ram only

### Create new database file
dbm = DatabaseManager(project_name, wdir, memory)

### Create new table 1st column is primary key

table_name = 'costumer'
first_column = 'id'
column_type = 'INTEGER'
dbm.create_new_table(table_name, first_column, column_type)

### Add many columns

columns = [
    'name',
    'address',
    'phone']

### storage classes: INTEGER, REAL, TEXT, NONE

columns_type = [ 
    'TEXT NON NULL',
    'TEXT NON NULL',
    'REAL NON NULL']

for (c,ct) in zip(columns,columns_type):
    dbm.add_new_column(table_name, c, ct)

dbm.get_table_col_info(table_name, True)


### Add new table with FOREIGN KEY

table_name = 'orders'
first_column = 'id'
column_type = 'INTEGER'
link_parent_id = 'customer_id'
parent_id_type = 'INTEGER'
parent_table = 'costumer'
parent_id = 'id'
dbm.create_new_table_foreign_key(table_name, first_column, column_type, \
    link_parent_id, parent_id_type, parent_table, parent_id)


### Add many columns

columns = [
    'sum']

columns_type = [ 
    'REAL NON NULL']
for (c,ct) in zip(columns,columns_type):
    dbm.add_new_column(table_name, c, ct)

dbm.get_table_col_info(table_name, True)
