#!/usr/bin/env python

# test5-row-factory.py
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


project_name = 'test1'
wdir = '.'
memory = True # store db in ram only

# Create new database file
dbm = DatabaseManager(project_name, wdir, memory)

# Create new table 1st column is primary key

table_name = 'costumer'
first_column = 'id'
column_type = 'INTEGER'
dbm.create_new_table(table_name, first_column, column_type)

# Add many columns

columns = [
    'name',
    'address',
    'phone']
# storage classes: INTEGER, REAL, TEXT, NONE
# 
columns_type = [ 
    'TEXT NON NULL',
    'TEXT NON NULL',
    'REAL NON NULL']

for (c,ct) in zip(columns,columns_type):
    dbm.add_new_column(table_name, c, ct)


# Add 1 column with default value
dbm.add_new_column_default(table_name,'debt', 'REAL NON NULL', 0)


dbm.get_table_col_info(table_name,True)

values = [
    4,
    'Mike',
    '201 Cascadilla St',
    60723016,
    0]

dbm.insert_record_row(table_name,values)
dbm.insert_record_row_string(table_name, "1,'john', 'First St', 6079234, 0")
dbm.insert_record_row_string(table_name, "2,'jack', 'Wait Ave', 6945872, 0")
dbm.insert_record_row_string(table_name, "3,'john', 'Second St', 61244354, 34")

dbm.get_total_nb_rows(table_name, True)

# returns a dictionary with number of entries per row
r = dbm.get_number_entries(table_name, True)
print r['name']

# returns a sqlite3.Row object 
# which can be used in a very handy way !!!
primary_key = 1
r = dbm.get_entry(table_name, primary_key, True)
print 'type'
print type(r)
print 'r'
print r
print len(r)
print r[2]
print r.keys() # get columns name
print r['name']
for member in r:
    print member


print '\nColumn names'
c = dbm.get_fields(table_name, True)
print type(c)
print c

### get entries_id by condition: 'key >= "value"'
condition = 'name = "john"'
print dbm.get_entries_id(table_name, condition)

condition = 'name = "johnny" AND address = "Second St"'
print dbm.get_entries_id(table_name, condition)

condition = "name = 'jack' OR debt >= 0.1"
print dbm.get_entries_id(table_name, condition)


primary_key = 1
column = 'name'
value = 'gigi'

dbm.update_record(table_name, primary_key, column, value)

rr = dbm.get_entry(table_name, 3, True)
print rr

test = ('''
    id      = {id}
    name    = {name} 
    address = {address}
    '''.format(**rr))
print test