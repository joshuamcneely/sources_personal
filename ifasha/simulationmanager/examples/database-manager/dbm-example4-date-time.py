#!/usr/bin/env python

# test3-new-rows-get-rows.py
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

table_name = 'orders'
first_column = 'id'
column_type = 'INTEGER'
dbm.create_new_table(table_name, first_column, column_type)

# Add many columns

columns = [
    'name',
    'time_stamp',
    'date',
    'time',
    'curr_time_stamp']

# storage classes: INTEGER, REAL, NUMERIC TEXT, NONE

columns_type = [ 
    'TEXT NON NULL',
    'TIMESTAMP NON NULL',
    'TIMESTAMP NON NULL',
    'TIMESTAMP NON NULL',
    'TIMESTAMP NON NULL']

for (c,ct) in zip(columns,columns_type):
    dbm.add_new_column(table_name, c, ct)

cdate = columns[1] 
ctime = columns[2]
ctstamp = columns[3]
ctcurr = columns[4]

# row 1
dbm.insert_row_date_time(table_name, 1, cdate, ctime)
dbm.update_time_stamp(table_name,1,ctstamp)
dbm.update_record(table_name, 1, cdate, '2013-02-02')
dbm.update_record(table_name, 1, ctstamp, '2013-02-02 00:00:00')


# row 2
dbm.insert_row_time_stamp(table_name, 2, ctstamp)

dbm.update_date_time(table_name, 2, cdate, ctime)

for i in range(0,10**7): ## wait 2 sec
    i += i + 1.

dbm.update_time_stamp(table_name, 2, ctcurr)

for id in range(1,3):
    dbm.get_entry(table_name,id,True)

# time stamp e.g. 2014-03-06 16:26:37
ids = dbm.get_ids_between_time_stamps(table_name, ctstamp, '2014-03-06 16:26:37','2016-04-05 00:00:00')
print type(ids)
print 'from 2014 to now'
print ids

ids = dbm.get_ids_between_time_stamps(table_name, ctstamp, '2012-03-06','2016-04-05')
print 'from 2012 to now'
print ids

ids = dbm.get_ids_between_time_stamps(table_name, ctstamp, '2014-03-06','2016-04-05')
print 'from 2014 to now'
print ids

ids = dbm.get_ids_between_time_stamps(table_name, cdate, '2014-03-06','2016-04-05')
print 'from 2014 to now'
print ids

ids = dbm.get_ids_from_time(table_name, ctstamp, '2016-03-08')
print 'form yesterday to now'
print ids

ids = dbm.get_ids_last_days(table_name, ctstamp, '1')
print 'last 1 day'
print ids

ids = dbm.get_ids_last_days(table_name, cdate, '1')
print 'last 1 day'
print ids