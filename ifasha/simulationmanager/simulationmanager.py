#!/usr/bin/env python
"""
simulationmanager.py

Code to manage simulations input and outputs

There is no warranty for this code

@version 0.1
@author Gabriele Albertini <galberti819@gmail.com>
@date     2016/03/04
@modified 2016/03/04

"""
from __future__ import print_function
from __future__ import division

from ifasha.simulationmanager import DatabaseManager
from ifasha.simulationmanager.utilities import execute
from ifasha.simulationmanager.definitions import Status, Time, Table, Type

import numpy as np


class SimulationManager(DatabaseManager): 
    """The `SimulationManager` inherits from `DatabaseManager`.
    It will help you manage and organize a large number of simulations. 
    Easily have an overview over the simulations being, retrieve their input parameter
    and generate input files.
    """

    # Constructor and Destructor
        
    def __init__(self, database_name, wdir, source_dir='.', memory=False):
        """Creates a SimulationManager object:
        calls constructor of the parent class DatabaseManager
        """
        DatabaseManager.__init__(self, database_name, source_dir, memory)    

        self.database_name = database_name

        # import location of useful directories
        
        # where the input files are created: builds
        self.wdir = wdir    
        
        # where the database file my_database.sqlite is stored
        self.source_dir = source_dir    

        # names of the tables time and status of the database
        self.time   = Time()
        self.status = Status()
        self.table  = Table()
        self.type   = Type()

        self.test = [] # only for test purpose

        # get column = fields per each table
        
        self.input_fld    = self.get_fields(self.table.input)
        self.analysis_fld      = self.get_fields(self.table.analysis)
        self.opt_fld      = self.get_fields(self.table.options)
        self.info_fld     = self.get_fields(self.table.simulation)        

        if len(self.input_fld) == 0:
            print('\nWARNING: Database not configured: no tables present')
            print('Run create_database!')

    def __del__(self):
        """ destructor
        """
        pass

    # Methods
    
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
        help(self.create_database)
        raise RuntimeError
    
    def new_simulation(self, input_parameters, options, print_out=False):
        """This method inserts a new simulation in the database
        inputs and options are give as dictionaries
        
        Args:
            input_parameters (dict): model specific parameters
            options (dict): output, submission, and other options
            print_out (bool): if True prints informations

        Returns:
            int: sim_id if new simulation inserted
            list: old_sim_ids if simulation with same input exist  
            int: -1 else
        
        
        First `input`

            It is assessed whether the input_parameters are already present in the database.
            If the input_parameters are unique, a new row is inserted in the table input.
            Else, the user can choose to create it anyway (for instance with different options)
        
        Second `options`
        
            If the options are unique, a new row is inserted in the table options.
            If nor the input_parameters neither the options are unique, the user can 
            choose whether to create a new simulation or not.
        
        Third `simulation`
        
            A new simulation is added to the database.
            Each simulation has a corresponding input and options. 
            The link between tables is done by using foreign keys:
            The simulation table has the fields input_id and options_id which 
            reference to a specific row in the input and option tables. 
            The Simulation table contains further informations such as:
            status (initial, generated, submitted, running, ... )
            time        
            comment
            rating
            
        Extras

            For each entry in the input table there is an entry in the analysis table,
            here are stored useful dimensionless parameters and other post-process data
        """
        create_simulation = True
        self.test = []
        # check field coherence 
        self.check_field(self.input_fld, input_parameters)
        self.check_field(self.opt_fld, options)

        input_parameters = self.load_static_get_id(input_parameters)
        # in the case of dynamic simulation it looks for the corresponding static one

        # INPUT
        
        # inserts only if new / otherwise returns id
        input_id = self.insert_record(self.table.input, input_parameters, print_out)
        self.test.append(self.is_new)
        
        # inserts only if new
        analysis_parameters = self.set_analysis_parameters(input_id, input_parameters)
        self.check_field(self.analysis_fld, analysis_parameters)
        self.insert_record(self.table.analysis, analysis_parameters, print_out)

        old_sim_ids = self.get_sim_ids(input_id)

        if len(old_sim_ids) != 0:
            self.warning_same_input(old_sim_ids, input_id, options)
            create_simulation = self.user_input_new_simulation()
        
        # OPTIONS
        options_id = self.insert_record(self.table.options, options, print_out)
        self.test.append(self.is_new)

        if create_simulation == True:
            old_sim_ids = self.get_sim_ids(input_id, options_id)
            if len(old_sim_ids) != 0:
                print('\nWARNING: simulation with input = {s}, option = {o} already exist:'\
                      .format(s=input_id, o=options_id))
                for sim_id in old_sim_ids:
                    sim_id = int(sim_id)
                    status = self.get_status(sim_id, print_out)
                    print('sim_id = {d}{id} \nstatus = {st}'\
                        .format(d=self.database_name, id=sim_id,st=status))
                    if status == self.status.broken:
                        print('\nAlready present simulation status is BROKEN!')
                        print('It might be useful to rerun it!')
                create_simulation = self.user_input_new_simulation()
        
        # SIMULATION
        if create_simulation == True:
            simulation_info = self.set_new_simulation(input_id, options_id, input_parameters)
            self.check_field(self.info_fld, simulation_info)
            sim_id = self.insert_record(self.table.simulation, simulation_info, print_out)
            self.set_time(sim_id, self.time.initial)
            self.test.append(self.is_new)
            print('Good! Simulation {}_{:0>4} inserted successfully :)'\
                .format(self.database_name, sim_id))
            return sim_id
            
        if create_simulation == False:
            old_sim_ids = self.get_sim_ids(input_id, options_id)
            self.test.append(False)
            if len(old_sim_ids) == 1:
                return int(old_sim_ids)
            else:
                return -1
            
    def user_input_new_simulation(self):
        """This method ask the user whether he wants to create a new simulation.
        """
        return self.user_input('Insert new simulation anyway?')

    def user_input(self, message):
        """This method ask the user whether he wants to continue.
        Returns a boolean
        """
        answer = raw_input( message + ' yes [y], no [n], break [B]')
        if answer == 'n':
            return False
        elif answer == 'y':
            return True
        elif answer == 'B':
            raise RuntimeError
        else:
            print('I did not understand... try again ')
            self.user_input(message)

    def warning_same_input(self, old_sim_ids, input_id, options):
        """Method printing a warning when a simulation with same input already exist:
        prints also a comparison between the options between the current simulation and
        the one already present in the database.
        """
        print('\nWARNING: simulation with input_id = {} already exist for simulations:'.format(input_id))  
        print(old_sim_ids)
        if raw_input('You might want to rerun it with different options \nshow old options yes/no,[y]/[N]')=='y':
            for sim_id in old_sim_ids:
                sim_id = int(sim_id)
                print('{:_>60}'.format('\n'))
                print('sim_id : ',sim_id)
                old_options = self.get_parameter_dict(sim_id, self.table.options)
                self.print_compare(old_options, options, self.table.options, title1=sim_id, title2='new')

    def set_new_simulation(self, input_id, options_id, input_parameters):
        """Virtual method! Implemented in child class!
        The fields of the simulation table are project and simulation type (static / dynamic / ...) specific. Therefore, 
        This method has to be implemented in the child class.
        
        Examples:     

            simulation_info = {
                self.table.input :      input_id,
                self.table.analysis :   input_id,
                self.table.options :    options_id,
                "status" :  self.status.initial
                }
            return simulation_info

        more comprehensive examples can be found in the child classes
        ifasha.simulationmanager.examples.sm_dynamicsimulation.py
        ifasha.simulationmanager.examples.sm_staticsimulation.py
        """
        help(self.create_database)
        raise RuntimeError

    def set_analysis_parameters(self, sim_id, input_parameters):
        """Virtual method! Implemented in child class!
        The fields of the analysis table are project and simulation type (static / dynamic / ...) specific. Threfore, 
        This method has to be implemented in the child class.

        analysis parameters could be dimensionless parameters describing the set-up or results of
        the post-processing
        
        e.g.

            analysis_parameters = {
                "input" :      input_id, 
                "Eratio" :     E_1 / E_0,
                "something_interesting_I_observed" : "to_be_filled",
                }
            return analysis_parameters
        """
        pass

    def load_static_get_id(self, parameter_dict):
        """This is an abstract method, which is defined in the child class
        It aims to find the corresponding static solution for a given dynamic simulation
        Do nothing in case of static simulation
        """
        return parameter_dict  

    def check_field(self, ref_fields, input_dict):
        """Checks if ref_fields fetched from the database corresponds with fields of the input dictionary
        """
        for ref_key in ref_fields:
            # add field 'id' if not present: it's supposed that the user does not specify the id
            # if self assigned!
            if ref_key == 'id':
                input_dict[ref_key] = '?'
            if ref_fields == self.info_fld:
            # add missing fields in case of simulation info table
                if not input_dict.has_key(ref_key):
                    input_dict[ref_key] = '?'
            if not input_dict.has_key(ref_key):
                msg=r'field "{0}" not present in input dictionary:{1}'.format(ref_key, input_dict.keys())
                raise RuntimeError(msg)

    def generate(self, sim_id):
        """Abstract method, which is defined in the child class

        This method generates the input files for a simulation with a specific sim_id
        For each simulation (sim_id) there is a specific input_id and option_id
        With this information options and input_parameters dictionaries can be extracted
        from the database and the input file can be generated.

        for more information about how to construct a generate function see
        ifasha.simulationmanager.examples.sm_generatedynamic
        ifasha.simulationmanager.examples.sm_generatestatic

        """
        help(self.generate)
        raise RuntimeError

    def check_generate(self, sim_id):
        """checks whether the input file has already been generated
        """

        if sim_id == -1:
            return False
        else:
            current_status = self.get_status(sim_id)
            if current_status == self.status.initial:
                return True
            else:
                print('\nWARNING: current status {0}\nyou already generate this input file: {1}_{2:0>4}'\
                      .format(current_status, self.database_name, sim_id))
                return False

    def launch_simulation_check(self, sim_id):
        """checks whether the input file has already been generated
        """
        current_status = self.get_status(sim_id)
        if current_status == self.status.generated:
            return True
        elif current_status == self.status.broken:
            return True
        else:
            print('\nWARNING: current status {0}\nyou already submitted this simulation: {1}_{2:0>4}'\
                .format(current_status, self.database_name, sim_id))
            return False

    def broken(self, sim_id, print_out=False):
        """Set current simulation status to `broken`
        use this option when the simulation did not run properly and you would like to rerun it
        """
        self.update_status_forced(sim_id, self.status.broken, print_out)
        print('{}{:0>4} Status update: {}'.format(self.database_name, sim_id, self.status.broken))

    def update_status_forced(self, sim_id, newstatus, print_out=False):
        """Updates simulation's status without checking for coherence
        """
        self.update_record_in_table(self.table.simulation, sim_id, self.status.status, newstatus, print_out)
        if print_out:
            print('{}{:0>4} Status update: {}'.format(self.database_name, sim_id, newstatus))

    def update_status(self, sim_id, newstatus, print_out=False):
        """Updates simulation's status without checking for coherence
        """
        if self.status_check(sim_id, newstatus):
            self.update_record_in_table(self.table.simulation, sim_id, self.status.status, newstatus, print_out)
            if print_out:
                print('{}{:0>4} Status update: {}'.format(self.database_name, sim_id, newstatus))
        
    def status_check(self, sim_id, newstatus):
        """Check that no status is jumped
        
        Args:
            newstatus (str): defined in definitions.py
            sim_id (int)
        
        Returns:
            True if no jump

        The different statuses are defined in the module definition
        Do not use the string directly, but the variable containing the string

        When dealing with a large number of simulation knowing the real status of each one can be challenging.
        Being able to access the status of each simulation easily can be really time-saving.
        Therefore it is important to consequently update the status each time an action is performed.

        The aim of this method is to avoid status jumps!

        e.g 
        This i a status jump:

            generated -> running  

        The normal flow of events would need: 

            generated -> submitted -> running

        To prevent status jumps, each status is assigned an integer. 
        This integer is the status' order e.g. ``order['initial'] = 0``

        When a status jump is attempted, a warning is raised. The user is now aware that something 
        unexpected happened with the flow of events.

        The normal flow is characterised by the following statuses:

            initial -> generated -> submitted -> running -> finished -> postprocessed

        when something goes wrong the status broken can be used
        """
        oldstatus = self.get_status(sim_id)
        oldstatus_order = self.status.order[oldstatus]
        newstatus_order = self.status.order[newstatus]

        if (oldstatus_order + 1) == newstatus_order:
            return True
        elif oldstatus == self.status.broken:
            return True
        else:
            print('\nWARNING: status jump')
            print('{}: {}'.format(oldstatus,oldstatus_order))
            print('{}: {}'.format(newstatus,newstatus_order))
            # return self.user_input('Continue anyway?')

    def set_time(self, sim_id, time_type):
        """set time when a specific event happens
        
        The time_stamp is inserted in the table Simulation

        `time_type` refers to the field (column) where the timestamp has to be saved

        It can be useful for instance to save the time when the simulation is launched and when it 
        finishes
        """
        self.update_time_stamp(self.table.simulation, sim_id, time_type)

    def update_info(self, sim_id, field, entry):
        """Updates info is table Simulation

        It can be used to add a comment or similar

        e.g.

            field             entry
            _________________________________________________________

            rating            1 boring - 3 interesting -5 awesome
            supershear        1 yes - 0 no
            comment           OK
        """
        self.update_record_in_table(self.table.simulation, sim_id, field, entry)
    
    def update_record_in_table(self, table, sim_id, fld, new_record, print_out=False):
        """Updates record in given table
        first gets the sub_id given table (not necessarily the same as the sim_id)
        
        Note: only option and simulation table can be changed

        raises:
            Warning: if table=option this will affect all simulations with this same sub_id
            Error: tables input and analysis cannot be changed
            Error: field which are keys for entries in other tables cannot be changed
        """
        sub_id = self.get_id(sim_id,table)
        if table==self.table.options:
            print('WARNING: affects all simulations w/ option id = {}'.format(sub_id))
        if table in [self.table.input,self.table.analysis,]:
            print('ERROR: you cannot change table: {}\n create new simulation instead!'.format(table))
            raise RuntimeError
        if fld in ['id','static_id',self.table.input,self.table.analysis,self.table.options]:
            print('ERROR: your change would affect databse integrity')
            raise RuntimeError
        self.update_record(table,sub_id,fld,new_record)

    # ACCESSORS

    def get_status(self, sim_id, print_out=False):
        """Returns status of a specific simulation
        """
        entry = self.get_entry(self.table.simulation, sim_id, print_out)
        return entry[self.status.status] 
    
    def get_id(self, sim_id, table):
        """Returns the sub-table id corresponding to a simulation.

        The idea (see example below) is that the simulation table (my_favorite_song) connects all the other tables which may be called sub-tables
        A sub-table (artist) might be the one which stores the input_parameters or options
        From the simultaion_id (song_id), which is unique for every simulation the id of its input_parameters, options etc can be 
        retrieved with this function.
        These ids are needed to access the data stored in the sub_tables.

        e.g.

        table `my_favorite_song`

            _______________________________________
            
            song_id     title               artist
            _______________________________________  
            1           'Chihuahua'         1
            2           'Pirates of Dance'  1
            3           'Ma cherie'         2


        table `artist`  

            _______________________________________
            
            artist_id      name            
            _______________________________________  
            1           'DJ Bobo'
            2           'DJ Antoine'

        """

        simulation_info = self.get_entry(self.table.simulation, int(sim_id))
        if table == self.table.simulation:
            return int(sim_id)
        else:
            return int(simulation_info[table])

    def get_sim_ids(self, input_id='?', options_id='?',analysis_id='?'):
        """Returns all simulation id having a specific input_parameters/options id
        
        Args:
            input_id (int)
            options_id (int)

        Returns:
            sim_ids (list) of (int)
            if no arguments: all sim_ids are returned
        """
        fields_values = np.array([
            [self.table.input , input_id],
            [self.table.options, options_id],
            [self.table.analysis, analysis_id]])
        condition = self.assemble_condition_fld_vl(fields_values)

        if condition == '' or condition == None:
            # get all sim_ids in database
            sim_ids = self.get_all_entries_id(self.table.simulation)
        else: 
            # get all sim-ids condition
            sim_ids = self.get_entries_id(self.table.simulation, condition)
        return sim_ids

    def get_sim_ids_condition(self, cond):
        """Retrieves all sim_ids which satisfy the condition specified in 
        
        if (cond) = dict()
        the condition implemented is equality ==
        
        e.g.

            cond_dict = {
                'S' : 2.0,
                'd' : true,
                ...} 
        
        if type(cond) = string()
        e.g.

        cond_str = "S >= 2.0 && d == true"
        C++/python logical operators && == > < != 
        || not implemented
        """

        if type(cond)==dict:
            cond_dict = cond
            opt_ids = self.get_entries_id_dict(self.table.options, cond_dict)
            input_ids = self.get_entries_id_dict(self.table.input, cond_dict)
            analysis_ids = self.get_entries_id_dict(self.table.analysis, cond_dict)
            
        elif type(cond)==str:
            cond_str = cond
            opt_ids = self.get_entries_id_str(self.table.options, cond_str)
            input_ids = self.get_entries_id_str(self.table.input, cond_str)
            analysis_ids = self.get_entries_id_str(self.table.analysis, cond_str)
        else:
            raise RuntimeError(self.get_sim_ids_condition.__doc__)

        input_ids = list(input_ids)

        sim_ids = []


        if opt_ids==[]: opt_ids=['?']
        if input_ids==[]: input_ids=['?']
        if analysis_ids==[]: analysis_ids=['?']
        
        if True:
            for i in input_ids:
                for o in opt_ids:
                    for a in analysis_ids:
                        ids = self.get_sim_ids(i, o, a)
                        for sim_id in ids:
                            sim_ids.append(sim_id)
        
        return sim_ids

    def get_all_sim_id(self):
        """Retrieves all simulation ids present in the database
        """
        return self.get_sim_ids()

    def get_sim_id_last_days(self, nb_days):
        """Retrieves all simulations inserted in the database during specified last number of days
        """
        return self.get_ids_last_days(self.table.simulation, self.time.initial, nb_days)


    # Access dictionaries: 

    def get_global_parameter_dict(self, sim_id):
        """Retrieves a dictionary with the complete information relative to a specific simulation
        this information is spread between different tables of the database  
        """
        i = self.get_parameter_dict(sim_id, self.table.input)
        r = self.get_parameter_dict(sim_id, self.table.analysis)
        o = self.get_parameter_dict(sim_id, self.table.options)
        s = self.get_parameter_dict(sim_id, self.table.simulation)

        # combine dicts
        parameter_dict = i.copy()
        parameter_dict.update(r)
        parameter_dict.update(o)
        parameter_dict.update(s)

        if parameter_dict['id'] != sim_id:
            print('\nERROR: id does not match provided simulation id')
            print('database id {}'.format(parameter_dict['id']))
            print('provided id {}'.format(sim_id))
            raise RuntimeError
        return parameter_dict
    
    def get_parameter_dict(self, sim_id, table):
        """Retrieves the information stored in a specific table concerning a specific simulation
        """
        id = self.get_id(sim_id, table)
        return dict(self.get_entry(table, id))

    # Display database content
    # these functions are very handy to access content of the database directly from the command line

    def print_parameter_dict(self, parameter_dict, fields):
        """Displays the information provided in parameter_dict as follows

        e.g.

            fields      parameter
            __________________________

            field1      p1  
            field2      p2
            ...

        """
        print('{:_>48}'.format('\n'))
        for fld in fields:
            if fld != 'id':
                print('{0:<24}{1:<20}'.format(fld,parameter_dict[fld]))

    def print_all_parameter_dict(self, sim_id):
        """Displays the complete information concerning a specific simulation
        """
        print('\nSimulation {}{:0>4}'.format(self.database_name, sim_id))
        
        for table in self.table.all:
            fields = self.get_fields(table)
            parameter_dict = self.get_parameter_dict(sim_id, table)
            print('{:_>48}'.format('\n'))
            print('\n\n{}'.format(table))
            self.print_parameter_dict(parameter_dict, fields)

    def print_compare(self, dict1, dict2=None, table=None,title1='dict1', title2='dict2'):
        """Displays the content of two dictionaries. If one dictionary does not have a specific key
        it will be displayed as '-'
        
        If a table is specified:
        Displays the content of two dictionaries relative to the same table (i.e. with the same keys)
        
        e.g.
        
            fields      dict1        dict2
            ________________________________

            field1      p11          p21
            field2      p12          p22
            field3      p13          -
            ...
        """
        if table == None:
            fields1 = dict1.keys()
            fields2 = dict2.keys()
            for f in fields2:
                if f not in fields1:
                    dict1[f] = '-'
            for f in fields1:
                if f not in fields2:
                    dict2[f] = '-'
            fields = dict1.keys()
        else:
            fields = self.get_fields(table)
        
        print('\n{0:<24}{1:<20}{2:<20}'.format('field', title1, title2))
        print('{:_>60}'.format('\n'))
        for fld in fields:
            if fld =='id':
                continue
            print('{0:<24}{1:<20}{2:<20}'.format(fld, dict1[fld], dict2[fld]))

    def print_database(self, flds = None, **kwargs):
        """For all simulations present in the database displays the specified fields
        
        Args:
            - fields (list): fields to display for each simulation
            - column_width (int): with of each column in the displayed table

        This function is very useful to quickly check from the command line the status of each
        simulation.

        If no fields are specified the fields in get_relevant_flds will be displayed.

        e.g.

        ``fields = ['id', 'status', 'comment', 'E1/E2']``

        ``column_width = 12``

            id  status      comment     E1/E2
            _______________________________________

            1   postprocess OK          0.9
            2   running     -           1.0
            3   ...

        """
        
        if flds == None:
            flds = self.get_relevant_parameters()

        # default values
        column_width = 9
        max_lines = 50

        sim_ids = self.get_all_sim_id()

        try:
            column_width = kwargs['column_width']
            max_lines = kwargs['max_lines']
            slim_flds = kwargs['slim_flds']
            sim_ids = kwargs['sim_ids']
        except:
            print('non valid argument ', kwargs, 'default values are used')
            pass
                    

        parameter_dict = self.get_global_parameter_dict(1)        


        column_layout, column_title_layout,width_layout = self.get_layout(parameter_dict,flds,slim_flds,column_width) 


        print('\n{} informations'.format(self.table.analysis))
        print(width_layout.format('\n'))
        print(column_title_layout.format(*flds))
        print(width_layout.format('\n'))
        print_line = True
        
        counter=0
        for sim_id in sim_ids:
            counter+=1
            if not print_line:
                continue
            sim_id = int(sim_id)
            parameter_dict = self.get_global_parameter_dict(sim_id)
            
            try:    
                print(column_layout.format(**parameter_dict))
            except:
                column_layout = self.get_layout(parameter_dict,flds,slim_flds,column_width)[0]
                print(column_layout.format(**parameter_dict))
            if counter == max_lines:
                counter=0
                answer = raw_input()#'press RETURN to continue')
                if answer == 'q':
                    print_line = False
                else:
                    print(width_layout.format('\n'))
                    print(column_title_layout.format(*flds))
                    print(width_layout.format('\n'))
                    pass

    def get_layout(self, parameter_dict,flds,slim_flds,column_width):
        column_title_layout = ''
        column_layout = ''
        width = 0

        for f in flds:
            if f in['id','static_id'] or f in slim_flds:
                c_width = 5
            elif f[-4:] == 'time':
                c_width = 20
            else:
                c_width = column_width
            width += c_width+1#gap
          
            if type(parameter_dict[f]) == unicode:
                column_layout += '{' + f + ':<'+str(c_width)+'.'+str(c_width-1)+'} '
                column_title_layout += '{:<'+str(c_width)+'.'+str(c_width-1)+'} '
            elif type(parameter_dict[f]) == float:
                column_layout += '{' + f + ':>'+str(c_width)+'.3} '
                column_title_layout +=    '{:>'+str(c_width)+'.'+str(c_width-1)+'} '
            else:# integers
                c_width=5
                column_layout += '{' + f + ':<'+str(c_width)+'} '
                column_title_layout += '{:<'+str(c_width)+'.'+str(c_width-1)+'} '
        width_layout = '{:_>'+str(width)+'}'
        return column_layout, column_title_layout,width_layout


    def get_relevant_parameters(self):
        """Retrieves fields which are supposed to be relevant 
        When the method print_database is called the default option displays these fields
        """
        flds = ['id', 'status', 'comment']
        rel_flds = self.get_fields(self.table.analysis)        
        for f in rel_flds:
            if f != 'id':
                flds.append(f)
        return flds
    
    def print_table(self, table, sim_id, c_width=10):
        """Displays the content of one row of a specific table
        """
        parameter_dict = self.get_parameter_dict(sim_id, table)
        flds = self.get_fields(table)
        column_title_layout = ''
        column_layout = ''
        is_floating_number = False
        width = 0
        for f in flds:
            width += c_width
            column_title_layout += '{:<'+str(c_width)+'}'
            if is_floating_number:
                column_layout += '{' + f + ':<'+str(c_width)+'.3f}'
            else:
                column_layout += '{' + f + ':<'+str(c_width)+'}'
            if f == 'msh':
                is_floating_number = True
                # after this only floating numbers
        width_layout = '{:_>'+str(width)+'}'
        print(column_title_layout.format(*flds))
        print(width_layout.format('\n'))
        print(column_layout.format(**parameter_dict))

