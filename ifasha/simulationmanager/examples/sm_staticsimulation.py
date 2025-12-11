#!/usr/bin/env python

# staticsimulation.py
#
# Code to manage simulations input and outputs
#
# There is no warranty for this code
#
# @version 0.1
# @author Gabriele Albertini <galberti819@gmail.com>
# @date     2016/03/04
# @modified 2016/03/04

from ifasha.simulationmanager import SimulationManager

from sm_generatestatic import generate_static
import ifasha.solidmechanics.definitions as smd
from ifasha.simulationmanager.definitions import Type

class StaticSimulation(SimulationManager):

    def __init__(self, project_name, wdir='.', source_dir='.', memory=False):
        self.project_name = project_name
        self.type = Type()
        self.type.mine = self.type.static
        self.database_name = '{}_{}'.format(project_name, self.type.static)
        SimulationManager.__init__(self, self.database_name, wdir, source_dir, memory) 

    def set_new_simulation(self, input_id, options_id, foo):
        return {
        self.table.input :      input_id,
        self.table.analysis :   input_id,
        self.table.options :    options_id,
        "status" :  self.status.initial,
        }
    
    def set_analysis_parameters(self, input_id, input_parameters):

        analysis_parameters = {
        "input" :      input_id, 
        "Eratio" :      float(input_parameters['Ehet']) / float(input_parameters[smd.E]),
        "ofhd" :     0.5*(float(input_parameters['ofhystart']) + float(input_parameters['ofhytrans'])),
        "ofht" :     (float(input_parameters['ofhytrans']) - float(input_parameters['ofhystart'])),
        "ofhs" :     float(input_parameters['ofhxstart']),
        "ofhl" :     (float(input_parameters['ofhxtrans']) - float(input_parameters['ofhxstart'])),
        "lgth" :     (float(input_parameters['lgth'])),
        "dx" :       (float(input_parameters['lgth']) / float(input_parameters['msh'])),
        }

        if 'rhohet' and 'nuhet' not in input_parameters.keys():
            analysis_parameters['cs_ratio'] = (float(input_parameters['Ehet'])/float(input_parameters[smd.E]))**0.5
        return analysis_parameters

    def warning_same_input(self, old_sim_ids, input_id, options):
        print('\nWARNING: Static simulation must have a unique input\nNo simulation added... \nChange the options manually in the .sub file...')
        
    def change_options(self, sim_id, new_options, print_out=False):
        print('\nChange options:')
        print('Simulation id: {}\nStatus: {}'.format(sim_id, self.get_status(sim_id)))
        old_options = self.get_parameter_dict(self.table.options, sim_id)
        self.print_compare(old_options, new_options, self.table.options)
        change_options = self.unser_input_options()
        if change_options:
            opt_id = self.get_id(sim_id, self.table.options)
            for fld in self.opt_fld:
                if fld != 'id' and new_options[fld]:
                    value = new_options[fld]
                    print fld, value
                    self.update_record(self.table.options, opt_id, fld, value, print_out)
            self.update_status_forced(sim_id, self.status.initial, print_out)
            print('Options updated! You will need to regenerate static input\n')
    
    def user_input_new_simulation(self):
        return False

    def unser_input_options(self):
        answer = raw_input('\nChange options? yes (y), no (n)\n')
        if answer == 'n':
            print('\nNo options changed... \n')
            return False
        elif answer == 'y':
            return True
        else:
            print('I did not understand... try again ')
            self.unser_input_options()

    def generate(self, sim_id):
        generate_static(self, sim_id)

    def create_database(self):

        self.c.execute('''
            CREATE TABLE input (
            id INTEGER PRIMARY KEY, -- unique
            
            -- geometry
            dim INTEGER NON NULL, 
            lgth  REAL NON NULL,
            hgth  REAL NON NULL,
            wdth  REAL NON NULL,
            msh   REAL NON NULL,
            asym TEXT NON NULL,

            -- off fault heterogeneity geometry
            ofh TEXT NON NULL,
            ofhxstart REAL,
            ofhxtrans REAL,
            ofhystart REAL,
            ofhytrans REAL,

            -- material properties
            E REAL NON NULL,
            nu REAL NON NULL,
            rho REAL NON NULL,
            psss INTEGER NON NULL, --plane stress

            Ehet REAL,

            sigi REAL NON NULL,
            taui REAL NON NULL

            )''')


        self.c.execute('''
            CREATE TABLE options (
            id INTEGER PRIMARY KEY, -- unique
            
            -- cluster setup time, nodes
            auto INTEGER
            -- hrs INTEGER NON NULL,
            -- nb_nodes INTEGER NON NULL
                        
            )
            ''')


        self.c.execute('''
            CREATE TABLE simulation (
            id INTEGER PRIMARY KEY, -- unique
            
            input INTEGER NON NULL,
            analysis INTEGER NON NULL,
            options INTEGER NON NULL,

            initial_time TEXT,
            start_time TEXT,
            finish_time TEXT,
            status TEXT,
            rating INTEGER,
            comment TEXT,

            -- foreign key set-up
            FOREIGN KEY(input) REFERENCES input(id),
            FOREIGN KEY(options) REFERENCES options(id)        
            )
            ''')

        self.c.execute('''
            CREATE TABLE analysis (
            id INTEGER PRIMARY KEY, -- unique
            input INTEGER NON NULL,

            Eratio REAL NON NULL,
            cs_ratio REAL NON NULL,
            ofhd REAL NON NULL, -- off fault heterogeneity fault distance 
            ofht REAL NON NULL, -- off fault heterogeneity thickness 
            ofhs REAL NON NULL, -- off fault heterogeneity start from left boundary 
            ofhl REAL NON NULL, -- off fault heterogeneity length
            lgth REAL NON NULL, -- Domain length 
            dx REAL NON NULL,   -- Lc div dx initial crack length definition
            
            -- foreign key set-up
            FOREIGN KEY(input) REFERENCES input(id)
            )
            ''')

        # Save (commit) the changes
        self.conn.commit()
        print('Database created succesfully!\nThis operation has to be done only once! :p')
