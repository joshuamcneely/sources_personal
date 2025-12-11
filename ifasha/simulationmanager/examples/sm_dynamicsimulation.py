#!/usr/bin/env python

# dynamicsimulation.py
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
from ifasha.solidmechanics import lefm
import ifasha.solidmechanics.definitions as smd
from sm_generatedynamic import generate_dynamic
from sm_staticsimulation import StaticSimulation
from ifasha.simulationmanager.definitions import Type

class DynamicSimulation(SimulationManager):

    def __init__(self, project_name, wdir='.', source_dir='.', memory=False):
        self.project_name = project_name
        self.type = Type()
        self.database_name = '{}_{}'.format(project_name, self.type.dynamic)
        SimulationManager.__init__(self, self.database_name, wdir, source_dir, memory) 

    def set_new_simulation(self, input_id, options_id, input_parameters):
        return {
        self.table.input :      input_id,
        self.table.analysis :   input_id,
        self.table.options :    options_id,
        'static_id' : input_parameters['static_id'],
        'status' :  self.status.initial,
        'supershear' :  -1,
        }


    def set_analysis_parameters(self, input_id, input_parameters):
        Lc = self.compute_Lc(input_parameters)
        S = lefm.compute_S(input_parameters)
        Xc0 = lefm.compute_Xc0(input_parameters,input_parameters)
        dx = (float(input_parameters['lgth']) / float(input_parameters['msh']))

        analysis_parameters = {
        'input' :       input_id, 
        'Lc' :          Lc,
        'Eratio' :      float(input_parameters['Ehet'])/float(input_parameters[smd.E]),
        'S' :           S,
        'ofhd_Lc' :     float(input_parameters['ofhystart']) / Lc,
        'ofht_Lc' :     (float(input_parameters['ofhytrans']) - float(input_parameters['ofhystart'])) / Lc,
        'ofhs_Lc' :     float(input_parameters['ofhxstart']) / Lc,
        'ofhl_Lc' :     (float(input_parameters['ofhxtrans']) - float(input_parameters['ofhxstart'])) / Lc,
        'lgth_Lc' :     (float(input_parameters['lgth']) / Lc),
        'Lc_dx' :       Lc / dx,
        'Xc0_dx' :      Xc0 / dx,
        }

        if 'rhohet' and 'nuhet' not in input_parameters.keys():
            analysis_parameters['cs_ratio'] = (float(input_parameters['Ehet'])/float(input_parameters[smd.E]))**0.5

        return analysis_parameters

    def generate(self, sim_id):
        generate_dynamic(self, sim_id)

    def compute_Lc(self,setup_prop):
        taui = float(setup_prop[smd.taui])
        if setup_prop['interface']=='lswfric':
            taur = float(setup_prop[smd.muk])*float(setup_prop[smd.sigi])
            Gamma = lefm.compute_Gamma(setup_prop)
        else:
            taur = float(setup_prop[smd.taur])
            Gamma = float(setup_prop[smd.Gamma])
        delta_tau = taui - taur
        Lc = lefm.compute_critHalfLength(Gamma, delta_tau, setup_prop)
        if setup_prop['ccrack'] == 'true':
            return Lc
        else:
            return Lc*1.12

    def get_relevant_parameters(self):
        rel_flds = self.get_fields(self.table.analysis)
        flds = ['id', 'status', 'supershear']
        for f in rel_flds:
            if f != 'id':
                flds.append(f)
        return flds

    def load_static_get_id(self, input_parameters):
        static_simulation = StaticSimulation(self.project_name, self.wdir, self.source_dir, self.memory)
        static_id = static_simulation.get_entries_id_dict(self.table.input, input_parameters, True)
        static_simulation.__del__()
        if len(static_id) == 0:
            print('\nERROR: Matching static simulation found!')
            return 1/0
        elif len(static_id) == 1:
            static_id = int(static_id)
            print('\nMatching static simulation found: {}_sts_{:0>4}'.format(self.project_name, static_id))
            input_parameters['static_id'] = static_id
        else:
            print('\nWARNING: more than one matching static simulations found:')
            for i in static_id:
                print('{}{0:0<4}'.format(self.database_name, int(i)))
        return input_parameters

    def insert_comment(self, sim_id, comment, print_out=False):
        self.update_record(self.table.simulation, sim_id, 'comment', comment, print_out)
        print('{}_{:0>4} Comment inserted: {} '.format(self.database_name, sim_id, comment))

    def is_supershear(self, sim_id, transition, print_out=False):
        param_dict = self.get_global_parameter_dict(sim_id)
        Lc = self.compute_Lc(param_dict)
        transition = transition/Lc
        self.update_record(self.table.simulation, sim_id, 'supershear', transition, print_out)
        print('{0}_{1:0>4} set supershear! \ntransition: {2} \n'.format(self.database_name, sim_id,transition))

    def is_not_supershear(self, sim_id, print_out=False):
        transition = -2
        self.update_record(self.table.simulation, sim_id, 'supershear', transition, print_out)
        print('{0}_{1:0>4} is not supershear! \ntransition: {2} \n'.format(self.database_name, sim_id,transition))

    def create_database(self):

        self.c.execute('''
            CREATE TABLE input (
            id INTEGER PRIMARY KEY, -- unique
            
            -- geometry
            dim   INTEGER NON NULL,
            lgth  REAL NON NULL,
            hgth  REAL NON NULL,
            wdth  REAL NON NULL,
            msh   REAL NON NULL,
            asym  TEXT NON NULL,
            ccrack TEXT NON NULL,
        
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

            -- 
            Ehet REAL,
    
            sigi REAL NON NULL,
            taui REAL NON NULL,

            -- interface properties
            interface TEXT NON NULL, 
            mus REAL,
            muk REAL,
            dc  REAL,
            Gamma REAL,
            tauc REAL,
            taur REAL,

            -- nucleation 
            nuchalfsize REAL NON NULL,
            nucspeed REAL NON NULL,
            nuccenter REAL NON NULL,
            nucwzone REAL NON NULL
                        
            )''')


        self.c.execute('''
            CREATE TABLE options (
            id INTEGER PRIMARY KEY, -- unique
            
            -- cluster setup time, nodes
            hrs INTEGER NON NULL,
            nb_nodes INTEGER NON NULL,

            -- paraview dump
            para_max INTEGER NON NULL,
            para_int INTEGER NON NULL,
            
            -- datamanager dump
            dump_heights TEXT,
            binary_shear_info INTEGER NON NULL,
            binary_normal_info INTEGER NON NULL,
            cohesive_normal INTEGER NON NULL,
            cohesive_shear INTEGER NON NULL,
            strength INTEGER NON NULL,
            shear_gap INTEGER NON NULL,
            absolute_shear_gap INTEGER NON NULL,
            normal_gap INTEGER NON NULL,
            shear_gap_rate INTEGER NON NULL,
            normal_gap_rate INTEGER NON NULL,
            full_field INTEGER NON NULL
            )
            ''')

        # obsolete dump names
        """ 
        is_in_contact INTEGER NON NULL,
        contact_pressure INTEGER NON NULL,
        is_sticking INTEGER NON NULL,
        frictional_strength INTEGER NON NULL,
        friction_traction INTEGER NON NULL,
        slip INTEGER NON NULL,
        cumulative_slip INTEGER NON NULL,
        slip_velocity INTEGER NON NULL,    
        velocity INTEGER NON NULL,    
        """
        
        self.c.execute('''
            CREATE TABLE simulation (
            id INTEGER PRIMARY KEY, -- unique
            
            static_id TEXT NON NULL, -- if static implicit
            input INTEGER NON NULL,
            analysis INTEGER NON NULL,
            options INTEGER NON NULL,

            initial_time TEXT,
            start_time TEXT,
            finish_time TEXT,
            status TEXT,
            rating INTEGER,
            comment TEXT,
            supershear REAL, -- location where transition occur

            -- foreign key set-up
            FOREIGN KEY(input) REFERENCES input(id),
            FOREIGN KEY(options) REFERENCES options(id)        
            )
            ''')

        self.c.execute('''
            CREATE TABLE analysis (
            id INTEGER PRIMARY KEY, -- unique
            input INTEGER NON NULL,
            
            Lc REAL NON NULL,
            Eratio REAL NON NULL,
            cs_ratio REAL NON NULL,
            S REAL NON NULL,
            ofhd_Lc REAL NON NULL, -- off fault heterogeneity fault distance divided by Lc
            ofht_Lc REAL NON NULL, -- off fault heterogeneity thickness divided by Lc
            ofhs_Lc REAL NON NULL, -- off fault heterogeneity start from left boundary 
            ofhl_Lc REAL NON NULL, -- off fault heterogeneity length
            lgth_Lc REAL NON NULL, -- Domain length dimensionless
            Lc_dx REAL NON NULL,   -- Lc div dx initial crack length definition
            Xc0_dx REAL NON NULL,  -- cohesive zone length definition

            -- foreign key set-up
            FOREIGN KEY(input) REFERENCES input(id)
            )
            ''')
        # Save (commit) the changes
        self.conn.commit()
        print('Database created succesfully!\nThis operation has to be done only once! :p')




