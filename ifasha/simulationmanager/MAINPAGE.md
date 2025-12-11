## SimulationManager

The SimulationManager aims to help you organize a large number of simulations. The inputs 
parameters, options, and eventually also analysis parameters are stored into a database file.

The informations stored in the database can be quickly accessed in order to perform useful 
tasks, such as generating output files, tracking the status of a new simulation (`generated
-> submitted -> running` etc.), accessing input parameters for data analysis and storing key 
analysis parameters.

A database allows to quickly identify a simulation thanks to its `simulation_id`, which in 
database jargon is a primary key (i.e. a unique index that makes a table row unique,
it is used to identify an row). This makes it possible to avoid having very large file 
names containing the information of a simulation. 

More, if there is a dependence between different kind of simulations (e.g. an implicit
simulation solving a static problem and an explicit dynamic simulation, which needs the 
equilibrium condition as initial condition) it can be stored in the database. For instance
the dynamic simulation in the example above will have an attribute static_id, which enables
to identify the corresponding initial condition.


## DatabaseArchitecture

In this section, the architecture of this database is presented. First, the general 
concept and interdependence between tables is explained. Second, each table is shortly
presented, including an example.

This database contains 4 different tables: `simulation, input, options, analysis`.

Instead of having one single huge table containing the simulation-info (id, status, ...)
all the input parameters (time step, E_modulus, density, ...) and all the options 
(number of cores, max. computation time, ...), the simulations table contains an 
input_id and options_id which reference to a specific row in the input-table and 
options-table.

This is implemented by using foreign keys. In a relational databases, a foreign 
key is a field in one table (e.g. `simulation` table) that uniquely identifies a 
row of another table (e.g. `input` table). That means that there is no direct 
hierarchy between tables, but references.

The fact of having many small tables instead of consolidating the data in a single 
big table not only has the advantage that a small table is more easy to handle
(e.g. display its content) and avoids repetitions in storing data. It is very useful
for selecting and managing data. For instance, options such as the number of cores
or the time-step of dumping the outputs should not affect the result of a simulation. 
On the other hand, input parameters such as material properties and geometry are 
directly inherent to the model. By separating this information it is easily identified
whether two simulation have the same input-parameters or not, and if they were ran 
with different options. This should also help the analysis process.


### `simulation` Table

The simulation table is the master table of the database. It contains general 
information about a simulation, such as the simulation_id (which is the primary
key), input_id, options_id, status etc. It's role is to redirect to the informations
present in the other sub-tables of the database. For instance, the column input 
references the column id in table input.

e.g.

After inserting your first few simulations in the database your simulation-table 
could look very similar to this:

    simulation 
    ___________________________________________________________________________

    id  input  options  analysis  (static_id)  start_time  status   comment 
    ___________________________________________________________________________

    1   1      1        1         1            2016/05/04  running  -        
    2   2*     1**      2         1            2016/05/04  launched -        



### `input` Table

The `input` table contains the parameters describing the model being simulated. These 
are directly linked with the physics being reproduced (material properties, geometry,
loading, mesh, time-step, etc.)
Note that the first column (`id`) is the primary key of the `input` table, which 
is referenced in the `simulation` table as the column `input`. 

e.g.

    input 
    ___________________________________________________________________________________

    id    E       nu      mesh    length  strength    loading
    ___________________________________________________________________________________

    1     4.2e9   0.3     1000    0.1     200e6       80e6
    2*    4.2e9   0.3     1000    0.1     200e6       100e6

    * simulation 1 and 2 have different input parameters


### `options` Table

The `options` table contains the parameters which do not affect the physics being 
modelled. These can be specific to the output of the simulation (dump frequency,
dump fields, ...) or to the submission on the cluster (number of nodes, assigned
computation time, ...).

e.g.

    options
    ___________________________________________________________________________________

    id    N_nodes   hours   dump_traction  dump_strength
    ___________________________________________________________________________________

    1**   2         2       True           True

    ** both simulations have the same options

        
### `analysis` Table

The `analysis` table contains information about the simulation analysis, such as
dimensionless parameters describing the set-up or key results of the simulation.
This table is optional.


[License](LICENSE.md)
