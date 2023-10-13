#Download the library
import sys
import cplex
import docplex.mp

# Set up the prescriptive model
# first import the Model class from docplex.mp
from docplex.mp.model import Model

# create one model instance, with a name
m = Model(name='telephone_production')

# by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
desk = m.continuous_var(name='desk')
cell = m.continuous_var(name='cell')

# write constraints
# constraint #1: desk production is greater than 100
m.add_constraint(desk >= 100)

# constraint #2: cell production is greater than 100
m.add_constraint(cell >= 100)

# constraint #3: assembly time limit
ct_assembly = m.add_constraint( 0.2 * desk + 0.4 * cell <= 400)

# constraint #4: paiting time limit
ct_painting = m.add_constraint( 0.5 * desk + 0.4 * cell <= 490)

# express the objective
m.maximize(12 * desk + 20 * cell)

m.print_information()


#Solve with the model
s = m.solve()
m.print_solution()

#Infeasible models

# create a new model, copy of m
im = m.copy()
# get the 'desk' variable of the new model from its name
idesk = im.get_var_by_name('desk')
# add a new (infeasible) constraint
im.add_constraint(idesk >= 1100);
# solve the new proble, we expect a result of None as the model is now infeasible
ims = im.solve()
if ims is None:
    print('- model is infeasible')

