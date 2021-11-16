import os
import sys
from mpi4py import MPI

import numpy as np
import time
import dedalus.public as d3
from scipy.special import jv
import logging
logger = logging.getLogger(__name__)

# TODO: remove azimuth library? might need to fix DCT truncation
# TODO: automate hermitian conjugacy enforcement
# TODO: finalize filehandlers to process virtual file

#Resolutions:
# 64 x 32 at Ra = 1e5
# 96 x 48 at Ra = 1e6 (marginal)
# 256 x 128 at Ra = 1e7



# Parameters
Nphi, Nr = 256, 128
Ra = 1e7
Pr = 1
dealias = 3/2
stop_sim_time = 50
timestepper = d3.SBDF2
initial_timestep = 0.5
max_timestep = initial_timestep
dtype = np.float64

data_dir = './' + sys.argv[0].split('.py')[0]
data_dir += '_Ra{}_Pr{}_{}x{}/'.format(Ra, Pr, Nphi, Nr)

if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.makedirs('{:s}/'.format(data_dir))
logger.info('saving run in {}'.format(data_dir))


# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype, azimuth_library='matrix')
phi, r = basis.local_grids()
S1_basis = basis.S1_basis(radius=1)

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
p = dist.Field(name='p', bases=basis)
T = dist.Field(name='T', bases=basis)
tau_u = dist.VectorField(coords, name='tau_u', bases=S1_basis)
tau_T = dist.Field(name='tau_T', bases=S1_basis)

# Substitutions
nu = np.sqrt(Pr/Ra)
kappa = nu/Pr

integ = lambda A: d3.Integrate(A, coords)
lift_basis = basis.clone_with(k=2) # Natural output basis
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)

#strain_rate = d3.grad(u) + d3.trans(d3.grad(u))
#shear_stress = d3.azimuthal(strain_rate(r=1))

#Heating
Q = dist.Field(name='Q', bases=basis)
Q['g'] = 4 
Q = d3.Grid(Q).evaluate()

r_vec = dist.VectorField(coords, name='er', bases=basis.radial_basis)
r_vec['g'][1] = r

# Problem
problem = d3.IVP([p, u, T, tau_u, tau_T], namespace=locals())
problem.add_equation("div(u) = 0")
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u,-1) - r_vec*T = - dot(u, grad(u))")
problem.add_equation("dt(T) - kappa*lap(T)  + lift(tau_T,-1)                 = - dot(u, grad(T)) + kappa*Q")
#problem.add_equation("shear_stress = 0")
problem.add_equation("azimuthal(u(r=1)) = 0")
problem.add_equation("radial(u(r=1)) = 0", condition="nphi != 0")
problem.add_equation("p(r=1) = 0", condition='nphi == 0') # Pressure gauge
problem.add_equation("T(r=1) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
T.fill_random('g', seed=42, distribution='standard_normal') # Random noise
T.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes
T['g'] *= 1e-3
T['g'] += 1 - r**2

# Analysis
snapshots = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=0.1, max_writes=20)
snapshots.add_task(T, scales=(1, 1))

scalars = solver.evaluator.add_file_handler(data_dir+'scalars', sim_dt=0.01)
scalars.add_task(integ(0.5*d3.dot(u,u)), name='KE')

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(d3.dot(u,u), name='u2')

timestep = initial_timestep
CFL = d3.CFL(solver, initial_timestep, cadence=1, safety=0.2, threshold=0.1, max_dt=max_timestep)
CFL.add_velocity(u)


# Main loop
hermitian_cadence = 100
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        solver.step(timestep)
        timestep = CFL.compute_timestep()
        if (solver.iteration-1) % 10 == 0:
            max_u = np.sqrt(flow.max('u2'))
            logger.info("Iteration=%i, Time=%e, dt=%e, max(u)=%e" %(solver.iteration, solver.sim_time, timestep, max_u))
        # Impose hermitian symmetry on two consecutive timesteps because we are using a 2-stage timestepper
        if solver.iteration % hermitian_cadence in [0, 1]:
            for f in solver.state:
                f.require_grid_space()
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*dist.comm.size))

# Post-processing
if dist.comm.rank == 0:
    scalars.process_virtual_file()

