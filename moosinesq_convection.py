"""
Dedalus script for moosinesq convection

Usage:
    moosinesq_convection.py [options] 
    moosinesq_convection.py <config> [options] 

Options:
    --Ra=<Rayleigh>            The Rayleigh number [default: 1e7]
    --nr=<n>                   radial resolution [default: 128]
    --gamma=<g>                Inverse damping timescale [default: 40]

    --label=<label>            Optional additional case name label
    --restart=<restart>        Path to checkpoint file to restart from

    --plot_model               If flagged, create and plt.show() some plots of the 1D atmospheric structure.
"""

import os
import sys
import h5py
from mpi4py import MPI

from docopt import docopt
import numpy as np
import time
import dedalus.public as d3
from scipy.special import jv
import logging
logger = logging.getLogger(__name__)

args = docopt(__doc__)

# Parameters
Ra_str = args['--Ra']
gamma_str = args['--gamma']
Nr = int(args['--nr'])
Nphi = 2*Nr
Ra = float(Ra_str)
radius = 1
Pr = 1
gamma = float(gamma_str)
dealias = 3/2
stop_sim_time = 1e3
timestepper = d3.SBDF2
dtype = np.float64
initial_timestep = np.min((0.1, 1/gamma))
max_timestep = initial_timestep
safety = 0.15

data_dir = './' + sys.argv[0].split('.py')[0]
data_dir += '_Ra{}_Pr{}_gamma{}_{}x{}'.format(Ra_str, Pr, gamma_str, Nphi, Nr)
if args['--label'] is not None:
    data_dir += '_{}'.format(args['--label'])
data_dir += '/'
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.makedirs('{:s}/'.format(data_dir))
logger.info('saving run in {}'.format(data_dir))


# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=radius, dealias=dealias, dtype=dtype, azimuth_library='matrix')
phi, r = dist.local_grids(basis)
S1_basis = basis.S1_basis()
x = r * np.cos(phi)
y = r * np.sin(phi)

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
p = dist.Field(name='p', bases=basis)
T = dist.Field(name='T', bases=basis)
tau_u = dist.VectorField(coords, name='tau_u', bases=S1_basis)
tau_T = dist.Field(name='tau_T', bases=S1_basis)
tau_p = dist.Field(name='tau_p')

# Substitutions
nu = np.sqrt(Pr/Ra)
kappa = nu/Pr

integ = lambda A: d3.Integrate(A, coords)
lift_basis = basis#.clone_with(k=2) # Natural output basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)

mask = dist.Field(name='mask', bases=basis)
grid_slices = dist.layouts[-1].slices(mask.domain, dealias)
mask_file = 'masks/moosinesq_Ra{:.1e}_{}x{}_de{:.1f}_gamma{}.h5'.format(Ra, Nr,Nphi,dealias,gamma_str)
with h5py.File(mask_file) as f:
    logger.info('loading mask from {}'.format(mask_file))
    mask.change_scales(dealias)
    mask['g'] = f['mask'][:,grid_slices[-1]]
mask = d3.Grid(mask).evaluate()

y_vec = dist.VectorField(coords, name='er', bases=basis)
y_vec['g'][0] = np.cos(phi)
y_vec['g'][1] = np.sin(phi)
y_vec = d3.Grid(y_vec).evaluate()

T0 = dist.Field(name='T0', bases=basis)
T0['g'] = -y/(2*radius)

t = dist.Field()

# Problem
problem = d3.IVP([p, u, T, tau_p, tau_u, tau_T], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u,-1) = y_vec*T  - dot(u, grad(u)) - mask*u*gamma")
problem.add_equation("dt(T) - kappa*lap(T)  + lift(tau_T,-1)       = - dot(u, grad(T)) - mask*(T-T0)*gamma")
problem.add_equation("u(r=radius) = 0")
problem.add_equation("integ(p) = 0")
problem.add_equation("T(r=radius) = T0(r=radius)")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time


if args['--restart'] is None:
    # Initial conditions
    T.fill_random('g', seed=42, distribution='standard_normal') # Random noise
    T.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes
    T['g'] *= 1e-3
    T['g'] += T0['g']
    write_mode = 'overwrite'
else:
    logger.info('restarting from {}'.format(args['--restart']))
    write, initial_timestep = solver.load_state(args['--restart'])
    write_mode = 'append'

checkpoint = solver.evaluator.add_file_handler(data_dir+'checkpoint', sim_dt=10, max_writes=1, mode=write_mode)
checkpoint.add_tasks(solver.state, layout='g')

# Analysis
snapshots = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=0.05, max_writes=20, mode=write_mode)
snapshots.add_task(T, scales=(1, 1))
snapshots.add_task((1-mask)*T, scales=(1, 1), name='mask_T')
snapshots.add_task(d3.curl(u), scales=(1, 1), name='vorticity')

scalars = solver.evaluator.add_file_handler(data_dir+'scalars', sim_dt=0.01, mode=write_mode)
scalars.add_task(integ(0.5*d3.dot(u,u)), name='KE')

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(d3.dot(u,u), name='u2')

timestep = initial_timestep
CFL = d3.CFL(solver, initial_timestep, cadence=1, safety=safety, threshold=0.1, max_dt=max_timestep)
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
    logger.info('making final checkpoint')
    final_checkpoint = solver.evaluator.add_file_handler(data_dir+'final_checkpoint', sim_dt=timestep/10, max_writes=1, mode=write_mode)
    final_checkpoint.add_tasks(solver.state, layout='g')
    solver.step(timestep)

    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*dist.comm.size))

# Post-processing
if dist.comm.rank == 0:
    scalars.process_virtual_file()

