"""
Plots some disk slices

Usage:
    masked_movie.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: slices]
    --out_name=<out_name>               Name of figure output directory & base name of saved figures [default: snapshots]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]
"""
import re
from docopt import docopt
args = docopt(__doc__)
from plotpal.slices import SlicePlotter
import dedalus.public as d3
import numpy as np
from mpi4py import MPI
import h5py
import matplotlib.pyplot as plt

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
out_name    = args['--out_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)
dpi = int(args['--dpi'])

resolution_regex = re.compile('(.*)x(.*)')
for str_bit in root_dir.split('_'):
    if resolution_regex.match(str_bit):
        str_bit = str_bit.split('/')[0]
        res_strs = str_bit.split('x')
        Nphi, Nr = (int(res_strs[0]), int(res_strs[1]))

radius = 1 
dealias = 3/2
dtype = np.float64

coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype, comm=MPI.COMM_SELF)
basis = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=radius, dealias=dealias, dtype=dtype, azimuth_library='matrix')
phi_de, r_de = basis.local_grids((dealias, dealias))
rr, pp = np.meshgrid(r_de.flatten(), phi_de.flatten())

mask = dist.Field(name='mask', bases=basis)
grid_slices = dist.layouts[-1].slices(mask.domain, dealias)
with h5py.File('../masks/moosinesq_{}x{}_de{:.1f}.h5'.format(Nr,Nphi,dealias)) as f:
    mask.require_scales(dealias)
    mask['g'] = 1-f['mask'][()]

T = dist.Field(name='mask', bases=basis)


# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(root_dir, file_dir=data_dir, out_name=out_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'col_inch' : int(args['--col_inch']), 'row_inch' : int(args['--row_inch']) }

# Just plot a single plot (1x1 grid) of the field "T eq"
# remove_x_mean option removes the (numpy horizontal mean) over phi
plotter.setup_grid(num_cols=2, num_rows=1, polar=True, **plotter_kwargs)
plotter.add_polar_colormesh('T', azimuth_basis='phi', radial_basis='r', remove_x_mean=False, r_inner=0, r_outer=radius)
plotter.add_polar_colormesh('vorticity', azimuth_basis='phi', radial_basis='r', remove_x_mean=False, r_inner=0, r_outer=radius, cmap='PuOr_r', cmap_exclusion=0.04)

with plotter.my_sync:
    axs, caxs = plotter._groom_grid()
    tasks = []
    for k, cm in plotter.colormeshes:
        if cm.task not in tasks:
            tasks.append(cm.task)
    if not plotter.idle:
        while plotter.writes_remain():
            for ax in axs: ax.clear()
            for cax in caxs: cax.clear()
            dsets, ni = plotter.get_dsets(tasks)
            time_data = dsets[plotter.colormeshes[0][1].task].dims[0]

            for k, cm in plotter.colormeshes:
                ax = axs[k]
                cax = caxs[k]
                cm.plot_colormesh(ax, cax, dsets[cm.task], ni)
                if cm.task == 'T':
                    T['g'] = dsets['T'][ni,:]
                    T.require_scales(dealias)
                    T['g'] *= mask['g']
                    ax.pcolormesh(pp, rr, T['g'], cmap=cm.cmap, vmin=cm.current_vmin, vmax=cm.current_vmax, rasterized=True)
                    T.require_scales(1)

            plt.suptitle('t = {:.4e}'.format(time_data['sim_time'][ni]))
            plotter.grid.fig.savefig('{:s}/{:s}_{:06d}.png'.format(plotter.out_dir, plotter.out_name, int(time_data['write_number'][ni]+start_fig-1)), dpi=dpi, bbox_inches='tight')


