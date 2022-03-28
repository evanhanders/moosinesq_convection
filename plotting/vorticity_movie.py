"""
Plots some disk slices

Usage:
    masked_movie.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: slices]
    --out_name=<out_name>               Name of figure output directory & base name of saved figures [default: vorticity_movie]
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
from plotpal.file_reader import match_basis
import dedalus.public as d3
import numpy as np
from mpi4py import MPI
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import RegularGridInterpolator as RGI

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

resolution_regex = re.compile('(.*)x(.*)')
for str_bit in root_dir.split('_'):
    if resolution_regex.match(str_bit):
        str_bit = str_bit.split('/')[0]
        res_strs = str_bit.split('x')
        Nphi, Nr = (int(res_strs[0]), int(res_strs[1]))

radius = 1
dealias = 3/2
Ra = 1e11

###Prep mask stuff
with h5py.File('../masks/moosinesq_Ra{:.1e}_{}x{}_de{:.1f}_gamma100.h5'.format(Ra,Nr,Nphi,dealias)) as f:
    mask = f['mask'][()]
    phi = f['phi'][()]
    r = f['r'][()]
mask_interp = RGI((phi.ravel(), r.ravel()), mask, bounds_error=False, fill_value=1)

x_cartesian = np.linspace(-1, 1, int(2*Nr)+1)
y_cartesian = np.linspace(-1, 1, int(2*Nr)+1)
yy_c, xx_c = np.meshgrid(y_cartesian, x_cartesian)
rr_c = np.hypot(xx_c, yy_c)
pp_c = np.arctan2(yy_c, xx_c)
pp_c[pp_c < 0] += 2*np.pi
pmask = mask_interp((pp_c, rr_c))

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
out_name    = args['--out_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)
dpi = int(args['--dpi'])

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(root_dir, file_dir=data_dir, out_name=out_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'col_inch' : int(args['--col_inch']), 'row_inch' : int(args['--row_inch']) }

# Just plot a single plot (1x1 grid) of the field "T eq"
# remove_x_mean option removes the (numpy horizontal mean) over phi
fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_axes([0, 0, 1, 1], polar=False)
cax = fig.add_axes([0.8, 0.06, 0.15, 0.03])

tasks = ['vorticity']
vmin = -20
vmax = 20

color_plot = None
with plotter.my_sync:
    if not plotter.idle:
        while plotter.writes_remain():
            dsets, ni = plotter.get_dsets(tasks)
            time_data = dsets[tasks[0]].dims[0]

            if color_plot is None:
                r = match_basis(dsets[tasks[0]], 'r')
                phi = match_basis(dsets[tasks[0]], 'phi')
                rr, pp = np.meshgrid(r.ravel(), phi.ravel())
                xx = rr*np.cos(pp)
                yy = rr*np.sin(pp)
                color_plot = ax.pcolormesh(xx, yy, dsets[tasks[0]][ni], shading='auto', cmap='PuOr_r', vmin=vmin, vmax=vmax, rasterized=True)
                cbar = plt.colorbar(color_plot, cax=cax, orientation='horizontal')
                cbar.set_label('vorticity')
                cbar.set_ticks((vmin, 0, vmax))
                cbar.set_ticklabels(['{:.2f}'.format(vmin), '0', '{:.2f}'.format(vmax)])
                ax.set_yticks([])
                ax.set_xticks([])

                t_cmap = np.ones([256, 4])
                t_cmap[:, 3] = np.linspace(0, 0.5, 256)
                t_cmap = ListedColormap(t_cmap)

                color2 = ax.pcolormesh(xx_c, yy_c, pmask, shading='auto', cmap=t_cmap, vmin=0, vmax=1, rasterized=True)
                phi_line = np.linspace(0, 2.1*np.pi, 1000)
                ax.plot(np.cos(phi_line), np.sin(phi_line), c='k', lw=1)

                ax.set_xlim(-1.01, 1.01)
                ax.set_ylim(-1.01, 1.01)

                for direction in ['left', 'right', 'bottom', 'top']:
                        ax.spines[direction].set_visible(False)
            else:
                color_plot.set_array(dsets[tasks[0]][ni])

            fig.savefig('{:s}/{:s}_{:06d}.png'.format(plotter.out_dir, plotter.out_name, int(time_data['write_number'][ni]+start_fig-1)), dpi=300, bbox_inches='tight')
