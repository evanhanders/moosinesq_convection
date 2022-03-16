"""
This script merges a virtual file so that it can be scp'd off a supercomputer and used in visualization locally.

Usage:
    merge_virtual_files.py <root_dir>
"""
import hashlib
import pathlib
import os
import glob
import shutil

import h5py
import numpy as np
from mpi4py import MPI

from dedalus.tools.general import natural_sort
from dedalus.tools.post import get_assigned_sets, MPI_RANK, MPI_SIZE

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from docopt import docopt
args = docopt(__doc__)

root_dir = args['<root_dir>']
handlers = ['slices', ]
cleanup = False

def merge_virtual(joint_file, virtual_path):
    """
    Merge HDF5 setup from part of a distributed analysis set into a joint file.
    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    virtual_path : str or pathlib.Path
        Path to a joined virtual file 
    """
    virtual_path = pathlib.Path(virtual_path)
    logger.info("Merging setup from {}".format(virtual_path))

    with h5py.File(str(virtual_path), mode='r') as virtual_file:
        # File metadata
        try:
            joint_file.attrs['set_number'] = virtual_file.attrs['set_number']
        except KeyError:
            joint_file.attrs['set_number'] = virtual_file.attrs['file_number']
        joint_file.attrs['handler_name'] = virtual_file.attrs['handler_name']
        try:
            joint_file.attrs['writes'] = writes = virtual_file.attrs['writes']
        except KeyError:
            joint_file.attrs['writes'] = writes = len(virtual_file['scales']['write_number'])
        # Copy scales (distributed files all have global scales)
        virtual_file.copy('scales', joint_file)
        # Tasks
        virtual_tasks = virtual_file['tasks']

        joint_tasks = joint_file.create_group('tasks')
        for taskname in virtual_tasks:
            virtual_dset = virtual_tasks[taskname]
            joint_dset = joint_tasks.create_dataset(taskname, data=virtual_dset)

            # Dataset metadata
            joint_dset.attrs['task_number'] = virtual_dset.attrs['task_number']
            joint_dset.attrs['constant'] = virtual_dset.attrs['constant']
            joint_dset.attrs['grid_space'] = virtual_dset.attrs['grid_space']
            joint_dset.attrs['scales'] = virtual_dset.attrs['scales']



            # Dimension scales
            for i, virtual_dim in enumerate(virtual_dset.dims):
                joint_dset.dims[i].label = virtual_dim.label
                if joint_dset.dims[i].label == 't':
                    for sn in ['sim_time', 'world_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
                        scale = joint_file['scales'][sn]
                        joint_dset.dims.create_scale(scale, sn)
                        joint_dset.dims[i].attach_scale(scale)
                else:
                    if virtual_dim.label == 'constant' or virtual_dim.label == '':
                        scalename = 'constant' 
                    else:
                        hashval = hashlib.sha1(np.array(virtual_dset.dims[i][0])).hexdigest()
                        scalename = 'hash_' + hashval
                    scale = joint_file['scales'][scalename]
                    joint_dset.dims.create_scale(scale, scalename)
                    joint_dset.dims[i].attach_scale(scale)

def merge_virtual_file_single_set(set_path, cleanup=False):
    set_path = pathlib.Path(set_path)
    logger.info("Merging virtual file {}".format(set_path))

    set_stem = set_path.stem
    joint_path = set_path.parent.joinpath("{}_merged.h5".format(set_stem))

    # Create joint file, overwriting if it already exists
    with h5py.File(str(joint_path), mode='w') as joint_file:
        # Setup joint file based on first process file (arbitrary)
        merge_virtual(joint_file, set_path)

    # Cleanup after completed merge, if directed
    if cleanup:
        folder = set_path.parent.joinpath("{}/".format(set_stem))
        logger.info("cleaning up {}".format(folder))
        if os.path.isdir(folder):
            partial_files = folder.glob('*.h5')
            for pf in partial_files:
                os.remove(pf)
            os.rmdir(folder)
        os.remove(set_path)
        os.rename(joint_path, set_path)

for data_dir in handlers:
    base_path = '{}/{}'.format(root_dir, data_dir)
    set_path = pathlib.Path(base_path)

    #cleanup partial merged files
    partial_merged = set_path.glob('*_merged.h5')
    partial_merged = natural_sort(partial_merged)
    my_partial_merged = partial_merged[MPI_RANK::MPI_SIZE] 
    for cleanup_file in my_partial_merged:
        os.remove(cleanup_file)

    set_paths = get_assigned_sets(set_path, distributed=False)

    for set_path in set_paths:
        merge_virtual_file_single_set(set_path, cleanup=cleanup)

