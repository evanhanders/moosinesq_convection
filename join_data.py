'''
Join distributed dedalus output data.

Usage:
      join_data.py <case>... [--data_type=<data_type> --cleanup]

Options:
      --data_type=<data_type>      Type of data to join; if provided join a single data.
      --cleanup                    Cleanup after join

'''

import os
import sys
import logging
logger = logging.getLogger(__name__)

import dedalus.public
from dedalus.tools  import post

from docopt import docopt
import glob

args = docopt(__doc__)

data_dir = args['<case>'][0]
for d in glob.glob(data_dir):
    base_path = os.path.abspath(d)+'/'

    cleanup = args['--cleanup']

    logger.info("joining data from Dedalus run {:s}".format(d))

    if args['--data_type'] is not None:
        data_types=[args['--data_type']]
    else:
        data_types = ['slices', 'scalars', 'checkpoint',]

    for data_type in data_types:
        logger.info("merging {}".format(data_type))
        try:
            print(base_path+data_type)
            post.merge_virtual_analysis('{:s}/{:s}/'.format(base_path,data_type), cleanup=cleanup)
        except:
            logger.info("missing {}".format(data_type))
            
    logger.info("done join operation for {:s}".format(d))
