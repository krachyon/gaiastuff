"""
This script tries to verify if the classifier works as expected by trying it out against the training data.
Writes out table with classification to TABLE_PATH
"""
__license__ = "GPLv3"

import astropy.table
from pathlib import Path
from astroquery.gaia import Gaia
import h5py
import numpy as np

from common import read_or_query, get_dist_table, combine_tables
from classifier import calculate_classification

GAIA_TABLE_PATH = Path('./verify_gaia.ecsv')
DIST_TABLE_PATH = Path('./verify_dist.ecsv')
COMBINED_TABLE_PATH = Path('./verify_combined.ecsv')


def get_gaia_table_verify(good_fname: Path = Path('./good_ids.h5'),
                          bad_fname: Path = Path('./bad_ids.h5')):
    """Get some test data from gaia servers from files containing good and bad ids"""

    if not (good_fname.exists() and bad_fname.exists()):
        raise ValueError(f'Download {good_fname.name} and {bad_fname.name} '
                         f'from https://keeper.mpdl.mpg.de/d/21d3582c0df94e19921d/')

    n_results = 200
    # Read ids from files, launch Gaia archive queries
    good_ids = list(h5py.File(good_fname)['source_id'][:n_results].astype(int))
    good_idsstr = str(good_ids).replace('[', '(').replace(']', ')')
    good_job = Gaia.launch_job(f'SELECT * FROM gaiaedr3.gaia_source WHERE source_id in {good_idsstr}')
    bad_ids = list(h5py.File(bad_fname)['source_id'][:n_results].astype(int))
    bad_idsstr = str(bad_ids).replace('[', '(').replace(']', ')')
    bad_job = Gaia.launch_job(f'SELECT * FROM gaiaedr3.gaia_source WHERE source_id in {bad_idsstr}')

    # retrieve queries and tag results
    good = good_job.get_results()
    good['prior_knowledge'] = 'good'
    bad = bad_job.get_results()
    bad['prior_knowledge'] = 'bad'

    return astropy.table.vstack([good, bad], join_type='inner')


if __name__ == '__main__':

    gaia_table = read_or_query(GAIA_TABLE_PATH, get_gaia_table_verify)
    dist_table = read_or_query(DIST_TABLE_PATH, lambda: get_dist_table(gaia_table['source_id'].astype(str)))

    combined_table = combine_tables(gaia_table, dist_table).filled()

    combined_table['good'] = calculate_classification(combined_table)

    combined_table.write(COMBINED_TABLE_PATH, format='ascii.ecsv')


    frac_good = np.mean(combined_table['good'])
    print(frac_good)
    assert 0.48 < frac_good < 0.51

