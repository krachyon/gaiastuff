"""
This script tries to verify if the classifier works as expected by trying it out against the training data.
Writes out table with classification to TABLE_PATH
"""

import astropy.table
from pathlib import Path
from astroquery.gaia import Gaia
import h5py

from common import add_dist_table, classify_low_high_sn, get_models, normalize_table

TABLE_PATH = Path('./gaia_verify.ecsv')


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
    if not TABLE_PATH.exists():
        gaia_table = get_gaia_table_verify()
        combined_table = add_dist_table(gaia_table)
    else:
        combined_table = astropy.table.Table.read(TABLE_PATH, format='ascii.ecsv')

    combined_table['good'] = classify_low_high_sn(normalize_table(combined_table), *get_models())

    combined_table.write(TABLE_PATH, format='ascii.ecsv')



