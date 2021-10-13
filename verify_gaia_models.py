import astropy.table
from pathlib import Path
from astroquery.gaia import Gaia
import h5py

from common import *

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
        dist_table = get_dist_table(gaia_table)
        combined_table = astropy.table.join(gaia_table, dist_table, keys='source_id', uniq_col_name='{col_name}{table_name}', table_names=['','1'])
        combined_table['dist_nearest_neighbor_at_least_0_brighter'] = \
            combined_table['dist_nearest_neighbor_at_least_equally_bright']

        combined_table = normalize_table(combined_table)

        combined_table.write(TABLE_PATH, format='ascii.ecsv')
    else:
        combined_table = astropy.table.Table.read(TABLE_PATH, format='ascii.ecsv')

    # TODO HAX to deal with missing field, should contain same info?


    combined_table['good'] = classify_low_high_sn(combined_table.filled().as_array(), *get_models())


    print(np.mean(combined_table['good']))


