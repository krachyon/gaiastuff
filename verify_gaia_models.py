from common import *

TABLE_PATH = Path('./gaia_verify.ecsv')


def get_gaia_table_verify():
    fnames = ['good_ids.h5'] #'dubious_100pc_ids.h5',
    ids = []
    for fname in fnames:
        ids += list(h5py.File(fname)['source_id'][:2000])
    idsstr = str(ids).replace('[', '(').replace(']', ')')
    job = Gaia.launch_job(f'SELECT * FROM gaiaedr3.gaia_source WHERE source_id in {idsstr}')
    return job.get_results()

if __name__ == '__main__':
    if not TABLE_PATH.exists():

        gaia_table = get_gaia_table_verify()
        dist_table = get_dist_table(gaia_table)
        combined_table = astropy.table.join(gaia_table, dist_table, keys='source_id', metadata_conflicts='silent')

        combined_table.write(TABLE_PATH, format='ascii.ecsv')
    else:
        combined_table = astropy.table.Table.read(TABLE_PATH, format='ascii.ecsv')

    # TODO HAX to deal with missing field, should contain same info?
    combined_table['dist_nearest_neighbor_at_least_0_brighter'] = \
        combined_table['dist_nearest_neighbor_at_least_equally_bright']

    combined_table['good'] = classify_low_high_sn(combined_table.as_array())
