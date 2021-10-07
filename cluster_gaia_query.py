from common import *

TABLE_PATH = Path('./gaia.ecsv')

# define what you want to lookup here
# and how far away
CLUSTER_NAMES = ['M2', 'M15', 'M4', 'M71', 'NGC 411']
radius = astropy.coordinates.Angle('50"')


# Optionally resolve objects first to get more info to plot or if Gaia can't handle some names
# would have to extract coords from this and use in cone search
# object_table = Simbad.query_objects(CLUSTER_NAMES)
# print(object_table)


def get_gaia_table_cluster():
    jobs = [Gaia.cone_search(cluster, table_name="gaiaedr3.gaia_source", radius=radius) for cluster in CLUSTER_NAMES]
    tables = [job.get_results() for job in jobs]
    for gaia_table, name in zip(tables, CLUSTER_NAMES):
        gaia_table['lookup_name'] = name

    gaia_table = astropy.table.vstack(tables)
    return gaia_table



if __name__ == '__main__':
    if not TABLE_PATH.exists():

        gaia_table = get_gaia_table_cluster()

        dist_table = get_dist_table(gaia_table)
        combined_table = astropy.table.join(gaia_table, dist_table, keys='source_id', metadata_conflicts='silent')

        combined_table.write(TABLE_PATH, format='ascii.ecsv')
    else:
        combined_table = astropy.table.Table.read(TABLE_PATH, format='ascii.ecsv')

    # TODO HAX to deal with missing field, should contain same info?
    combined_table['dist_nearest_neighbor_at_least_0_brighter'] = \
        combined_table['dist_nearest_neighbor_at_least_equally_bright']

    assert set(required_fields).issubset(combined_table.colnames)
    # plt.errorbar(combined_table['ra'], combined_table['dec'], combined_table['ra_error'] / 60, combined_table['dec_error'] / 60, fmt='g.', capsize=2)
    # plt.show()

    combined_table['good'] = classify_low_high_sn(combined_table.as_array())