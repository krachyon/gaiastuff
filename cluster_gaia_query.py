"""This script looks up gaia sources around globular clusters and rates their astrometric quality.
Produces an ecsv-table that contains all gaia-columns and information about the
object around which the lookup was done ('target_name_id', 'target_dec', 'target_ra')
and a fractional goodnes parameter from the classifier ('good')
"""
__license__ = "GPLv3"

import re
from pathlib import Path

import astropy.coordinates
import astropy.table
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from tqdm import tqdm

from common import get_dist_table, combine_tables, calculate_distance, read_or_query, write_table
from classifier import calculate_classification

GAIA_TABLE_PATH = Path('./clusters_gaia.pkl.gz')
DIST_TABLE_PATH = Path('./clusters_targets_dist.pkl.gz')
COMBINED_TABLE_PATH = Path('./clusters_targets_combined.pkl.gz')


def get_gaia_table_cluster(object_table, radius):
    """Perform a gaia cone-search around the objects in cluster_names, with given radius.
    Return: combined gaia Table from all queries
    """
    Gaia.ROW_LIMIT = 5000
    object_table = object_table.copy()
    cluster_names = object_table['MAIN_ID']

    print('gaia query...')
    jobs = [Gaia.cone_search(cluster, table_name="gaiaedr3.gaia_source", radius=radius)
            for cluster in tqdm(cluster_names)]
    tables = [job.get_results() for job in jobs]

    object_table.rename_columns(['MAIN_ID', 'RA_d', 'DEC_d'], ['target_name', 'target_ra', 'target_dec'])

    for gaia_table, name in zip(tables, cluster_names):
        gaia_table['target_name'] = name

    def join_func(xcol: astropy.table.Column, ycol: astropy.table.Column):
        # remove extra spaces from Names, also unifies column types
        return [re.sub(r'\s+', ' ', x) for x in xcol], [re.sub(r'\s+', ' ', y) for y in ycol]

    gaia_table = astropy.table.vstack(tables)
    combined_table = astropy.table.join(gaia_table, object_table,
                                    join_funcs={'target_name': join_func})

    combined_table['target_radius'] = radius
    return combined_table


def main(run_model=True, limit=None):
    # Read table from
    # "http://simbad.u-strasbg.fr/simbad/sim-ref?querymethod=bib&simbo=on&submit=submit+bibcode&bibcode=2018A%26A...616A..12G"
    # and get the names of all globular clusters as list

    object_table = astropy.table.Table.read('gaiacolab2018.vo', format='votable')
    object_table = object_table[object_table['OTYPE_S'] == 'GlCl']

    # define what distance to the objects we want to look up stars in
    radius = 1*u.arcminute

    if limit:
        object_table = object_table[:limit]

    gaia_table = read_or_query(GAIA_TABLE_PATH, lambda: get_gaia_table_cluster(object_table, radius))
    dist_table = read_or_query(DIST_TABLE_PATH, lambda: get_dist_table(gaia_table['source_id'].astype(str)))

    combined_table = combine_tables(gaia_table, dist_table).filled()
    calculate_distance(combined_table)

    if run_model:
        combined_table['good'] = calculate_classification(combined_table)
    else:
        combined_table['good'] = combined_table['fidelity_v2']

    write_table(combined_table, COMBINED_TABLE_PATH)

    return combined_table


if __name__ == '__main__':
    combined_table = main()
