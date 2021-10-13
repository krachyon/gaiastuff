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

from common import add_dist_table, classify_low_high_sn, get_models, normalize_table

TABLE_PATH = Path('./gc_targets.ecsv')


def get_gaia_table_cluster(cluster_names, radius=astropy.coordinates.Angle('60"')):
    """Perform a gaia cone-search around the objects in cluster_names, with given radius.
    Return: combined gaia Table from all queries
    """

    jobs = [Gaia.cone_search(cluster, table_name="gaiaedr3.gaia_source", radius=radius) for cluster in cluster_names]
    tables = [job.get_results() for job in jobs]

    object_table = Simbad.query_objects(cluster_names)
    object_table.rename_columns(['MAIN_ID', 'RA', 'DEC'], ['target_name', 'target_ra', 'target_dec'])

    for gaia_table, name in zip(tables, cluster_names):
        gaia_table['target_name'] = name

    def join_func(xcol: astropy.table.Column, ycol: astropy.table.Column):
        # remove extra spaces from Names, also unifies column types
        return [re.sub(r'\s+', ' ', x) for x in xcol], [re.sub(r'\s+', ' ', y) for y in ycol]

    gaia_table = astropy.table.vstack(tables)
    gaia_table = astropy.table.join(gaia_table, object_table['target_name', 'target_ra', 'target_dec'],
                                    join_funcs={'target_name': join_func})

    return gaia_table


if __name__ == '__main__':
    # Read table from
    # "http://simbad.u-strasbg.fr/simbad/sim-ref?querymethod=bib&simbo=on&submit=submit+bibcode&bibcode=2018A%26A...616A..12G"
    # and get the names of all globular clusters as list
    candidate_sources = astropy.table.Table.read('davide2018.vo', format='votable')
    cluster_names = list(candidate_sources[candidate_sources['OTYPE_S'] == 'GlCl']['MAIN_ID'])

    # define what distance to the objects we want to look up stars in
    radius = astropy.coordinates.Angle('60"')

    if not TABLE_PATH.exists():
        gaia_table = get_gaia_table_cluster(cluster_names, radius)
        combined_table = add_dist_table(gaia_table)
        combined_table['target_radius'] = radius
        combined_table = combined_table.filled()
    else:
        combined_table = astropy.table.Table.read(TABLE_PATH, format='ascii.ecsv')

    combined_table['good'] = classify_low_high_sn(normalize_table(combined_table), *get_models())

    combined_table.write(TABLE_PATH, format='ascii.ecsv')
