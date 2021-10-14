"""
The code in this file is partially copied or adapted from
https://colab.research.google.com/drive/1lPzhGSSIjx2nQ7XM2v8bQZtkf0Atrk0z?usp=sharing
with the permission of the authors
"""
__license__ = "GPLv3"

from pathlib import Path
from typing import Iterable, Optional, Union, Callable
from tqdm import tqdm

import numpy as np


import astropy.table
from astroquery.utils.tap import TapPlus
import astropy.units as u
from astropy.coordinates import SkyCoord

import gzip
import pickle


# Helper functions for pickle based table storage because astropy.table.write
# is hideously slow
def read_table(filename: Path) -> astropy.table.Table:
    with gzip.open(filename, 'rb') as infile:
        return pickle.load(infile)


def write_table(table: astropy.table.Table, filename: Path) -> None:
    with gzip.open(filename, 'wb', compresslevel=4) as outfile:
        pickle.dump(table, outfile)


def chunker(seq, size):
    return list(seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_dist_table(gaia_ids: list[str]) -> astropy.table.Table:
    dist_service = TapPlus('http://dc.zah.uni-heidelberg.de/tap')

    print('gavo query...')
    jobs = []
    for chunk in tqdm(chunker(gaia_ids, 1900)):
        id_string = f'({",".join(chunk)})'
        query = f'SELECT * FROM gedr3spurnew.main WHERE source_id in {id_string}'
        job = dist_service.launch_job(query)
        jobs.append(job)

    dist_table = astropy.table.vstack([job.get_results() for job in jobs])
    return dist_table


def combine_tables(gaia_table: astropy.table.Table,
                                 dist_table: astropy.table.Table) -> astropy.table.Table:

    selector = ['source_id'] + [col for col in dist_table.colnames if col not in gaia_table.colnames]
    combined_table = astropy.table.join(gaia_table, dist_table[selector], keys='source_id', metadata_conflicts='silent')

    return combined_table


def calculate_distance(combined_table:astropy.table.Table) -> None:
    combined_table['target_ra'] = astropy.coordinates.Angle(
            combined_table['target_ra'], unit=u.hourangle)
    combined_table['target_dec'] = astropy.coordinates.Angle(
            combined_table['target_dec'], unit=u.deg)
    combined_table['ra'] = astropy.coordinates.Angle(
            combined_table['ra'], unit=u.deg)
    combined_table['dec'] = astropy.coordinates.Angle(
            combined_table['dec'], unit=u.deg)

    target_coordinates = SkyCoord(
            ra=combined_table['target_ra'], dec=combined_table['target_dec'], frame='icrs', equinox='J2000',
            unit=(u.hourangle, u.deg))
    combined_table['target_coordinates'] = target_coordinates

    source_coordinates = SkyCoord(ra=combined_table['ra'], dec=combined_table['dec'], frame='icrs', equinox='J2016',
                                  unit=(u.deg, u.deg))

    combined_table['source_coordinates'] = source_coordinates
    combined_table['target_dist'] = target_coordinates.separation(source_coordinates)


def read_or_query(fname: Path, query_func: Callable[[], astropy.table.Table]) -> astropy.table.Table:
    if fname.exists():
        table = read_table(fname)
    else:
        table = query_func()
        write_table(table, fname)

    return table

