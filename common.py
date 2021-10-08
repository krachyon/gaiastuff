from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import astropy.coordinates
import astropy.table
import h5py


from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from astroquery.utils.tap import TapPlus
import pyvo as vo


import tensorflow as tf
import os

required_fields = [
    'source_id',
    'l',
    'b',
    'parallax_error',
    'parallax_over_error',
    'pmra',
    'astrometric_sigma5d_max',
    'pmdec',
    'pmdec_error',
    'pmra_error',
    'astrometric_excess_noise',
    'visibility_periods_used',
    'ruwe',
    'astrometric_gof_al',
    'ipd_gof_harmonic_amplitude',
    'ipd_frac_odd_win',
    'ipd_frac_multi_peak',
    'ecl_lat',
    'matched_transits_removed',
    'astrometric_params_solved',
    'astrometric_excess_noise_sig',
    'phot_proc_mode',
    'phot_g_mean_mag',
    'phot_rp_mean_mag',
    'phot_bp_mean_mag',
    'parallax',
    'phot_bp_rp_excess_factor',
    'theta_arcsec_worst_source',
    'norm_dg',
    #'dist_nearest_neighbor_at_least_m30_brighter',  # 30 mag fainter
    #'dist_nearest_neighbor_at_least_m4_brighter',  # 4 mag fainter
    #'dist_nearest_neighbor_at_least_m2_brighter',  # 2 mag fainter
    'dist_nearest_neighbor_at_least_0_brighter',  # Same magnitude
    'dist_nearest_neighbor_at_least_2_brighter',  # 2 mag brighter
    'dist_nearest_neighbor_at_least_4_brighter',
    'dist_nearest_neighbor_at_least_6_brighter',
    'dist_nearest_neighbor_at_least_10_brighter'
]

feature_transforms = {
    'parallax_over_error': np.abs, # Use |parallax/error|
    'pmdec': np.abs,
    'pmra': np.abs,
    # 'ecl_lat': np.abs
}

feature_names = [
    # =====================
    # Gaia catalog columns:
    # =====================
    'parallax_error',
    'parallax_over_error',
    'pmra',
    'astrometric_sigma5d_max',
    'pmdec',
    'pmdec_error',
    'pmra_error',
    'astrometric_excess_noise',
    'visibility_periods_used',
    'ruwe',
    'astrometric_gof_al',
    'ipd_gof_harmonic_amplitude',
    'ipd_frac_odd_win',
    'ipd_frac_multi_peak',
    # 'ecl_lat',
    'matched_transits_removed',
    'astrometric_params_solved',
    'astrometric_excess_noise_sig',
    # 'phot_proc_mode',
    # =====================
    # Neighbor information:
    # =====================
    # 'theta_arcsec_worst_source',
    # 'norm_dg',
    # 'dist_nearest_neighbor_at_least_m30_brighter',
    # 'dist_nearest_neighbor_at_least_m4_brighter',
    # 'dist_nearest_neighbor_at_least_m2_brighter',
    'dist_nearest_neighbor_at_least_0_brighter',  # TODO this isn't in the data...
    'dist_nearest_neighbor_at_least_2_brighter',
    'dist_nearest_neighbor_at_least_4_brighter',
    'dist_nearest_neighbor_at_least_6_brighter',
    'dist_nearest_neighbor_at_least_10_brighter'
]



def get_dist_table(gaia_table):
    dist_service = TapPlus('http://dc.zah.uni-heidelberg.de/tap')
    ids = str(list(gaia_table['source_id'])).replace('[', '(').replace(']', ')')
    job = dist_service.launch_job(f'SELECT * FROM gedr3spur.main WHERE source_id in {ids}')
    return job.get_results()


def extract_features(*input_datasets, ignore_features=[]):
    """
    Extracts features from one or multiple structured arrays, and
    optionally returns an array containing labels (assumed to be
    uniform for each input dataset).

    Inputs:
      *input_datasets (np.recordarray): Structured numpy array(s)
        from which the features will be extracted.

      ignore_features (Optional[list]): A list of features to ignore.
    Returns:
      A dense array containing the features, with shape
      (# of data points, # of features). If `return_labels` is not
      `None`, a second dense array containing labels is returned,
      with shape (# of data points). If `bucket_weights` is not `None`, a
      final array containing weights to apply to each sample will be returned.
    """
    # Remove features that are in the ignore_features list
    feature_names_filt = []
    for n in feature_names:
        if n not in ignore_features:
            feature_names_filt.append(n)

    n_features = len(feature_names_filt)
    n_d = sum([len(d) for d in input_datasets])

    feat = np.empty((n_d, n_features), dtype='f4')

    k0 = 0
    for d in input_datasets:
        for i, key in enumerate(feature_names_filt):
            feat[k0:k0 + len(d), i] = d[key].astype('f4')
        k0 += len(d)

    for i, key in enumerate(feature_names_filt):
        if key in feature_transforms:
            f = feature_transforms[key]
            print(f'Applying function "{f.__name__}" to {key}')
            feat[:, i] = f(feat[:, i])

    return feat


def normalize_table(data: astropy.table.Table):
    clip_col = ('norm_dg', 'theta_arcsec_worst_source')
    for column_name in data.columns:
        if column_name.startswith('dist_nearest_neighbor') or (column_name in clip_col):
            idx = ~np.isfinite(data[column_name]) | (data[column_name] > 30.)
            data[column_name][idx] = 30.
    return data


def classify_low_high_sn(data, split_sn=4.5):
    prob = np.empty(len(data), dtype='f4')

    idx = np.abs(data['parallax_over_error']) < split_sn

    # Classify low-S/N sources using model that does not take
    # parallax_over_error into account
    if np.any(idx):

        model = tf.keras.models.load_model(Path('model_lowsnr'))
        features = extract_features(
            data[idx],
            ignore_features=['parallax_over_error'],
        )
        p = model.predict(features)
        prob[idx] = p.flat
        n_good = np.count_nonzero(p>0.5)
        pct_good = 100 * n_good / p.size
        print(f'{n_good} of {p.size} ({pct_good:.2f}%) low-S/N sources "good"')

    # Classify high-S/N sources using model that takes
    # parallax_over_error into account
    if np.any(~idx):
        model = tf.keras.models.load_model(Path('model_highsnr'))
        features = extract_features(data[~idx])
        p = model.predict(features)
        prob[~idx] = p.flat
        n_good = np.count_nonzero(p>0.5)
        pct_good = 100 * n_good / p.size
        print(f'{n_good} of {p.size} ({pct_good:.2f}%) high-S/N sources "good"')

    return prob