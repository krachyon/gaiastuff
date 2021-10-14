"""
The code in this file is partially copied or adapted from
https://colab.research.google.com/drive/1lPzhGSSIjx2nQ7XM2v8bQZtkf0Atrk0z?usp=sharing
with the permission of the authors
"""
__license__ = "GPLv3"

from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import Iterable, Optional, Union
import astropy.table

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
    'matched_transits_removed',
    'astrometric_params_solved',
    'astrometric_excess_noise_sig',
    'dist_nearest_neighbor_at_least_0_brighter',
    'dist_nearest_neighbor_at_least_2_brighter',
    'dist_nearest_neighbor_at_least_4_brighter',
    'dist_nearest_neighbor_at_least_6_brighter',
    'dist_nearest_neighbor_at_least_10_brighter'
]
feature_transforms = {
    'parallax_over_error': np.abs,  # Use |parallax/error|
    'pmdec':               np.abs,
    'pmra':                np.abs,
}


clip_col = ('theta_arcsec_worst_source',)
for key in feature_names:
    if key.startswith('dist_nearest_neighbor') or key in clip_col:
        feature_transforms[key] = lambda x: np.clip(x, 0., 5.)

def normalize_table(data: astropy.table.Table):
    """perform necessary clipping of outlier data for feature exctraction. Return copy"""
    data = data.copy()

    clip_col = ('norm_dg', 'theta_arcsec_worst_source')
    for column_name in data.columns:
        if column_name.startswith('dist_nearest_neighbor') or (column_name in clip_col):
            invalid_elements = ~np.isfinite(data[column_name]) | (data[column_name] > 30.)
            data[column_name][invalid_elements] = 30.
    return data


def extract_features(*input_datasets: Union[np.recarray, astropy.table.Table],
                     ignore_features: Iterable = tuple()):
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

    feature_names_filt = [name for name in feature_names if name not in ignore_features]

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
            feat[:, i] = f(feat[:, i])

    return feat


def classify_low_high_sn(data: Union[astropy.table.Table, np.recarray], low_snr_model, high_snr_model, split_sn=4.5):
    """
    classify astrometric quality of input data. Use either low_snr_model
    or high_snr_model depending on snr cuttoff split_sn

    Parameters
    ----------
    data:
        Table or recarray with columns specified in feature names. Make sure to call extract_features on it first!
    low_snr_model
    high_snr_model
    split_sn

    Returns
    -------
        array of probabilities of source being "good"

    """
    prob = np.empty(len(data), dtype='f4')

    idx = np.abs(data['parallax_over_error']) < split_sn

    # Classify low-S/N sources using model that does not take
    # parallax_over_error into account
    if np.any(idx):
        features = extract_features(
                data[idx],
                ignore_features=['parallax_over_error'],
        )
        p = low_snr_model.predict(features)
        prob[idx] = p.flat

    # Classify high-S/N sources using model that takes
    # parallax_over_error into account
    if np.any(~idx):
        features = extract_features(data[~idx])
        p = high_snr_model.predict(features)
        prob[~idx] = p.flat

    return prob


def get_models():
    return tf.keras.models.load_model(Path('model_lowsnr')), tf.keras.models.load_model(Path('model_highsnr'))


def calculate_classification(combined_table: astropy.table.Table) -> np.ndarray:
    probs = classify_low_high_sn(normalize_table(combined_table), *get_models())
    return probs
