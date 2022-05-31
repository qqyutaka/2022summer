# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms.

import numpy as np
import scipy as sp
import scipy.signal

# import scipy.stats
import pandas as pd

import librosa
import datetime

from audiot.audio_features import AudioFeatures


class SpectralFeatures:
    """
    Represents a matrix of features (optionally with associated labels) for an audio recording.

    Attributes:
        features (ndarray): 2D array containing the feature values, shape (n_frequencies,
            n_samples).
        frequency_axis (ndarray): 1D array of shape (n_frequencies) containing the frequency 
            associated with each frequency band over which the features were computed.  These may 
            or may not be spaced linearly, depending on the feature type.
        time_axis (ndarray): 1d array of shape (n_samples) containing the starting times in seconds
            for the frame over whihc each feature vector was computed.
        time_step (float): The step size (in seconds) between the starting times of adjacend time 
            frames.
        frame_duration (float): The duration in seconds of the time frames over which each feature 
            sample is computed.
    """

    def __init__(
        self, features, frequency_axis, time_axis, time_step, frame_duration,
    ):
        """
        See the class docstring.
        """
        self.features = np.asarray(features)
        self.frequency_axis = np.asarray(frequency_axis)
        self.time_axis = np.asarray(time_axis)
        self.time_step = time_step
        self.frame_duration = frame_duration
        # Validation
        if len(self.frequency_axis) != self.features.shape[0]:
            raise RuntimeError("Length of frequency_axis must match the number of rows in features")
        if len(self.time_axis) != self.features.shape[1]:
            raise RuntimeError("Length of time_axis must match the number of columns in features")

    @property
    def n_frequencies(self):
        """Gets the number of feature dimensions in the feature array."""
        return self.features.shape[0]

    @property
    def n_samples(self):
        """Gets the number of feature samples in the feature array."""
        return self.features.shape[1]

    def apply_mel_filter_bank(
        self, n_mels, min_frequency=None, max_frequency=None, normalize_filters=True
    ):
        """
        Applies the specified Mel filter bank to these, and returns the result as a new
        SpectralFeatures object.
        """
        if min_frequency is None:
            min_frequency = self.frequency_axis.min()
        if max_frequency is None:
            max_frequency = self.frequency_axis.max()
        mel_filter_bank, mel_center_freqs = get_mel_filter_bank(
            frequencies=self.frequency_axis,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            n_mels=n_mels,
            normalize_filters=normalize_filters,
        )
        mel_spectral_features = mel_filter_bank.dot(self.features)
        return SpectralFeatures(
            features=mel_spectral_features,
            frequency_axis=mel_center_freqs,
            time_axis=self.time_axis,
            time_step=self.time_step,
            frame_duration=self.frame_duration,
        )

    def as_audio_features(self):
        """
        Convert this to an AudioFeatures object.
        """
        return AudioFeatures(
            features=self.features.T,
            frame_start_times=self.time_axis,
            frame_duration=self.frame_duration,
        )

    def extract_segments(self, segments):
        """
        Extracts slices of the features matrix for each segment and returns them in a list.
        """
        segment_features = []
        for segment_start, segment_end, _ in segments:
            segment_mask = (segment_start <= self.time_axis) & (self.time_axis < segment_end)
            segment_features.append(self.features[:, segment_mask])
        return segment_features

    def extract_grid_features_for_segments(self, segment_index_list, n_freq_divisions, n_time_divisions):
        """
        Extracts features for each of the specified segments, where the features will have a
        consistent number of dimensions regardless of the durations of the individual segments.  It
        does this by stretching a n_freq_divisions by n_time_divisions grid over the full duration
        of each segment, then averaging the values of the spectral features covered by each grid 
        cell.

        Args:
            segment_index_list (list of tuples): A list of tuples, where each tuple specifies the 
                starting index, ending index, and signal strength (ignored) for each segment to be 
                extracted.
            n_freq_divisions (int): The number of frequency divisions to split each segment into
                when extracting features.
            n_time_divisions (int): The number of time divisions to split each segment into when 
                extracting features.

        Returns:
            ndarray:  A 2d array of shape (n_features, n_segments) containing the features extracted
                for each segment, where n_features = n_time_divisions*n_freq_divisions.  The
                ordering of the features cycles through all the time windows for a given frequency
                first, before cycling to the next frequency to the next frequency.
        """
        n_segments = len(segment_index_list)
        n_features = n_time_divisions * n_freq_divisions
        features = np.zeros([n_features, n_segments])
        for segment_idx in range(n_segments):
            segment_start, segment_end, _ = segment_index_list[segment_idx]
            segment_features = self.features[:, segment_start:segment_end]
            feature_patch = resize_matrix_by_averaging(
                segment_features, (n_freq_divisions, n_time_divisions)
            )
            features[:, segment_idx] = feature_patch.ravel()
        return features


def hertz_to_mel(hertz):
    """
    Convert hertz to mels using the most common / standard formula.
    """
    return 2595 * np.log10(1 + hertz / 700)


def mel_to_hertz(mel):
    """
    Convert mels to hertz using the most common / standard formula.
    """
    hz = 700 * (10 ** (mel / 2595) - 1)
    return hz


def get_mel_filter_bank(frequencies, min_frequency, max_frequency, n_mels, normalize_filters=True):
    """
    Constructs a mel filter bank matrix that can be applied to a data matrix by pre-multiplying:
    filtered_result = mel_filter_bank.dot(data_matrix)

    Args:
        frequencies (ndarray): A vector containing the frequencies for each row of the data matrix
            that the filter bank will be applied to.
        min_frequency (float): The starting frequency of the lowest triangular filter in the filter
            bank.
        max_frequency (float): The ending frequency of the hightest triangular filter in the filter
            bank.
        n_mels (int): The number of triangular filters in the filter bank.
        normalize_filters (bool): Whether or not to normalize the filter bank matrix such that each
            row sums to 1.
    
    Returns:
        mel_filter_bank, center_frequencies (ndarray, ndarray): A tuple containing the filter bank
            matrix and a vector specifying the center frequencies of each of the triangular filters.
    """
    mel_filter_bank = np.zeros((n_mels, len(frequencies)))
    min_mel = hertz_to_mel(min_frequency)
    max_mel = hertz_to_mel(max_frequency)
    bin_edges_mel = min_mel + (max_mel - min_mel) * np.arange(0, n_mels + 2) / (n_mels + 1)
    bin_edges_hz = mel_to_hertz(bin_edges_mel)
    center_frequencies = bin_edges_hz[1:-1]
    for i in range(n_mels):
        low_hz = bin_edges_hz[i]
        mid_hz = bin_edges_hz[i + 1]
        high_hz = bin_edges_hz[i + 2]
        ramp_up = (frequencies - low_hz) / (mid_hz - low_hz)
        ramp_down = (high_hz - frequencies) / (high_hz - mid_hz)
        mel_filter_bank[i, :] = np.maximum(0, np.minimum(ramp_up, ramp_down))
    if normalize_filters:
        mel_filter_bank = mel_filter_bank / mel_filter_bank.sum(axis=1)[:, np.newaxis]
    return mel_filter_bank, center_frequencies


def resize_matrix_by_averaging(input_matrix, new_shape):
    """
    Resizes the input_matrix to a new shape by superimposing two grids (one of the original shape /
    dimensionality, and one of the desired shape) and performing a weighted average of the values
    from the original matrix to determine each value in the new matrix, where the wieghts correspond
    to the amount of overlap the cells from the grid with the old dimensionality have with each cell
    from the grid with the new dimensionality.

    For example, 
        [[1, 2, 3, 4], 
         [5, 6, 7, 8]] 
    resized to shape (3,2) would yield 
        [[1.5, 3.5],
         [3.5, 5.5],
         [5.5, 7.5]].

    Args:
        input_matrix (ndarray): 2d matrix of values to be resized.
        new_dimensions (int, int): A tuple of ints specifying the desired shape as (n_rows, 
            n_columns).
    """
    n_rows, n_cols = input_matrix.shape
    row_resizer = get_row_resizing_matrix(n_rows, new_shape[0])
    col_resizer = get_row_resizing_matrix(n_cols, new_shape[1]).transpose()
    return row_resizer.dot(input_matrix).dot(col_resizer)


def get_row_resizing_matrix(old_dimension, new_dimension):
    """
    Constructs a matrix that can be used to resize the number of rows (or columns) of another 
    matrix.  The values of the new matrix will be weighted averages of the overlapping values of the
    old matrix, with the weighting corresponding to the amount of overlap.

    To resize number of rows:
        result = resizing_matrix.dot(data_matrix)
    To resize number of columns:
        result = data_matrix.dot(resizing_matrix.transpose())

    Args:
        old_dimension (int): The number of rows in the matrix to be resized (or the number of 
            columns if the result will be transposed to resize the columns instead).
        new_dimension (int): The desired number of row in the resized matrix (or the desired number
            of columns if the result will be transposed to resize the columns instead).

    Returns:
        resizing_matrix (ndarray): The matrix that can be pre-multiplied to resize rows, or that 
            can be transposed and then post-multiplied to resize columns.
    """
    bin_step = old_dimension / new_dimension
    bin_edges = np.arange(new_dimension + 1) * bin_step
    resizing_matrix = np.zeros((new_dimension, old_dimension))
    for bin_idx in range(new_dimension):
        left_coord = bin_edges[bin_idx]
        right_coord = bin_edges[bin_idx + 1]
        left_idx = int(left_coord)
        right_idx = int(right_coord)
        if left_idx == right_idx:
            resizing_matrix[bin_idx, left_idx] = 1
        else:
            resizing_matrix[bin_idx, left_idx] = (1 - (left_coord - left_idx)) / bin_step
            resizing_matrix[bin_idx, (left_idx + 1) : right_idx] = 1 / bin_step
            if right_coord > right_idx:
                resizing_matrix[bin_idx, right_idx] = (right_coord - right_idx) / bin_step
    return resizing_matrix

