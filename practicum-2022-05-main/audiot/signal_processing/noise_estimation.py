
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy as sp


class NoiseEstimatorInterface(metaclass=ABCMeta):
    @abstractmethod
    def estimate_noise_magnitude_spectrum(self, magnitude_stft, sample_rate, frame_step):
        """
        Args:
            magnitude_stft (ndarray): Magnitude of the STFT with shape [n_freq_bands, 
                n_time_windows].
            sample_rate (float): The sampling rate of the original signal, in Hz.
            frame_step (int): The step size (in samples) between the beginnings of adjacent STFT
                windows.
        
        Returns:
            ndarray: The estimated magnitude spectrum of the noise component of the signal, with 
                shape [n_freq_bands, n_time_windows].
        """
        raise NotImplementedError("This method should be overridden.")


class MedianFilterNoiseEstimator(NoiseEstimatorInterface):
    """
    Estimates the noise magnitude spectrum by windowing the input magnitude spectrum and running
    a median filter on it.  The estimated spectrum is linearly blended between the center points
    of each window over which the medians were taken.
    """

    def __init__(self, noise_estimation_period_seconds, step_seconds=None):
        self.noise_estimation_period_seconds = noise_estimation_period_seconds
        if step_seconds:
            self.step_seconds = step_seconds
        else:
            self.step_seconds = self.noise_estimation_period_seconds / 2

    def estimate_noise_magnitude_spectrum(self, magnitude_stft, sample_rate, frame_step):
        filter_length = int(self.noise_estimation_period_seconds * sample_rate / frame_step + 0.5)
        middle_offset = filter_length // 2
        filter_step = int(self.step_seconds * sample_rate / frame_step + 0.5)
        if magnitude_stft.shape[1] - filter_length + 1 <= 0:
            # Signal duration is shorter than noise_estimation_period_seconds, so just estimate
            # over the full length of the signal instead.
            sparse_noise_estimates = np.median(magnitude_stft, axis=1, keepdims=True)
        else:
            sparse_noise_estimates = np.hstack(
                tuple(
                    np.median(
                        magnitude_stft[:, filter_idx : filter_idx + filter_length],
                        axis=1,
                        keepdims=True,
                    )
                    for filter_idx in range(
                        0, magnitude_stft.shape[1] - filter_length + 1, filter_step
                    )
                )
            )
        noise_estimate = np.zeros(magnitude_stft.shape)
        linear_fade_out = (np.arange(filter_step, 0, -1) / filter_step)[np.newaxis, :]
        linear_fade_in = 1 - linear_fade_out
        # Fill in the frames before the mid point of the first median filter with the results of
        # the first median filter.
        noise_estimate[:, :middle_offset] = sparse_noise_estimates[:, [0]]
        # Linearly blend between the mid points where each median filter was taken
        for filter_idx in range(sparse_noise_estimates.shape[1] - 1):
            start_idx = middle_offset + filter_idx * filter_step
            noise_estimate[:, start_idx : start_idx + filter_step] = sparse_noise_estimates[
                :, [filter_idx]
            ].dot(linear_fade_out) + sparse_noise_estimates[:, [filter_idx + 1]].dot(linear_fade_in)
        # Fill in the frames after the mid point of the last median filter with the results of
        # the last median filter.
        noise_estimate[
            :, middle_offset + (sparse_noise_estimates.shape[1] - 1) * filter_step :
        ] = sparse_noise_estimates[:, [-1]]
        return noise_estimate


class RankFilterNoiseEstimator(NoiseEstimatorInterface):
    """
    Estimates the noise magnitude spectrum by windowing the input magnitude spectrum and running
    a rank filter on it.  The estimated spectrum is linearly blended between the center points
    of each window over which the rank filters were taken.
    """

    def __init__(self, rank_percentile, noise_estimation_period_seconds, step_seconds=None):
        self.rank_percentile = rank_percentile
        self.noise_estimation_period_seconds = noise_estimation_period_seconds
        if step_seconds:
            self.step_seconds = step_seconds
        else:
            self.step_seconds = self.noise_estimation_period_seconds / 2

    def estimate_noise_magnitude_spectrum(self, magnitude_stft, sample_rate, frame_step):
        filter_length = int(self.noise_estimation_period_seconds * sample_rate / frame_step + 0.5)
        rank = int(self.rank_percentile * (filter_length - 1) + 0.5)
        middle_offset = filter_length // 2
        filter_step = int(self.step_seconds * sample_rate / frame_step + 0.5)
        sparse_noise_estimates = np.hstack(
            tuple(
                np.partition(
                    magnitude_stft[:, filter_idx : filter_idx + filter_length], kth=rank, axis=1,
                )[:, [rank]]
                for filter_idx in range(0, magnitude_stft.shape[1] - filter_length + 1, filter_step)
            )
        )
        noise_estimate = np.zeros(magnitude_stft.shape)
        linear_fade_out = (np.arange(filter_step, 0, -1) / filter_step)[np.newaxis, :]
        linear_fade_in = 1 - linear_fade_out
        # Fill in the frames before the mid point of the first median filter with the results of
        # the first median filter.
        noise_estimate[:, :middle_offset] = sparse_noise_estimates[:, [0]]
        # Linearly blend between the mid points where each median filter was taken
        for filter_idx in range(sparse_noise_estimates.shape[1] - 1):
            start_idx = middle_offset + filter_idx * filter_step
            noise_estimate[:, start_idx : start_idx + filter_step] = sparse_noise_estimates[
                :, [filter_idx]
            ].dot(linear_fade_out) + sparse_noise_estimates[:, [filter_idx + 1]].dot(linear_fade_in)
        # Fill in the frames after the mid point of the last median filter with the results of
        # the last median filter.
        noise_estimate[
            :, middle_offset + (sparse_noise_estimates.shape[1] - 1) * filter_step :
        ] = sparse_noise_estimates[:, [-1]]
        return noise_estimate


class MedianFilterNoiseEstimatorSlow(NoiseEstimatorInterface):
    """
    NOT RECOMMENDED.  Instead, use MedianFilterNoiseEstimator.
    Estimates noise by taking a median filter with the specified length around each frame of the
    stft.

    Doing a full median filter across every row of the STFT is VERY slow.  
    Also, the median filter pads the beginning and end of the signal with zeros, causing the 
    median filter to under-estimate the noise at the beginning and ending of the recording.  
    This in turn causes an audible increase in volume at the beginning and end since the Wiener
    filter thinks there's less noise there (when there isn't) and thus attenuates things less.
    The MedianFilterNoiseEstimator class does not have the boundary condition problem, runs
    much faster, and seems to produce output that is nearly indistinguishable to this (other 
    than not having the boundary volume issues).
    """

    def __init__(self, noise_estimation_period_seconds):
        self.noise_estimation_period_seconds = noise_estimation_period_seconds

    def estimate_noise_magnitude_spectrum(self, magnitude_stft, sample_rate, frame_step):
        filter_length = int(self.noise_estimation_period_seconds * sample_rate / frame_step + 0.5)
        noise_estimate = sp.signal.medfilt2d(magnitude_stft, kernel_size=(1, filter_length))
        return noise_estimate
