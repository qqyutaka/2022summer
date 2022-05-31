from pathlib import Path

import numpy as np
import scipy as sp
import scipy.signal
import pandas as pd

from audiot.audio_signal import AudioSignal
from audiot.signal_processing.wiener_filter import WienerFilter
from audiot.signal_processing.noise_estimation import MedianFilterNoiseEstimator


class AutoSegmenter:
    @classmethod
    def get_default_segmenter(cls, min_frequency=0, max_frequency=np.inf):
        """
        Return an AutoSegmenter object with settings tuned to values that seem to work pretty well
        for a wide variety of data.  Only signal energy within the frequency range defined by the
        min and max frequencies will be examined when segmenting sounds out, so those values can be
        used to better target sounds that are expected to be within a given frequency range.
        """
        return cls(
            detection_threshold=0.15,
            context_threshold=0.05,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            rank_percentile=1.0,
            segment_smoothing_period=0.03,
            wiener_filter=WienerFilter.get_very_aggressive_noise_reduction_filter(),
        )

    def __init__(
        self,
        detection_threshold=0.15,
        context_threshold=0.05,
        min_frequency=0,
        max_frequency=np.inf,
        rank_percentile=1.0,
        segment_smoothing_period=0.03,
        wiener_filter=None,
    ):
        """
        Args:
            detection_threshold (float):  A threshold in the range [0.0, 1.0] that determines when
                the detection of a segment should be triggered.  This threshold is compared against
                values coming from the Wiener transfer function, which roughly represent the 
                percentage of the current signal that is thought to be foreground noise (as opposed
                to background noise).
            context_threshold (float):  A threshold in the range [0.0, detection_threshold] that
                determines how much context around a detected segment should also be included as 
                part of that segment.  This value should be lower than or equal to the 
                detection_threshold.
            min_frequency (float):  The lower bound of the frequency range that should be examined 
                to determine the segmentation.
            max_frequency (float):  The upper bound of the frequency range that should be examined 
                to determine the segmentation.
            rank_percentile (float):  A percentile value in the range [0.0, 1.0] that determines 
                which value from each column of the Wiener transfer function is chosen to be 
                compared against the thresholds.  A value of 1.0 indicates that the largest (peak) 
                value should be used.  Smaller values require 
            segment_smoothing_period (float):  The length (in seconds) of the median filter used to
                smooth the values chosen from each column of the Wiener transfer function before
                they are compared against the thresholds.
            wiener_filter (WienerFilter): The object used to estimate how much of the energy in each
                frequency band is signal versus how much is noise (the Wiener transfer function), 
                which is used for segmentation.
        """
        self.detection_threshold = detection_threshold
        self.context_threshold = context_threshold
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.segment_smoothing_period = segment_smoothing_period
        self.rank_percentile = rank_percentile
        if wiener_filter is None:
            self.wiener_filter = WienerFilter.get_very_aggressive_noise_reduction_filter()
        else:
            self.wiener_filter = wiener_filter

    def segment_signal_strength_features(self, signal_strength_features):
        """
        Returns the detected sound segments as a list of 3-tuples, where each tuple is of the form
        (segment_start_index, segment_end_index, signal_strength).  The signal strength value is a
        rough estimate of what percentage of the energy is signal (as opposed to background noise)
        in the frequency bands in which the signal is present, and will be in the range [0.0, 1.0].
        """
        self.features = signal_strength_features
        self.freq_mask = (self.min_frequency <= self.features.frequency_axis) & (
            self.features.frequency_axis < self.max_frequency
        )
        # Find rank index closest to the specified percentile value
        self.rank = int(
            self.rank_percentile * (self.features.features[self.freq_mask, :].shape[0] - 1) + 0.5
        )
        # Find the closest odd median filter kernel size to the specified duration in seconds
        self.med_filt_kernel_size = int(
            ((self.segment_smoothing_period / self.features.time_step) // 2) * 2 + 1
        )
        # Find the rank'th smallest value in each column of the transfer function
        self.rank_tf = np.partition(
            self.features.features[self.freq_mask, :], kth=self.rank, axis=0
        )[self.rank, :]
        # Median filter to smooth out spikes / noise
        self.rank_tf_filt = sp.signal.medfilt(self.rank_tf, kernel_size=self.med_filt_kernel_size)
        above_detection_mask = self.detection_threshold <= self.rank_tf_filt
        # The median filter sometimes threw off boundaries with small gaps near the beginning or
        # end.  OR in the non-median filtered context as well to help fix that.
        above_context_mask = (self.context_threshold <= self.rank_tf_filt) | (
            self.context_threshold <= self.rank_tf
        )
        # Extract the segments -- Walk through the segments detected by the context threshold, and
        # only add them to the list of detected segments if the detection threshold is also
        # exceeded somewhere within that segment.
        segment_index_list = []
        start_idx = np.argmax(above_context_mask)
        if above_context_mask[start_idx]:
            while True:
                idx_step = np.argmin(above_context_mask[start_idx:])
                if idx_step == 0:
                    end_idx = self.features.n_samples
                else:
                    end_idx = start_idx + idx_step
                if np.any(above_detection_mask[start_idx:end_idx]):
                    signal_strength = np.mean(
                        np.max(self.features.features[self.freq_mask, start_idx:end_idx], axis=0)
                    )
                    # There are occasional edge cases where values on the extreme ends of the median
                    # filter can cause a detection in the middle of the filter window that doesn't
                    # actually end up getting linked to the energy on either end that caused it, and
                    # thus you can end up with a segment with a very low (or even zero)
                    # signal_strength value.  Filter these out by double checking that the
                    # signal_strength is at least larger than the context threshold.
                    if self.context_threshold <= signal_strength:
                        segment_index_list.append((start_idx, end_idx, signal_strength))
                if end_idx >= self.features.n_samples:
                    break
                idx_step = np.argmax(above_context_mask[end_idx:])
                if idx_step == 0:
                    break
                start_idx = end_idx + idx_step
        return segment_index_list

    def segment_signal(self, audio_signal):
        """
        Returns the detected sound segments as a list of 3-tuples, where each tuple is of the form
        (segment_start_time_seconds, segment_end_time_seconds, signal_strength).  The signal
        strength value is a rough estimate of what percentage of the energy is signal (as opposed
        to background noise) in the frequency bands in which the signal is present, and will be in 
        the range [0.0, 1.0].
        """
        features = self.wiener_filter.get_signal_strength_features(audio_signal)
        segment_index_list = self.segment_signal_strength_features(features)
        segment_time_list = self.convert_segments_from_indexes_to_seconds(segment_index_list, features)
        return segment_time_list

    @classmethod
    def convert_segments_from_indexes_to_seconds(cls, segment_index_list, signal_strength_features):
        segment_time_list = [
            (
                signal_strength_features.time_axis[start_idx],
                signal_strength_features.time_axis[end_idx - 1] + signal_strength_features.frame_duration,
                sig_pct,
            )
            for start_idx, end_idx, sig_pct in segment_index_list
        ]
        return segment_time_list

    @classmethod
    def convert_segment_list_to_labels_dataframe(cls, segment_time_list, label_to_assign=""):
        labels_df = pd.DataFrame(segment_time_list, columns=("onset", "offset"))
        labels_df["event_label"] = label_to_assign
        return labels_df

