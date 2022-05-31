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


class AudioFeatures:
    """
    Represents a matrix of features (optionally with associated labels) for an audio recording.

    Attributes:
        features (ndarray): 2D array containing the feature values, shape (n_samples, n_features).
        frame_start_times (ndarray): 1D array of shape (n_samples) containing the starting times in 
            seconds for the frame over which each feature vector was computed.  These values are 
            relative to start_datetime, if it is specified.
        frame_duration (float): The duration in seconds of the frames over which each feature sample 
            is computed.
        start_datetime (datetime): The datetime that the frame_start_times are relative to, or None 
            if there's not a specific datetime for the data. Usually this is the datetime
            corresponding to the beginning of the frame over which the first feature vector was
            calculated.
        event_names (list): A list of strings specifying the name of each event that can occur.  
            None if not applicable.
        events (ndarray): 2D array containing the estimated probability in the range [0, 1] of each 
            event being present for each feature sample.  Shape (n_samples, n_event_types).  None 
            if not applicable.
        true_events (ndarray): 2D array containing the true probabilities in the range [0, 1] of 
            each event being present for each feature sample.  Shape (n_samples, n_event_types).  
            None if not applicable.
    """

    def __init__(
        self,
        features,
        frame_start_times,
        frame_duration,
        start_datetime=None,
        event_names=None,
        events=None,
        true_events=None,
    ):
        """
        See the class docstring.
        """
        # Validation checks
        if event_names and events is not None and len(event_names) != events.shape[1]:
            raise ValueError(
                "Mismatch between the number event names and the number of events in the events "
                "array."
            )
        if events is not None and true_events is not None and events.shape != true_events.shape:
            raise ValueError("The events and true_events arrays should be the same size.")
        self.features = np.asarray(features)
        self.frame_start_times = np.asarray(frame_start_times)
        self.frame_duration = frame_duration
        self.start_datetime = start_datetime
        self.event_names = event_names
        self.events = np.asarray(events) if events is not None else None
        self.true_events = np.asarray(true_events) if true_events is not None else None

    @property
    def n_samples(self):
        """Gets the number of feature samples in the feature array."""
        return self.features.shape[0]

    @property
    def n_features(self):
        """Gets the number of feature dimensions in the feature array."""
        return self.features.shape[1]

    @property
    def n_event_types(self):
        """Gets the number of event types."""
        return len(self.event_names) if self.event_names else None

    @property
    def frame_end_times(self):
        """Gets the ending time (in seconds) for each frame."""
        return self.frame_start_times + self.frame_duration

    def deltas(self, n_samples_per_side=2):
        """
        Computes the deltas for the features represented by this object and returns them as a 
        separate Features object.  To keep the the output delta features the same dimension as 
        the original input features, the beginning and end of the feature matrix is padded by 
        duplicating the first and last sample when the delta computation window extends beyond the 
        bounds of the original feature matrix.

        Args:
            n_samples_per_side (int):  The number of samples on either side of the current sample to
                use to estimate the derivative.  The default value of 2 (which is the value that is 
                commonly used e.g. for MFCCs) estimates the derivative over a window containing 5
                samples: two to the left of the current sample, the middle sample that we are
                estimating the deltas for, and two samples to the right of it.
        Returns:
            Features: A Features object containing the deltas of these features (and with the same
                dimensions).
        """
        padded_features = np.vstack(
            (
                np.tile(self.features[0, :], (n_samples_per_side, 1)),
                self.features,
                np.tile(self.features[-1, :], (n_samples_per_side, 1)),
            )
        )
        # See the formula to calculate the deltas at this webpage:
        # http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
        # d_t = \frac
        # 	{ \sum_{n=1}^N n(c_{t+n} - c_{t-n}) }
        # 	{ 2\sum_{n=1}^N n^2 }
        # Where:
        #   d_t is the delta value for the feature under consideration at time t.
        #   N in the above equation corresponds to n_samples_per_side in this function.
        #   c_{t} corresponds to the value of the feature we are estimating deltas for at time t.
        #   c_{t+1} corresponds to the next value of the feature (at time t+1), c{t-1} corresponds
        #       to the previous value of that feature (at time t-1), etc.
        # To calculate the above for N=2, you end up with the numerator being:
        #       1*(c_{t+1} - c_{t-1}) + 2*(c_{t+2} - c_{t-2})
        #   which, rearranged, is:
        #       -2*c_{t-2} + -1*c_{t-1} + 0*c_{t} + 1*c_{t+1} + 2*c_{t+2}
        # This can be written as an element-wise multiplication between two vectors:
        #       [-2  -1  0  1  2] .* [c_{t-2}  c_{t-1}  c_{t}  c_{t+1}  c_{t+2}]
        # If we want to compute the above across every value of t, a fast way to do it is to just
        # convolve the [2 1 0 -1 -2] vector with our vector of feature values,
        # [c_{0}, c_{1}, c{2}, ...].  (Note that the convolution operation includes an inherent
        # "flip" of the convolution kernel vector, so we have to reverse the ordering of the vector
        # passed in as the kernel to get the desired result.)
        # If we want to do the above for a bunch of different feature dimensions instead of just one
        # feature, we can do a 2d convolution with the same convolution kernel, but applied to a 2d
        # feature matrix.  This is what the code below does.

        # Construct a convolution kernel we can use to calculate the numerator of the equation
        # above:
        convolution_kernel = np.arange(n_samples_per_side, -n_samples_per_side - 1, step=-1)
        deltas_numerator = sp.signal.convolve2d(
            convolution_kernel[:, np.newaxis], padded_features, mode="valid"
        )
        return AudioFeatures(
            features=deltas_numerator / (convolution_kernel ** 2).sum(),
            frame_start_times=self.frame_start_times,
            frame_duration=self.frame_duration,
            start_datetime=self.start_datetime,
            event_names=self.event_names,
            events=self.events,
            true_events=self.true_events,
        )

    def stack_adjacent_frames(self, n_frames_to_stack):
        """
        Shifts and stacks the feature matrix to incorporate more time information.

        The results are only constructed where valid data is available (i.e. no filler values are
        used to extend past the ends of the available data).  This will cause the number of samples 
        in the result to be smaller by (n_frames_to_stack-1) than that of the current object.

        Note that if frame_start_times is not regularly spaced, then stacking adjacent frames would
        cause inconsistency in the resulting features because adjacent frames that are spaced 
        further apart would be stacked the same as those spaced closer together.  This is most
        likely to occur if features are calculated from multiple different files, then concatenated
        together, and then this method is called (thus attempting to stack across file boundaries).
        In this case, the features from each individual file should be stacked first, then the 
        resulting stacked features can be concatenated together into one object.
        
        Args:
            n_frames_to_stack (int): The number of frames that should be stacked together to 
                construct a single resulting feature vector.

        Returns:
            An AudioFeatures object containing the stacked features.

        Raises:
            ValueError: If frame_start_times is not regularly spaced.  This would cause 
                inconsistencies when stacking adjacent frames with varying temporal spacing.
        """
        if n_frames_to_stack < 2:
            raise ValueError("n_frames_to_stack should be 2 or greater.")
        frame_steps = np.diff(self.frame_start_times)
        if not np.all(np.isclose(frame_steps, frame_steps[0])):
            raise ValueError(
                "Stacking adjacent frames will cause inconsistencies because frame_start_times is "
                "not regularly spaced.  See the note in the docstring."
            )
        n_samples = self.n_samples - n_frames_to_stack + 1
        n_features = n_frames_to_stack * self.n_features
        stacked_features = np.zeros((n_samples, n_features))
        for start_idx in range(n_frames_to_stack):
            feature_idx = start_idx * self.n_features
            stacked_features[:, feature_idx : feature_idx + self.n_features] = self.features[
                start_idx : start_idx + n_samples
            ]
        frame_duration = self.frame_duration + (n_frames_to_stack - 1) * frame_steps[0]
        return AudioFeatures(
            features=stacked_features,
            frame_start_times=self.frame_start_times[:n_samples],
            frame_duration=frame_duration,
            start_datetime=self.start_datetime,
            event_names=self.event_names,
            events=None,
            true_events=None,
        )

    def downsample(self, downsampling_factor, start_index=0):
        """
        Returns a downsampled Features object.  If downsampling_factor is set to 3, the returned
        Features object will contain every 3rd sample from this Features object, starting with the
        sample at start_index.

        Args:
            downsampling_factor (int): The factor by which to downsample.  If set to 4, the results 
                will contain every 4th sample from this Features object.
            start_index (int): The index at which to start the downsampling.  If set to 1 with the 
                downsampling_factor set to 3, then the samples with indexes 1, 4, 7, 10, etc will be
                returned.
        
        Returns:
            Features: A Features object containing the downsampled features.
        """
        if self.events:
            events = self.events[start_index::downsampling_factor]
        else:
            events = None
        if self.true_events:
            true_events = self.true_events[start_index::downsampling_factor]
        else:
            true_events = None
        return AudioFeatures(
            features=self.features[start_index::downsampling_factor, :],
            frame_start_times=self.frame_start_times[start_index::downsampling_factor],
            frame_duration=self.frame_duration,
            start_datetime=self.start_datetime,
            event_names=self.event_names,
            events=events,
            true_events=true_events,
        )

    def match_labels(
        self,
        labels,
        onset_col="onset",
        offset_col="offset",
        label_col="event_label",
        overlap_threshold=0.5,
        set_true_events=True,
    ):
        """
        Returns an events array constructed by matching label data up with the feature samples.

        Matches label data (from a pandas dataframe) up with feature samples, ignoring any event 
        labels that do not match one of the labels in self.event_names.  The data frame passed in 
        should only contain events for this specific data (and not for other files).  If multiple of
        the same event label overlap, the overlapping area will be counted multiple times when 
        determining whether or not to apply the label to a frame.

        Args:
            labels (DataFrame): A Pandas DataFrame containing the starting time, ending time, and 
                label string for each label associated with this data.
            onset_col (string): The name of the column containing the starting time for each label.
            offset_col (string): The name of the column containing the ending time for each label.
            label_col (string): The name of the column containing the label strings.
            overlap_threshold (float): The fraction of a feature frame that must overlap with a 
                given label for that feature frame to be assigned that label.
            set_true_events (bool): Whether or not the true_events field of this object should be 
                set to the results of this operation.
        
        Returns:
            ndarray: An events matrix containing zeros and ones based on whether each feature frame
                was labeled with that event.  Shape (n_samples, n_event_types).
        """
        events = np.zeros([self.n_samples, self.n_event_types])
        threshold_overlap_length = self.frame_duration * overlap_threshold
        # Iterate through each event type
        for event_idx, event_name in enumerate(self.event_names):
            total_overlap_per_frame = np.zeros(self.n_samples)
            label_times = labels[labels[label_col] == event_name][
                [onset_col, offset_col]
            ].to_numpy()
            if label_times.size == 0:
                continue
            for label_start, label_end in label_times:
                overlap_per_frame = np.maximum(
                    0,
                    np.minimum(label_end, self.frame_end_times)
                    - np.maximum(label_start, self.frame_start_times),
                )
                total_overlap_per_frame += overlap_per_frame
            events[:, event_idx] = total_overlap_per_frame >= threshold_overlap_length
        if set_true_events:
            self.true_events = events
        return events

    @classmethod
    def stack(cls, *audio_features):
        """
        Stacks multiple AudioFeatures objects into one AudioFeatures object.  All input
        AudioFeatures objects must have the same values for n_samples, and the output object will
        also have the same n_samples.  The n_features property for the output object will be the sum
        of all the n_features properties of the inputs.  The event information in the output object
        will be copied from the first input object, while the even information from subsequent 
        inputs will be ignored.

        Args:
            *audio_features: Variable length argument list containing each AudioFeatures object to 
                be stacked, in the order in which they should be stacked.

        Returns:
            The AudioFeatures object containing the stacked features.

        Raises:
            ValueError: If the AudioFeatures objects are not compatible to be stacked (different 
                numbers of samples or different frame durations, etc).
        """
        frame_duration = audio_features[0].frame_duration
        if not all(audio_features[0].n_samples == feat.n_samples for feat in audio_features):
            raise ValueError(
                "Cannot stack AudioFeatures objects with different numbers of samples."
            )
        if not all(frame_duration == feat.frame_duration for feat in audio_features):
            raise ValueError(
                "Cannot stack AudioFeatures objects with different frame_duration values."
            )
        # Use the frame start times from the first AudioFeatures object passed in.
        return cls(
            features=np.hstack([feat.features for feat in audio_features]),
            frame_start_times=audio_features[0].frame_start_times,
            frame_duration=frame_duration,
            start_datetime=audio_features[0].start_datetime,
            event_names=audio_features[0].event_names,
            events=audio_features[0].events,
            true_events=audio_features[0].true_events,
        )

    @classmethod
    def concatenate(cls, *audio_features):
        """
        Concatenates multiple AudioFeatures objects into one AudioFeatures object.  The output will
        have n_samples equal to the sum of n_samples of all of the inputs.  The n_features property
        must be the same for all inputs, and will also be the same for the output.

        Args:
            *audio_features: Variable length argument list containing each AudioFeatures object to 
                be concatenated, in the order in which they should be concatenated.

        Returns:
            The AudioFeatures object containing the concatenated features.

        Raises:
            ValueError: If the AudioFeatures objects are not compatible to be concatenated 
                (different numbers of feature dimensions or different frame durations, etc).
        """
        # TODO: This method has not been tested very thoroughly.
        frame_duration = audio_features[0].frame_duration
        start_datetime = audio_features[0].start_datetime
        n_samples = sum(feat.n_samples for feat in audio_features)
        if not all(audio_features[0].n_features == feat.n_features for feat in audio_features):
            raise ValueError(
                "Cannot concatenate AudioFeatures objects with different numbers of features."
            )
        if not all(frame_duration == feat.frame_duration for feat in audio_features):
            raise ValueError(
                "Cannot concatenate AudioFeatures objects with different frame_duration values."
            )
        # Only keep event info if the event types match up across all AudioFeatures objects.
        event_names = audio_features[0].event_names
        if event_names and all(event_names == feat.event_names for feat in audio_features):
            if all(feat.events is not None for feat in audio_features):
                events = np.vstack([feat.events for feat in audio_features])
            else:
                events = None
            if all(feat.true_events is not None for feat in audio_features):
                true_events = np.vstack([feat.true_events for feat in audio_features])
            else:
                true_events = None
        else:
            event_names = None
            events = None
            true_events = None
        # If absolute start times are given for all AudioFeatures objects, use them to compute the
        # frame_start_times. Otherwise just concatenate each AudioFeatures object on the end of the
        # previous using relative frame_start_times.
        frame_start_times = np.zeros(n_samples)
        start_idx = 0
        if all(feat.start_datetime is not None for feat in audio_features):
            for feat in audio_features:
                frame_start_times[start_idx : start_idx + feat.n_samples] = (
                    feat.frame_start_times + (feat.start_datetime - start_datetime).total_seconds()
                )
                start_idx += feat.n_samples
        else:
            start_time = 0
            for feat in audio_features:
                frame_start_times[start_idx : start_idx + feat.n_samples] = (
                    feat.frame_start_times + start_time
                )
                start_time += feat.frame_end_times[-1]
                start_idx += feat.n_samples
        return cls(
            features=np.vstack([feat.features for feat in audio_features]),
            frame_start_times=frame_start_times,
            frame_duration=frame_duration,
            start_datetime=start_datetime,
            event_names=event_names,
            events=events,
            true_events=true_events,
        )


# Functions ########################################################################################


def calc_log_mel_energy_features(
    audio_signal,
    window_length_seconds=0.05,
    overlap=0.5,
    min_frequency=100,
    max_frequency=8000,
    n_mels=13,
    channel=0,
):
    """
    Compute and return an AudioFeatures object containing log mel energy features for the provided
    AudioSignal object.

    Args:
        audio_signal (AudioSignal): The object containing the audio waveform to compute the 
            features from.  If it has multiple channels, only the waveform specified by the channel
            parameter will be used in computing the features.
        window_length_seconds (float): The length of the window (in seconds) over which each feature
            vector is computed.
        overlap (float): A number in the range [0,1) specifying how much overlap there should be 
            between adjacent windows that we compute features for.  A value of 0 gives no overlap,
            while 0.5 would give 50% overlap.
        min_frequency (float): The minimum frequency (in Hertz) over which the features will be 
            computed.  This is the frequency where the lower edge (or start) of the first triangular 
            filter in the Mel-scaled filter bank will be located.
        max_frequency (float): The maximum frequency (in Hertz) over which the features will be 
            computed.  This is the frequency where the upper edge (or end) of the last triangular 
            filter in the Mel-scaled filter bank will be located.
        n_mels (int): This specifies the number of filters in the Mel-scaled filter bank.  This
            number of triangular filters will be spaced according to the Mel scale between the 
            frequencies specified by min_frequency and max_frequency.  This number also determines
            the number of dimensions each output feature vector will have.  Higher values will give
            features with higher resolution along the frequency axis, while lower numbers will give
            lower resolution.  In speech recognition, this has often (historically) been set to 13
            when computing MFCCs (which can be computed from log mel energies with a couple more 
            computation steps), and then deltas and delta-deltas were computed and stacked on top of 
            the original 13 MFCC features to give a total of 39-dimensional features.
        channel (int): Which channel of the AudioSignal object to compute the features from.

    Returns:
        AudioFeatures: The computed log mel energy features, with the feature dimensionality equal
            to n_mels.
    """
    n_fft = int(window_length_seconds * audio_signal.sample_rate + 0.5)
    # We want a hop_length of at least 1 so that the window does actually slide across the signal
    # by at least one sample at a time.
    hop_length = int(max(1, n_fft * (1 - overlap) + 0.5))
    # Note: The power parameter below is the exponent applied to the melspectrogram. So setting it
    # to 1 gives energy, setting it to 2 gives power, etc. However, since we're taking the
    # logarithm afterward, this just turns into a multiplicative factor that would just scale the
    # resulting features: log(x^2) = 2 * log(x). Thus, there's not really much point to set it to
    # anything other than 1, which corresponds with the name "log mel energies" anyway (as opposed
    # to "log mel powers" or something else).
    mel_energy = librosa.feature.melspectrogram(
        y=audio_signal.signal[:, channel],
        sr=audio_signal.sample_rate,
        power=1,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=min_frequency,
        fmax=max_frequency,
        n_mels=n_mels,
    )
    log_mel_energy = np.log(mel_energy)
    frame_duration = n_fft / audio_signal.sample_rate
    # Apparently librosa pads the signal so that the feature frame at index t is centered on the
    # waveform sample at index t*hop_length.  So the first frame is actually centered at time 0,
    # meaning its left half extends before the waveform begins.  Thus I subtract off half of the
    # frame duration from the frame_start_times.
    frame_start_times = (
        np.arange(0, log_mel_energy.shape[1]) * hop_length / audio_signal.sample_rate
        - frame_duration / 2
    )
    return AudioFeatures(
        features=log_mel_energy.T,
        frame_start_times=frame_start_times,
        frame_duration=frame_duration,
    )


if __name__ == "__main__":
    # Test match_labels_dataframe()
    n_features = 2
    frame_size = 10
    step_size = 5
    sample_rate = 100
    frame_start_times = np.arange(3, step=step_size / sample_rate)
    frame_duration = frame_size / sample_rate
    data = [
        [0.06, 0.14, "dog"],
        [0.98, 1.02, "dog"],
        [1.05, 1.06, "dog"],
        [1.07, 1.09, "dog"],
        [2.02, 2.03, "dog"],
        [2.20, 2.80, "dog"],
        [0.70, 2.40, "cat"],
        [1.70, 2.20, "giraffe"],
    ]
    event_names = ["dog", "cat", "elephant"]
    test_labels = pd.DataFrame(data, columns=["onset", "offset", "event_label"])
    feat = AudioFeatures(
        features=sp.rand(len(frame_start_times), n_features),
        frame_start_times=frame_start_times,
        frame_duration=frame_duration,
        start_datetime=None,
        event_names=event_names,
        events=np.zeros([60, len(event_names)]),
    )
    feat.match_labels(test_labels)
    print(feat.n_features)
    print(feat.n_samples)
    print(feat.frame_start_times)
    print(feat.frame_end_times)
    print(feat.n_event_types)
    print(np.hstack((feat.true_events, feat.frame_start_times[:, np.newaxis])))
    print(feat.features[:5, :])
    print(feat.features[-5:, :])
    print("-" * 80)
    concat_feat = AudioFeatures.concatenate(feat, feat)
    print(concat_feat.n_features)
    print(concat_feat.n_samples)
    print(concat_feat.frame_start_times)
    print(concat_feat.frame_end_times)
    print(concat_feat.event_names)
    print(concat_feat.n_event_types)
    print(np.hstack((concat_feat.true_events, concat_feat.frame_start_times[:, np.newaxis])))
    print(concat_feat.features[:5, :])
    print(concat_feat.features[-5:, :])
    print("-" * 80)
    stacked_feat = feat.stack_adjacent_frames(2)
    print(stacked_feat.features[:5, :])
    print(stacked_feat.features[-5:, :])
    print(stacked_feat.frame_duration)
    print(stacked_feat.n_samples)
    print(stacked_feat.n_features)
    stacked_feat = feat.stack_adjacent_frames(3)
    print(stacked_feat.features[:5, :])
    print(stacked_feat.features[-5:, :])
    print(stacked_feat.frame_duration)
    print(stacked_feat.n_samples)
    print(stacked_feat.n_features)
    stacked_feat = feat.stack_adjacent_frames(5)
    print(stacked_feat.features[:5, :])
    print(stacked_feat.features[-5:, :])
    print(stacked_feat.frame_duration)
    print(stacked_feat.n_samples)
    print(stacked_feat.n_features)
    downsampled_feat = stacked_feat.downsample(3)
    print(downsampled_feat.features[:2, :])
    print(downsampled_feat.features[-2:, :])
    print(downsampled_feat.frame_duration)
    print(downsampled_feat.n_samples)
    print(downsampled_feat.n_features)
    downsampled_feat = stacked_feat.downsample(3, start_index=2)
    print(downsampled_feat.features[:2, :])
    print(downsampled_feat.features[-2:, :])
    print(downsampled_feat.frame_duration)
    print(downsampled_feat.n_samples)
    print(downsampled_feat.n_features)
    print(stacked_feat.frame_start_times[:10])
    print(downsampled_feat.frame_start_times[:10])
    print(feat.frame_duration)
    print(stacked_feat.frame_duration)
    print(downsampled_feat.frame_duration)
