import numpy as np
import scipy as sp
import scipy.fft
import scipy.signal
from audiot.audio_signal import AudioSignal
from audiot.signal_processing.noise_estimation import MedianFilterNoiseEstimator
from audiot.spectral_features import SpectralFeatures

import matplotlib as mpl
import matplotlib.pyplot as plt


class WienerFilter:
    @classmethod
    def get_perceptually_pleasant_noise_reduction_filter(cls):
        """
        Returns a noise reduction filter tuned to be perceptually pleasant, meaning that it tries
        to reduce noise without introducing noticeable artifacts or distortions in how the audio
        sounds.
        """
        return cls(
            noise_estimator=MedianFilterNoiseEstimator(noise_estimation_period_seconds=10),
            attenuation_limit=0.07,
            oversubtraction_factor=2,
            approximate_frame_length_seconds=0.01,
            transfer_function_smoothing_period=0.1,
        )
    
    @classmethod
    def get_very_aggressive_noise_reduction_filter(cls):
        """
        Returns a very aggressive noise reduction filter that will completely zero out most of the
        signal except for places where the energy clearly sticks out relative to the context around
        it.
        """
        # No attenuation limit needed since we're not reconstructing a denoised waveform that
        # needs to have residual musical noise masked out for better human perception.
        # Oversubtraction set high to require sounds to be decently louder than the estimated
        # background noise before any energy makes it through the filter.
        return cls(
            noise_estimator=MedianFilterNoiseEstimator(noise_estimation_period_seconds=10),
            attenuation_limit=0.0,
            oversubtraction_factor=5,
            approximate_frame_length_seconds=0.01,
            transfer_function_smoothing_period=0.1,
        )

    def __init__(
        self,
        noise_estimator=None,
        attenuation_limit=0.07,
        oversubtraction_factor=2,
        approximate_frame_length_seconds=0.01,
        transfer_function_smoothing_period=0.1,
    ):
        if noise_estimator is None:
            self.noise_estimator = WienerFilter.MedianFilterNoiseEstimator(
                noise_estimation_period_seconds=10
            )
        else:
            self.noise_estimator = noise_estimator
        self.attenuation_limit = attenuation_limit
        self.oversubtraction_factor = oversubtraction_factor
        self.approximate_frame_length_seconds = approximate_frame_length_seconds
        self.transfer_function_smoothing_period = transfer_function_smoothing_period
        # Small number used to avoid division by zero (which leads to NaN / inf contamination)
        self.tiny_float = np.finfo(np.float_).eps

    def _round_to_nearest_power_of_two(self, number):
        rounded_up = int(2 ** np.ceil(np.log2(number)))
        rounded_down = int(2 ** np.floor(np.log2(number)))
        if rounded_up - number < number - rounded_down:
            return rounded_up
        else:
            return rounded_down

    def high_pass(self, audio_signal, cutoff_frequency, butterworth_order=2):
        second_order_sections = sp.signal.butter(
            butterworth_order,
            cutoff_frequency,
            btype="highpass",
            output="sos",
            fs=audio_signal.sample_rate,
        )
        filtered_signal = sp.signal.sosfilt(second_order_sections, audio_signal.signal, axis=0)
        return AudioSignal(filtered_signal, audio_signal.sample_rate)

    def get_signal_strength_features(self, audio_signal=None):
        """
        Returns signal strength features for the given audio signal.  These features are a 2d matrix
        (similar to a spectrogram) where each cell contains an estimate of what percentage of the 
        signal energy in that cell is from a foreground sound.
        
        Args:
            audio_signal (AudioSignal): The audio signal over which to compute the Wiener transfer 
                function spectrogram.  If None, returns the transfer function spectrogram for the 
                last AudioSignal processed (or raises an exception if no signals have been 
                processed yet).

        Returns:
            SpectralFeatures:  A SpectralFeatures object containing features that represent the 
                signal strength values.
        """
        if audio_signal is not None:
            self._compute_wiener_transfer_function(audio_signal)
        try:
            n_nonnegative_freqs = self.wiener_transfer_function.shape[0] // 2
            tf = self.wiener_transfer_function[:n_nonnegative_freqs, :]
            tf_freqs = self.stft_frequencies[:n_nonnegative_freqs]
            tf_times = self.stft_times
            return SpectralFeatures(features=tf, frequency_axis=tf_freqs, time_axis=tf_times,
                time_step=self.time_step, frame_duration=self.frame_duration)
        except AttributeError:
            raise RuntimeError(
                "An audio_signal must be supplied if one has not previously been processed yet."
            )

    def _compute_wiener_transfer_function(self, audio_signal):
        # Round the frame length to the nearest power of 2 for simpler FFTs
        self.frame_length = self._round_to_nearest_power_of_two(
            self.approximate_frame_length_seconds * audio_signal.sample_rate
        )
        self.frame_duration = self.frame_length / audio_signal.sample_rate
        self.frame_step = self.frame_length // 2
        self.time_step = self.frame_step / audio_signal.sample_rate
        # Windowing for STFT
        window_start_indices = np.arange(
            0, audio_signal.n_samples - self.frame_length + 1, self.frame_step
        )
        sliding_window = np.hstack(
            tuple(
                audio_signal.signal[idx : idx + self.frame_length] for idx in window_start_indices
            )
        )
        # Compute STFT
        # Use nfft=frame_length*2 to avoid circular convolution problems (the second half of the
        # frame will be zero padded)
        self.nfft = self.frame_length * 2
        self.analysis_window = np.hamming(self.frame_length)
        self.signal_stft = sp.fft.fft(
            sliding_window * self.analysis_window[:, np.newaxis], n=self.nfft, axis=0,
        )
        self.stft_frequencies = (
            np.arange(self.signal_stft.shape[0]) * audio_signal.sample_rate / self.nfft
        )
        # Wrap frequencies at or above the Nyquist rate to their equivalent negative frequencies
        self.stft_frequencies[self.nfft // 2 :] = (
            self.stft_frequencies[self.nfft // 2 :] - audio_signal.sample_rate
        )
        self.stft_times = window_start_indices / audio_signal.sample_rate
        # Noise PSD estimation
        self.magnitude_stft = np.abs(self.signal_stft)
        self.noise_estimate = self.noise_estimator.estimate_noise_magnitude_spectrum(
            magnitude_stft=self.magnitude_stft,
            sample_rate=audio_signal.sample_rate,
            frame_step=self.frame_step,
        )
        # Compute Wiener transfer function
        # Also use a tiny float number to prevent division of zero by zero in case there are any
        # zero values in magnitude_stft (this would introduce nans that would then spread through
        # out the rest of the processed signal).
        self.wiener_transfer_function = np.maximum(
            self.attenuation_limit,
            np.maximum(
                0,
                np.abs(self.magnitude_stft)
                - self.oversubtraction_factor * np.abs(self.noise_estimate),
            )
            ** 2
            / np.maximum(self.tiny_float, self.magnitude_stft) ** 2,
        )

    def filter_signal(self, audio_signal):
        # Estimate noise and compute a Wiener transfer function
        self._compute_wiener_transfer_function(audio_signal)
        # Smooth the transfer function (to help avoid audible artifacts from the transfer function
        # rapidly changing between different slightly-wrong values every frame).
        self.transfer_function_smoothing_kernel_length = (
            self.transfer_function_smoothing_period * audio_signal.sample_rate
        ) / self.frame_step // 2 * 2 + 1
        self.smoothed_transfer_function = self._smooth_with_kernel(
            self.wiener_transfer_function,
            np.hanning(self.transfer_function_smoothing_kernel_length),
        )
        self.output_stft = self.signal_stft * self.smoothed_transfer_function
        # Inverse STFT (construct the noise-reduced waveform)
        output_frames = sp.fft.ifft(self.output_stft, axis=0)
        self.synthesis_window = self._build_synthesis_window(self.analysis_window)
        output_frames = output_frames[: self.frame_length, :] * self.synthesis_window[:, np.newaxis]
        output_waveform = np.zeros(audio_signal.signal.shape)
        for frame_idx in range(output_frames.shape[1]):
            start_idx = frame_idx * self.frame_step
            output_waveform[start_idx : start_idx + self.frame_length] += np.real(
                output_frames[:, (frame_idx,)]
            )
        # Package result
        return AudioSignal(output_waveform, audio_signal.sample_rate)

    def _smooth_with_kernel(self, data, kernel):
        middle_idx = kernel.size // 2
        if kernel.size <= 1:
            return data
        kernel = kernel / np.sum(kernel)
        result = sp.signal.convolve2d(data, kernel[np.newaxis, :], mode="same")
        edge_scaling = 1 / (1 - np.cumsum(kernel[:middle_idx])[::-1])
        result[:, :middle_idx] = result[:, :middle_idx] * edge_scaling
        result[:, -middle_idx:] = result[:, -middle_idx:] * edge_scaling[::-1]
        return result

    def _build_synthesis_window(self, analysis_window):
        half_frame_index = np.arange(0, self.frame_step)
        # Each frame overlaps 50% with its two neighbors.  So we'll need to blend the overlapping
        # outputs from each frame.  We do this with a triangular window that is 0 at the beginning
        # and end of the frame, and peaks at 1 in the middle---thus creating a smooth linear
        # blending between the output of one frame and the next.
        linear_blend_window = np.concatenate((half_frame_index, half_frame_index[::-1])) / (
            self.frame_step - 1
        )
        # To avoid scaling some output samples more than others, we need to undo both the scaling
        # caused by the overlapping analysis windows as well as the scaling that comes from us
        # linearly fading one output frame into the next (over the overlapping portion).  Compute
        # the normalization factor for each sample in an overlapping region.
        overlapping_windows_normalization = 1 / (
            # Right half of analysis window, faded out with the right half of the linear blend /
            # fade window.
            analysis_window[self.frame_step :] * linear_blend_window[self.frame_step :]
            # Left half of analysis window, faded in with the left half of the linear blend / fade
            # window.
            + analysis_window[: self.frame_step] * linear_blend_window[: self.frame_step]
        )
        # The synthesis window for a single frame will cover two overlap regions (one for the
        # overlap with the previous frame, one for the overlap with the next frame).  Since this
        # window will be applied to each frame separately, it needs to be faded in and out using the
        # linear blending window.  Then, we can apply it to each frame's output and simply add it
        # to the overlapping outputs of the adjacent frames---and everything will be scaled
        # properly.
        synthesis_window = linear_blend_window * np.concatenate(
            (overlapping_windows_normalization, overlapping_windows_normalization)
        )
        return synthesis_window

