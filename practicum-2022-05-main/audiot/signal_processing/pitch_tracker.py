import pickle
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal
import random
from pathlib import Path
from audiot.audio_signal import AudioSignal
from audiot.signal_processing.wiener_filter import WienerFilter
from audiot.signal_processing.auto_segmenter import AutoSegmenter


class PitchTracker:
    def __init__(self, kernel_size=9, shift_cost_multiplier=0.01, shift_cost_exponent=2):
        if kernel_size < 3:
            raise RuntimeError("kernel_size must be 3 or greater.")
        step_size = 2 * np.pi / (kernel_size - 1)
        kernel = np.cos(np.arange(-np.pi, np.pi + step_size, step_size))
        kernel = kernel - (sum(kernel) / kernel_size)
        self.kernel = kernel[::-1, np.newaxis]
        self.shift_cost_multiplier = shift_cost_multiplier
        self.shift_cost_exponent = shift_cost_exponent

    def track_pitch_across_segment(self, segment_signal_strength_spectrogram):
        # TODO: Add additional reward for harmonics (via matrix multiplication on rewards matrix)
        n_freqs, n_times = segment_signal_strength_spectrogram.shape
        # Matrix to track the best previous neighbor to connect to for maximum reward for every 
        # point in the grid.  Used to construct the path with maximum reward at the end by walking
        # backwards through the best predecessors (dynamic programming algorithm).
        best_predecessor = np.zeros(segment_signal_strength_spectrogram.shape, dtype=int)
        base_rewards = sp.signal.convolve2d(
            segment_signal_strength_spectrogram, self.kernel, mode="same"
        )
        accumulated_rewards = base_rewards.copy()
        freq_axis = np.arange(n_freqs)
        for time_idx in range(1, n_times):
            for freq_idx in range(n_freqs):
                best_predecessor[freq_idx, time_idx] = np.argmax(
                    accumulated_rewards[:, time_idx - 1]
                    - np.abs(freq_idx - freq_axis) ** self.shift_cost_exponent
                    * self.shift_cost_multiplier
                )
                accumulated_rewards[freq_idx, time_idx] += (
                    accumulated_rewards[best_predecessor[freq_idx, time_idx], time_idx - 1]
                    - np.abs(freq_idx - best_predecessor[freq_idx, time_idx])
                    ** self.shift_cost_exponent
                    * self.shift_cost_multiplier
                )
        reverse_path = [np.argmax(accumulated_rewards[:, -1])]
        for time_idx in range(n_times - 1, 0, -1):
            reverse_path.append(best_predecessor[reverse_path[-1], time_idx])
        pitch_indexes = reverse_path[::-1]
        peak_strengths = [
            base_rewards[pitch_indexes[t], t] for t in range(accumulated_rewards.shape[1])
        ]
        return pitch_indexes, peak_strengths
    
    @classmethod
    def pitch_indexes_to_frequencies(cls, pitch_indexes, frequency_axis):
        return [frequency_axis[idx] for idx in pitch_indexes]


def test_pitch_tracking_on_file(file_path, max_plots_per_file=None):
    print(file_path)
    audio_signal = AudioSignal.from_file(file_path)

    pitch_tracker = PitchTracker(kernel_size=9)
    wiener_filter = WienerFilter.get_very_aggressive_noise_reduction_filter()
    auto_segmenter = AutoSegmenter.get_default_segmenter()

    # Get signal strength features and run segmentation
    (
        features,
        freqs,
        times,
        time_step,
        window_duration,
    ) = wiener_filter.get_signal_strength_features(audio_signal)
    segments = auto_segmenter.segment_signal_strength_features(
        features, freqs, times, time_step, window_duration
    )

    # Get a spectrogram too so we can plot that if we want
    (spectrogram_frequencies, spectrogram_times, spectrogram) = scipy.signal.spectrogram(
        audio_signal.signal[:, 0],
        fs=audio_signal.sample_rate,
        nperseg=wiener_filter.frame_length,
        noverlap=wiener_filter.frame_length - wiener_filter.frame_step,
        nfft=wiener_filter.frame_length * 2,
    )
    spectrogram = np.log(spectrogram)

    # Run pitch tracking on segments and make plots
    if max_plots_per_file and len(segments) > max_plots_per_file:
        segments = random.sample(segments, max_plots_per_file)
    plot_count = 0
    context = 5
    for start_idx, end_idx, signal_strength in segments:
        segment_features = features[:, start_idx:end_idx]
        segment_spectrogram = spectrogram[
            :, max(0, start_idx - context) : min(spectrogram.shape[1], end_idx + context)
        ]
        pitch_track, pitch_strength = pitch_tracker.track_pitch_across_segment(segment_features)
        plt.figure()
        plt.imshow(segment_features, aspect="auto", origin="lower", interpolation="none")
        # plt.imshow(segment_spectrogram, aspect="auto", origin="lower", interpolation="none")
        plt.colorbar()
        plt.plot(pitch_track, color="red", linewidth=1)
        pitch_strength_multiplier = segment_features.shape[0] / 4
        plt.plot(np.array(pitch_strength) * pitch_strength_multiplier, color="yellow", linewidth=1)
        plot_count += 1

        plt.figure()
        plt.imshow(segment_spectrogram, aspect="auto", origin="lower", interpolation="none")
        plot_count += 1
        if plot_count >= 20:
            plt.show()
            plot_count = 0
    if plot_count >= 0:
        plt.show()
    print("Done!")


if __name__ == "__main__":
    # input_folder = Path("test_data")
    # file_path = input_folder / "TRF0_mic14_2020-12-17_01.20.00.flac"
    input_folder = Path("D:/phdsync/TrainingData/TestFiles")
    input_files = list(input_folder.glob("*.flac"))
    random.shuffle(input_files)
    for input_file in input_files:
        test_pitch_tracking_on_file(input_file, max_plots_per_file=20)


#    input_file = "peak_tracking/sample_tfs_17.29.53.pickle"
#    # input_file = "peak_tracking/sample_tfs_17.29.55.pickle"
#    # input_file = "peak_tracking/sample_tfs_17.29.59.pickle"
#    # input_file = "peak_tracking/sample_tfs_17.30.00.pickle"
#    with open(input_file, "rb") as file_in:
#        sample_tfs = pickle.load(file_in)
#    n_samples = len(sample_tfs)
#    print(f"n_samples = {n_samples}")
#    selected_samples = random.sample(range(n_samples), 20)
#    for idx in selected_samples:
#        tf = sample_tfs[idx]
#        peak_index_list = track_peak_across_segment(tf)
#        plt.figure()
#        plt.imshow(tf, aspect="auto", origin="lower", interpolation="none")
#        plt.plot(peak_index_list, color="red", linewidth=0.5)
#        plt.title(f"Sample {idx}")
#    plt.show()
#    print("done!")
#
