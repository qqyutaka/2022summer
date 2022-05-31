# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms. 

import sys
from pathlib import Path

import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp

from audiot.audio_features import AudioFeatures, calc_log_mel_energy_features
from audiot.audio_signal import AudioSignal
from audiot.audio_labels import load_labels


if __name__ == "__main__":
    # Test reading in an audio file, calculating features, and associating cough labels
    project_folder = Path(__file__).absolute().parents[1]
    file_path = project_folder / "test_data/LT3-G3_2014-11-05_15.58.00_ch2.flac"
    label_path = project_folder / "test_data/LT3-G3_2014-11-05_15.58.00_ch2_label.txt"
    print(f"Loading {file_path}")
    signal = AudioSignal.from_file(file_path)
    features = calc_log_mel_energy_features(signal)
    features.event_names = ["cough"]
    labels = load_labels(label_path)
    print(labels)
    features.match_labels(labels)

    # Stack the label data on top of the features and plot
    min_val = np.amin(features.features)
    max_val = np.amax(features.features)
    plot_data = np.vstack(
        (features.features.T, features.true_events.T * (max_val - min_val) + min_val)
    )
    plt.figure()
    plt.imshow(
        plot_data,
        interpolation="nearest",
        aspect="auto",
        origin="lower",
        extent=(features.frame_start_times[0], features.frame_end_times[-1], 0, plot_data.shape[0]),
    )
    plt.title("Log Mel energy spectrum")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency band")
    plt.colorbar()

    # Example of computing deltas and delta-deltas and adding them to your stack of features.
    # The n_samples_per_side parameter impacts how large of a time window the derivatives (deltas)
    # are estimated over.  Note that the scaling of the deltas tends to be different from the 
    # scaling of the original features (and ditto with the delta-deltas).
    deltas = features.deltas(n_samples_per_side=2)
    delta_deltas = deltas.deltas(n_samples_per_side=2)
    features_with_deltas = AudioFeatures.stack(features, deltas, delta_deltas)
    features_with_deltas.match_labels(labels)

    # Stack the label data on top of the features and plot
    min_val = np.amin(features_with_deltas.features)
    max_val = np.amax(features_with_deltas.features)
    plot_data = np.vstack(
        (
            features_with_deltas.features.T,
            features_with_deltas.true_events.T * (max_val - min_val) + min_val,
        )
    )
    plt.figure()
    plt.imshow(
        plot_data,
        interpolation="nearest",
        aspect="auto",
        origin="lower",
        extent=(
            features_with_deltas.frame_start_times[0],
            features_with_deltas.frame_end_times[-1],
            0,
            plot_data.shape[0],
        ),
    )
    plt.title("Log Mel energy features with deltas and delta-deltas")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency band")
    plt.colorbar()

    plt.show()
