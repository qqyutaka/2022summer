# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms.

"""
This is example code of how we would like to be able to easily train and run your cough detector.
We should be able to import the "train_cough_detector", "run_cough_detector", "save_cough_detector",
and "load_cough_detector" functions from your python module (like I do from the
examples.dummy_cough_detector module below), install any extra packages if needed, and then run code
like the sample code below without needing to modify it.

Your train_cough_detector function should take in a list of paths to audio files and a list of
paths to their corresponding label files, and should return a cough detector object/model freshly
trained on that data.

Your run_cough_detector function should take in the same object/model returned by your
train_cough_detector function and a path to a single audio file, and it should return a labels
DataFrame containing the starting time, ending time, and label for each of the detected coughs.

Your save_cough_detector and load_cough_detector functions should save your cough detector object/
model out to a file and load it in from a file, respectively.

The code below trains a not-very-good cough detector that simply sums up the log mel energy features
and compares the result to a threshold to classify it as a cough or not.  After training, it then
runs that detector on all the same audio files and writes out label files containing all the 
detected coughs into a "DummyCoughLabels" folder (which will be created if it doesn't exist).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import scipy as sp

# We should be able to just change the name of the module below to the name of your module, and then
# run this code (assuming we've installed the necessary packages).  E.g.:
#   from your_folder.your_module import train_cough_detector, run_cough detector, save_cough_detector, load_cough_detector
from examples.dummy_cough_detector import (
    train_cough_detector,
    run_cough_detector,
    save_cough_detector,
    load_cough_detector,
)
from audiot.audio_labels import save_labels

# Find all the labeled audio files
data_folder = Path(__file__).absolute().parents[1] / "test_data"
audio_files = list(data_folder.rglob("*.flac"))
# Construct the paths to the corresponding label files by replacing ".flac" with "_label.txt"
label_files = [Path(str(af).replace(".flac", "_label.txt")) for af in audio_files]

# Train the detector on the training data
cough_detector_model = train_cough_detector(audio_files, label_files)

# Test saving and loading the detector
model_file_path = Path("saved_cough_detector.pickle")
save_cough_detector(cough_detector_model, model_file_path)
del cough_detector_model
cough_detector_model = load_cough_detector(model_file_path)

# Run the detector on each audio file and write out the detected coughs as Audacity label files.
print("Classifying files", end="")
output_folder = Path("DummyCoughLabels")
output_folder.mkdir(exist_ok=True)
for audio_file in audio_files:
    print(".", end="")
    # Run the detector on the audio file.  The labels DataFrame returned here should match the
    # formatting of the DataFrames read in by the load_labels() function in audiot\audio_labels.py
    # (e.g. it should have columns "onset", "offset", and "event_label").  We'll compute performance
    # values by comparing the labels returned here against cleaned human label files.
    labels_dataframe = run_cough_detector(cough_detector_model, audio_file)
    # Write out the detected labels as an Audacity-formatted label file.
    output_file = output_folder / audio_file.name.replace(".flac", "_dummy-labels.txt")
    save_labels(labels_dataframe, output_file)
# Print a newline after the above print statements that did not include a newline.
print("")

print("Done!")
