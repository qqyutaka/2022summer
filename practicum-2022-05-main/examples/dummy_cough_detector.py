# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms. 

"""
Please include a module docstring (like this one) that specifies what packages need to be installed
(and how to install them) to run your code.  We'll also appreciate if the code is commented and 
readable).

To install needed libraries, run:
    conda install library-your-code-needs
    conda install library-your-code-needs-2
    ...
    (You can assume we've already installed the ones mentioned in the PythonSetupNotes, like 
    scikit-learn and pandas).

Your module should include the following four functions that behave like the ones defined in this 
file (same name, same format of inputs / outputs).  See the docstrings in the function definitions
below for more information about how they should behave:
    cough_detector = train_cough_detector(audio_files_list, label_files_list)
    labels_dataframe = run_cough_detector(cough_detector, one_audio_file_path)
    save_cough_detector(cough_detector, file_name)
    cough_detector = load_cough_detector(file_name)
"""

import sys
from pathlib import Path

import librosa
import pandas as pd
import numpy as np
import scipy as sp
import pickle

from audiot.audio_features import AudioFeatures, calc_log_mel_energy_features
from audiot.audio_signal import AudioSignal
from audiot.audio_labels import load_labels, clean_overlapping_labels


def train_cough_detector(audio_files_list, label_files_list):
    """
    Dummy example of how your train_cough_detector function should operate.  This function just 
    "trains" a threshold that will classify any feature vector whose sum is above that threshold as 
    a cough (which is not a very good way to do it, but is simple for the purposes of
    this example).

    Args:
        audio_files_list (list of Path):  A list containing the paths (as pathlib.Path objects) to
            all of the audio files that will be used for training.
        label_files_list (list of Path):  A list containing the paths (as pathlib.Path objects) to
            the label files for each of the corresponding audio file paths in audio_files_list.  
            This list should have the same length as audio_files_list.
        
    Returns:
        Object:  An object representing the trained cough detector model.  This can be any type of
            object you want, but should match what your run_cough_detector function expects to be
            passed in as the first parameter (named cough_detector_model in this file).  For this 
            dummy example, it returns a tuple containing the threshold and the percentile value
            used to set that threshold based on the coughs seen in the training data: (threshold,
            percentile_to_use_as_threshold).
    """
    # Read in features for all the audio files and combine into one array
    print("Loading features...")
    features_by_file = []
    for audio_file, label_file in zip(audio_files_list, label_files_list):
        signal = AudioSignal.from_file(audio_file)
        features = calc_log_mel_energy_features(signal)
        features.event_names = ["cough"]
        labels = load_labels(label_file)
        features.match_labels(labels)
        features_by_file.append(features)
    training_features = AudioFeatures.concatenate(*features_by_file)
    # My dummy detector just uses a threshold and classifies any feature samples whose sum is higher
    # than that threshold as a cough (not a very good method as any sufficiently loud sound would
    # trigger it).  It "trains" the threshold by setting it to a certain percentile value amongst
    # those seen for the coughs in the training data.
    print("Training...")
    percentile_to_use_as_threshold = 90
    # Extract just the features that were labeled as coughs
    cough_mask = training_features.true_events == 1
    cough_features = training_features.features[cough_mask[:, 0], :]
    # Sum each feature vector labeled as a cough, then set the threshold to the value at the
    # specified percentile:
    cough_features_summed = np.sum(cough_features, axis=1)
    threshold = np.percentile(cough_features_summed, percentile_to_use_as_threshold)
    # Print the range of values seen and the chosen threshold value:
    print(
        f"min={np.min(cough_features_summed)}, max={np.max(cough_features_summed)}, thresh={threshold}"
    )
    print("Finished!")
    # Return whatever object(s) you need to classify new samples in your run_cough_detector
    # function.  If you need multiple things that aren't already packaged together in a single
    # object, you can package them up in a tuple by using the comma operator like below.  (I really
    # only need the threshold value to do classification, but I return which percentile I used to
    # choose the threshold as well (to demo making a tuple):
    return threshold, percentile_to_use_as_threshold


def run_cough_detector(cough_detector, audio_file):
    """
    Dummy example of how to structure the run_cough_detector() function so that it's easy for us
    to import it from your module and run it ourselves.  In this case, this function just takes the
    threshold that was trained in the function above and classifies all the feature samples whose 
    sums are above that threshold as coughs.

    Args:
        cough_detector (Object): Python Object containing your trained classifier model.  In this
            case, I just have a tuple containing the threshold and the percentile that was used
            to select that threshold value.
        audio_file (Path): A pathlib.Path object containing the path to the audio recording that 
            the cough detector should be run on.  You'll need to load this file, calculate whatever
            features you are using from it (along with the time windows that they span), classify 
            them as coughs or not, then build and return a labels_dataframe that matches format 
            you get when you read in an Audacity label file using audio_labels.load_labels().
    
    Returns:
        DataFrame: A pandas DataFrame with an "onset" column specifying the beginning time of each
            cough detected in the file (in seconds), an "offset" column specifying the ending time
            of each cough detected in the file (also in seconds), and an "event_label" column 
            containing the label for each detected event ("cough" in this case).
    """
    # Unpackage the values from the tuple I returned from my train_cough_detector function above.
    # (The percentile variable doesn't actually get used and is just unpacked as an example.
    # Typically, the convention in Python is to name variables that you're not going to use with a
    # single underscore character instead of giving them an actual name like this.  Note that that
    # will also prevent pylint from complaining about an unused variable like it does for this.)
    threshold, percentile = cough_detector
    # Read in the audio file and calculate the features I need to pass into the classifier
    signal = AudioSignal.from_file(audio_file)
    features = calc_log_mel_energy_features(signal)
    # Run my classifer on the features.  In this example code, my dummy detector just sums up the
    # log mel energy features and classifies any that meet that threshold as coughs (this is not a
    # very good method as any sufficiently loud sound would trigger a cough detection).
    features_summed = np.sum(features.features, axis=1)
    detected_coughs = threshold <= features_summed
    # Create a labels dataframe with a cough label for each frame that was classified as a cough,
    # as well as the starting and ending times (onsets and offsets) for each of those frames:
    labels_dataframe = pd.DataFrame(
        {
            "onset": pd.Series(features.frame_start_times[detected_coughs]),
            "offset": pd.Series(features.frame_end_times[detected_coughs]),
            "event_label": pd.Series("cough", index=range(np.sum(detected_coughs))),
        }
    )
    # If there are multiple cough labels in a row (for adjacent or overlapping frames), combine
    # those labels into a single longer cough label.
    labels_dataframe = clean_overlapping_labels(labels_dataframe)
    # Return the labels dataframe that now matches the format you get when loading audacity label
    # files.
    return labels_dataframe


def save_cough_detector(cough_detector, output_file_path):
    """
    Dummy example of how your save_cough_detector() function might operate to save your trained 
    cough detection model out to a file.  If there's a standard way to save out the type of model
    you're using to a file, you should use that.  Otherwise, pickle might be a good / simple option.
    """
    with open(output_file_path, "wb") as file_out:
        pickle.dump(cough_detector, file_out)


def load_cough_detector(input_file_path):
    """
    Dummy example of how your load_cough_detector() function might operate to load your trained 
    cough detection model in from a file.  If there's a standard way to save and load the type of
    model you're using, you should use that.  Otherwise, pickle might be a good / simple option.
    """
    with open(input_file_path, "rb") as file_in:
        cough_detector = pickle.load(file_in)
    return cough_detector

