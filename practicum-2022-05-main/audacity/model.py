# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms. 
 
"""
Placeholder for a model that can create predited labels from an audio file.
"""
import os
from audiot.audio_labels import load_labels

def predict_labels(audiofile):
    # Find the label file in the same folder as the audio file.
    folder, filename = os.path.split(audiofile)
    name = os.path.splitext(filename)[0]
    labelfile = None
    for file in os.listdir(folder):
        if file.startswith(name) and file.endswith('.txt'):
            labelfile = os.path.join(folder,file)
    if labelfile is None:
        print('Label file not found. Check that it is in the same folder as the audio file.')

    # Load labels from existing label file. Return these as the "predictions"
    labels = load_labels(labelfile)
    return labels
