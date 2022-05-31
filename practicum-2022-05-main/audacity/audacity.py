# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms.

import os

from audacity.pipeutils import *
from audiot.audio_labels import load_labels
from audacity import pipeutils
import pandas as pd

def connect():
    pipe_setup()

def create_project(audiofile, projfolder=None):
    """
    Create a new Audacity project and import an audio file. Save project to specified folder and use
    the same name as the source audio file. If destination folder is not specified, save to same
    location as the source audio. Before running this function, ensure that Audacity is open and
    mod-script-pipe is enabled.
    """
    if projfolder is None:
        projfolder = os.path.split(audiofile)[0]
    if not os.path.exists(projfolder):
        os.makedirs(projfolder)
    # Create new project
    do_command('New:')
    # Add audio file to project
    do_command('Import2: Filename="'+audiofile+'"')
    # Save Audacity project with same name as audio file in specified project folder
    name = os.path.splitext(os.path.basename(audiofile))[0]
    projfile = os.path.join(projfolder, name + '.aup')
    if os.path.isfile(projfile):
        print('Project file already exists. Overwriting.')
        os.remove(projfile)
    do_command('SaveProject2: Filename="'+projfile+'"')

def open_project(filename):
    """
    Open specified Audacity project with given filename. Before running this function, ensure that
    Audacity is open and mod-script-pipe is enabled.
    Args:
        filename: audacity project file specified
    """
    do_command('OpenProject2:Filename="'+filename+'"')

def add_labels(labels):
    """
    Add labels to current Audacity project from 'labels' dataframe. Before running this function,
    ensure that Audacity is open and mod-script-pipe is enabled.
    """
    # Select region at time 0 so all new labels have index 0 when created
    do_command('Select:End="0" Mode="Set" RelativeTo="ProjectStart" Start="0"')
    # Iteratively create a new label with 'index' 0 and update its details
    for _,row in labels.iterrows():
        do_command('AddLabel:')
        label_command = ('SetLabel: Label="0"' +
              ' Text="' + row['event_label'] + '"' +
              ' Start=' + str(row['onset']) +
              ' End=' + str(row['offset']))
        do_command(label_command)
    do_command('Save:')

def import_labels_from_file(labelfile):
    """
    Get labels from specified labels file and use them to populate the labels of current Audacity
    project. Before running this function, ensure that Audacity is open and mod-script-pipe is
    enabled.

    Args:
        labelfile: specified labels to import to audacity project

    """
    labels = load_labels(labelfile)
    add_labels(labels)

def get_labels(onset_col="onset", offset_col="offset", label_col="event_label"):
    """
    Function to retrieve the labels from an audacity project, the labels are put into a dataframe.
    Args:
        onset_col: start time of specific label
        offset_col: end time of label entry
        label_col: label name of entry

    Returns:
        dataframe of label data

    """
    import json
    response = do_command('GetInfo:Type=Labels')
    json_data = response[:response.rfind(']')+1] #JSON is everything up to the last ']'
    labels_list = json.loads(json_data)[0][1]
    labels_data = pd.DataFrame(labels_list, columns=[onset_col, offset_col, label_col])
    return labels_data

def save_and_exit():
    """
    Save and close the current project.
    """
    do_command('Save:')
    send_command('Exit:')

def close():
    """
     close the current project.
    """
    do_command('Close:')
