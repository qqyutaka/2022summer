# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms. 

"""
Functions for running the Audacity automation workflows that prepare Audacity projects for manual
labeling.
"""
import pandas as pd

audacity_file = 'C:\\Program Files (x86)\\Audacity\\audacity.exe'  # THIS IS BASED ON DEFAULT INSTALL


def single_audacity_project_from_file(audiofile, projectfolder=None, predictlabels=False):
    """
    Open Audacity, create a single project from a specified file, save the project, and close
    Audacity. If predicting labels, the labels must be in the same folder as the audio file.
    Args:
        audiofile: flac file to be saved into a project
        projectfolder: Folder to save the project, DEFAULT = working directory
        predictlabels: Boolean that indicates
    """
    import os, time
    from audacity import model
    from audacity.audacity import connect, create_project, add_labels

    # Launch Audacity
    os.startfile(audacity_file)
    time.sleep(4)  # Give Windows time to open the application or the subsequent commands will fail

    # Connect to Audacity
    connect()

    # Create project
    create_project(audiofile, projectfolder)

    # Optionally, predict labels and populate project with predictions
    if predictlabels:
        labels = model.predict_labels(audiofile)
        add_labels(labels)

    # Save current project and close all Audacity windows (not working)
    #save_and_exit()


def audacity_projects_from_folder(audiofolder, projectfolder=None, predictlabels=False):
    """
    Open Audacity, create a series of projects for files in a directory. If adding labels to projects, labels must be
    in same folder as audio files.
    Args:
        audiofolder: folder of audio files
        projectfolder: folder to save projects to
        predictlabels: boolean indicating whether labels are to be added Default = False (no Labels)
    """
    import os, time
    from audacity import model
    from audacity.audacity import connect, create_project, add_labels, save_and_exit

    # Launch Audacity
    os.startfile(audacity_file)
    time.sleep(5)  # Give Windows time to open the application or the subsequent commands will fail
    # Create project

    # Connect to Audacity
    connect()

    filelist = [file for file in os.listdir(audiofolder) if file.endswith('.flac')]
    file_counter = 0
    for file in filelist:

        audiofile = os.path.join(audiofolder, file)
        create_project(audiofile, projectfolder)

        # Optionally, predict labels and populate project with predictions
        if predictlabels:
            labels = model.predict_labels(audiofile)
            add_labels(labels)
        file_counter += 1
        if file_counter % 7 == 0:
            save_and_exit()
            time.sleep(5)
            # Launch Audacity
            os.startfile(audacity_file)
            time.sleep(5)  # Give Windows time to open the application or the subsequent commands will fail
            # Create project
            connect()
    # Save current project and close all Audacity windows
    save_and_exit()


def export_labels_from_file(projectfile, outputfolder=None):
    """
    Export labels of Audacity Projects in a folder.
    Args:
        projectfile: Audacity project file Labels will be extracted from.
        outputfolder: Folder where the exported labels will go. Defaulted to *None*

    Returns:
        Pandas Dataframe of labels loaded into the project. Also exports the label out to folder specified.
    """
    import os, time
    from audacity import model
    from audacity.audacity import connect, open_project, get_labels, save_and_exit
    from pathlib import Path

    # Launch Audacity
    os.startfile(audacity_file)
    time.sleep(5)  # Give Windows time to open the application or the subsequent commands will fail
    # Connect to audacity
    connect()
    # Open Project
    open_project(projectfile)

    df = get_labels()
    if outputfolder is None:
        out_to = projectfile[0:len(projectfile) - 4] + "_label.txt"
    else:
        file_name = Path(projectfile).name
        file_name = file_name[0:len(file_name) - 4] + "_label.txt"
        out_to = os.path.join(outputfolder, file_name)
    df.to_csv(out_to, sep='\t', index=False, header=False)
    return df
    # Save current project and close all Audacity windows
    save_and_exit()


def export_labels_from_folder(projectfolder, outputfolder=None):
    """
    Export labels of Audacity Projects in a folder. Saves those labels to .txt files with _label tag.
    Args:
        projectfolder: Folder where the audacity projects are located.
        outputfolder: Folder where the exported labels will go. Defaulted to *None*

    Returns:
        Text files of labels
    """
    import os, time
    from pathlib import Path
    from audacity import model
    from audacity.audacity import connect, open_project, get_labels, save_and_exit

    # Launch Audacity
    os.startfile(audacity_file)
    time.sleep(5)  # Give Windows time to open the application or the subsequent commands will fail
    # Connect
    connect()

    filelist = [file for file in os.listdir(projectfolder) if file.endswith('.aup') or file.endswith('.aup3')]
    print(filelist)
    file_counter = 0
    for file in filelist:
        projectfile = os.path.join(projectfolder, file)
        open_project(projectfile)
        print("opened!")
        if outputfolder is None:
            out_to = file[0:len(file) - 4] + "_label.txt"
        else:
            file_name = Path(file).name
            file_name = file_name[0:len(file_name) - 4] + "_label.txt"
            out_to = os.path.join(outputfolder, file_name)
            try:
                df = get_labels()
                df.to_csv(out_to, sep='\t', index=False, header=False)
            except:
                print("Warning: No labels for " + file + "..... The project likely has no labels.")
                df = pd.DataFrame()
                df.to_csv(out_to, sep='\t', index=False, header=False)
        file_counter += 1
        if file_counter % 4 == 0:
            save_and_exit()
            time.sleep(3)
            # Launch Audacity
            os.startfile(audacity_file)
            time.sleep(3)  # Give Windows time to open the application or the subsequent commands will fail
            # Create project
            connect()
    # Save current project and close all Audacity windows
    save_and_exit()
