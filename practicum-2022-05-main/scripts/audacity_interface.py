# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms. 

import sys
from pathlib import Path

# Add the main project directory (parent folder) to sys.path so we can import our project modules
project_folder = Path(__file__).absolute().parent.parent
sys.path.append(str(project_folder))

import argparse
import os
from audacity import workflow

parser = argparse.ArgumentParser(description="Build audacity projects from audio and optional labelled data."
                                             "Also export labels from audacity projects.")
parser.add_argument("input_folder", help="Path to folder/file containing audio files to be packed into .aup")
parser.add_argument(
    "-e",
    "--export",
    help="indicates the folder/file indicated will be used for exporting labels.",
    action="store_true"
)
parser.add_argument(
    "-l",
    help="indicate whether labels for file(s) are present. (only used for building .aup)",
    action="store_true"
)
parser.add_argument(
    "-o",
    "--outfolder",
    # nargs='?',
    # const='arg_was_not_given',
    help="Path to where you want built projects/extracted labels to be saved."
         "defaulted to same folder as"
)

args = parser.parse_args()
input_folder = Path(args.input_folder).resolve()
exporting = args.export
labels_present = args.l

if args.outfolder is None:
    if str(input_folder).endswith(".aup") or str(input_folder).endswith(".flac")\
            or str(input_folder).endswith(".aup3"):
        out_folder = input_folder.parent
    else:
        out_folder = input_folder
else:
    out_folder = Path(args.outfolder).resolve()


if exporting:
    if str(input_folder).endswith(".aup") or str(input_folder).endswith(".aup3"):
        input_file = str(input_folder)
        out_folder = str(out_folder)
        print(out_folder)
        workflow.export_labels_from_file(input_file, out_folder)
    else:
        input_folder = str(input_folder)
        out_folder = str(out_folder)
        workflow.export_labels_from_folder(input_folder, out_folder)
else:
    if str(input_folder).endswith(".flac"):
        input_file = str(input_folder)
        out_folder = str(out_folder)
        workflow.single_audacity_project_from_file(input_file, out_folder, labels_present)
    else:
        input_folder = str(input_folder)
        out_folder = str(out_folder)
        workflow.audacity_projects_from_folder(input_folder, out_folder, labels_present)

print("Successful!")