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
from audiot.label_checker import LabelChecker


default_allowed_labels = ["cough"]

parser = argparse.ArgumentParser(description="Check label files for common problems")
parser.add_argument("input_folder", help="Path to folder containing the label files to be checked")
parser.add_argument(
    "file_glob",
    nargs="?",
    default="*.txt",
    help="Glob pattern that determines which files are checked (defaults to *.txt)",
)
parser.add_argument(
    "-r",
    "--recursive",
    help="Recursively traverse the subdirectories of labels_folder",
    action="store_true",
)
parser.add_argument(
    "-validlabels",
    nargs="+",
    default=default_allowed_labels,
    help='Valid label names. Default is "cough" only.',
)

args = parser.parse_args()
input_folder = Path(args.input_folder)
file_glob = args.file_glob
recursive = args.recursive
allowed_labels = args.validlabels


checker = LabelChecker(allowed_labels)

if recursive:
    file_list = list(input_folder.rglob(file_glob))
else:
    file_list = list(input_folder.glob(file_glob))
n_problem_files = 0
for file_path in file_list:
    if not checker.check_label_file(file_path):
        n_problem_files += 1

print("=" * 80)
print(f"Number of files processed: {len(file_list)}")
if n_problem_files > 0:
    print(f"ERRORS DETECTED in {n_problem_files} files.")
    print("Please correct the problems in the files listed above.")
else:
    print("Done!  No problems detected in the label files.")
