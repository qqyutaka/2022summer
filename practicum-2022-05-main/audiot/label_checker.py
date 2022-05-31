# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms. 

from audiot.audio_labels import load_labels
import os


class LabelChecker:
    """
    This class contains functionality to check label files for common errors.
    """

    def __init__(self, allowed_labels, allow_questionmarks=True):
        """
        Creates a LabelChecker object.

        Args:
            allowed_labels (List of str): A list of all the valid label names.
            allow_questionmarks(bool): Whether or not it is valid to append a question mark to the
                end of an otherwise valid label name.
        """
        self.allowed_labels = [label_name.strip().lower() for label_name in allowed_labels]
        self.allow_questionmarks = allow_questionmarks
        if self.allow_questionmarks:
            self.allowed_labels.extend([label + "?" for label in self.allowed_labels])

    def locate_zero_length_labels(self, labels):
        """
        Returns the subset of the labels dataframe where the label duration is zero.
        """
        label_durations = labels["offset"] - labels["onset"]
        zero_length_labels = labels[label_durations == 0]
        return zero_length_labels

    def locate_empty_labels(self, labels):
        """
        Returns the subset of the labels dataframe where the event_label is empty.
        """
        empty_labels = labels[labels["event_label"] == ""]
        return empty_labels

    def locate_invalid_labels(self, labels):
        """
        Returns the subset of the labels dataframe where the event_label does not match one of the
        strings enumerated in self.allowed_labels.
        """
        invalid_indices = []
        for i, name in enumerate(labels["event_label"]):
            if not name.lower().strip() in self.allowed_labels:
                invalid_indices.append(i)
        invalid_labels = labels.iloc[invalid_indices]
        return invalid_labels

    def check_label_file(self, label_file_path):
        """
        Checks the specified file in the following ways, and returns True if all checks pass:
            1. Check if it can be successfully read in by the load_labels() function.
            2. Check that there are no zero-length labels.
            3. Check that there are no empty labels.
            4. Check that all the labels are in allowed_labels.
        If any check fails, False is returned and messages are printed listing the file name and
        all the items that failed each check.
        """
        try:
            labels = load_labels(label_file_path)
        except:
            print("=" * 80)
            print(f"ERROR: unable to read / parse file:\n  {label_file_path}")
            return False
        zero_length_labels = self.locate_zero_length_labels(labels)
        empty_labels = self.locate_empty_labels(labels)
        invalid_labels = self.locate_invalid_labels(labels)
        if zero_length_labels.empty and empty_labels.empty and invalid_labels.empty:
            return True
        print("=" * 80)
        print(f"ERROR in {label_file_path}:")
        if not zero_length_labels.empty:
            print("The following zero-length labels are not allowed:")
            print(zero_length_labels)
        if not empty_labels.empty:
            print("The following empty labels are not allowed:")
            print(empty_labels)
        if not invalid_labels.empty:
            print("The following label names are not in the accepted set of labels:")
            print(invalid_labels)
        return False

