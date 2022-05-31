# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms. 

import numpy as np
import scipy as sp
import pandas as pd
from pathlib import Path


def load_labels(label_file, onset_col="onset", offset_col="offset", label_col="event_label"):
    """
    Loads a label file (in the format exported by Audacity) and returns it as a Pandas DataFrame.

    This function also does some cleaning (removes frequency values if exported by audacity and 
    merges identical overlapping labels), and ensures the timing values are numeric.

    Args:
        label_file (str): Path to a label file following the Audacity format.
        onset_col (str): The name that will be assigned to the column representing the start time of
            each label in the output DataFrame.
        offset_col (str): The name that will be assigned to the column representing the end time of
            each label in the output DataFrame.
        label_col (str): The name that will be assigned to the column representing the text of 
            each label in the output DataFrame.

    Returns:
        DataFrame: A Pandas DataFrame containing the cleaned label data.
    """
    label_data = pd.read_csv(
        label_file, sep="\t", header=None, names=[onset_col, offset_col, label_col]
    )
    # Clean out frequency data if any
    label_data = label_data[label_data[onset_col] != "\\"]
    # Ensure numeric columns are numeric (and not strings)
    label_data[onset_col] = pd.to_numeric(label_data[onset_col])
    label_data[offset_col] = pd.to_numeric(label_data[offset_col])
    # Fill empty labels with empty string
    label_data[label_col] = label_data[label_col].fillna("").apply(str)
    # Clean up any overlapping, identical labels
    label_data = clean_overlapping_labels(label_data, onset_col, offset_col, label_col)
    return label_data


def save_labels(labels_dataframe, output_file, onset_col="onset", offset_col="offset", label_col="event_label"):
    """
    Saves out a label file in the exact same format as label files exported from Audacity.

    Args:
        labels_dataframe (DataFrame): A Pandas DataFrame object containing the labels to save out.
        output_file (Path-like): The path to the file to save out, usually with a ".txt" extension.
        onset_col (str): The name of the column in labels_dataframe that contains the start time of
            each label.
        offset_col (str): The name of the column in labels_dataframe that contains the end time of 
            each label.
        label_col (str): The name of the column in labels_dataframe that contains the text of each
            label.
    """
    labels_dataframe.to_csv(
        output_file,
        columns=[onset_col, offset_col, label_col],
        sep="\t",
        header=False,
        index=False,
        float_format="%.6f",
    )


def clean_overlapping_labels(
    label_data, onset_col="onset", offset_col="offset", label_col="event_label"
):
    """
    Examines label_data and merges any identical, adjacent labels that overlap (or start and end at
    the exact same time) into a single label.
    """
    labels = label_data.sort_values(onset_col)

    event_types = np.unique(labels[label_col])
    for event_type in event_types:
        event_indices = labels[labels[label_col] == event_type].index
        for idx in range(len(event_indices) - 2, -1, -1):
            if (
                labels.loc[event_indices[idx], "offset"]
                >= labels.loc[event_indices[idx + 1], "onset"]
            ):
                # Merge the two labels
                labels.loc[event_indices[idx], "offset"] = max(
                    labels.loc[event_indices[idx], "offset"],
                    labels.loc[event_indices[idx + 1], "offset"],
                )
                # Remove the extra label
                labels = labels.drop(event_indices[idx + 1])

    return labels


def merge_labels(
    labels1,
    labels2,
    onset_col="onset",
    offset_col="offset",
    label_col="event_label",
    intersect_labels=True,
):
    """
    Merges labels from two different sources labeling the same data, and returns the result.

    Note that this method assumes that there are no overlapping, identical labels in the label 
    data being merged.  Use the clean_overlapping_labels on the data from each file first to 
    ensure that.

    Args:
        labels1 (DataFrame): A Pandas DataFrame containing the label data from the first source.
            This includes the starting time, ending time, and label string for each label.
        labels2 (DataFrame): A Pandas DataFrame containing the label data from the second 
            source.
        onset_col (string): The name of the column containing the starting time for each label.
        offset_col (string): The name of the column containing the ending time for each label.
        label_col (string): The name of the column containing the label strings.
        intersect_labels (bool): If true, the merge operation is an intersection (or logical 
            AND operation).  So the results will only contain labels where both sources agreed
            on the label.  If false, the merge operation is a union (or logical OR operation).
            So the results will contain a label if either of the two sources had it.  Defaults
            to True.
    
    Returns:
        DataFrame: A dataframe containing the merged labels and using the same column names 
            specified in the args.
    """
    if not intersect_labels:
        raise RuntimeError("Union merging not supported yet.")
    event_types = np.unique(pd.concat([labels1[label_col], labels2[label_col]]))
    merged_labels = pd.DataFrame(columns=[onset_col, offset_col, label_col])
    for event_type in event_types:
        events1 = labels1[labels1[label_col] == event_type].sort_values(onset_col)
        n_events1 = events1.shape[0]
        events2 = labels2[labels2[label_col] == event_type].sort_values(onset_col)
        n_events2 = events2.shape[0]
        events1_idx = 0
        events2_idx = 0
        while events1_idx < n_events1 and events2_idx < n_events2:
            # Compute overlap between these two labels
            onset1 = events1[onset_col].iloc[events1_idx]
            offset1 = events1[offset_col].iloc[events1_idx]
            onset2 = events2[onset_col].iloc[events2_idx]
            offset2 = events2[offset_col].iloc[events2_idx]
            overlap = min(offset1, offset2) - max(onset1, onset2)
            if overlap > 0:
                merged_labels = merged_labels.append(
                    {
                        label_col: event_type,
                        onset_col: max(onset1, onset2),
                        offset_col: min(offset1, offset2),
                    },
                    ignore_index=True,
                )
            # Everything should now be merged up through the earliest of the two label end times.
            # So increment that index
            if offset1 <= offset2:
                events1_idx += 1
            else:
                events2_idx += 1
    return merged_labels


def tally_multiple_labelings(
    labels_list,
    onset_col="onset",
    offset_col="offset",
    label_col="event_label",
    label_weight=1.0,
    uncertain_label_weight=0.5,
):
    """
    Merges multiple different people's labels for the same recording together into a single 
    DataFrame that lists how many people agreed on a label along with the starting and ending time
    for that consensus. Labels with questionmarks (eg "cough?") are counted with the weight 
    specified by uncertain_label_weight (half weight by default).

    Note that this method assumes that there are no overlapping, identical labels in the label 
    data from any one person (if there is, it will count as multiple votes from a single person).  
    Use the clean_overlapping_labels on the label data from each person first to ensure this is not 
    the case.  However, it is OK for different labels to overlap (e.g. a 'cough' label and a 'rale' 
    label) within a single label file.

    Args:
        labels_list (list of DataFrame): A list of Pandas DataFrame objects containing the label 
            data from each source.  This includes the starting time, ending time, and label string 
            for each label.
        onset_col (str): The name of the column containing the starting time for each label.
        offset_col (str): The name of the column containing the ending time for each label.
        label_col (str): The name of the column containing the label strings.
        label_weight (float): The weighting given to labels without a questionmark appended.
        uncertain_label_weight (float): The weighting given to labels with a questionmark appended
            to denote uncertainty.
    
    Returns:
        DataFrame: A dataframe containing the merged labels with an additional count column 
            specifying how many labelers agreed on the label.
    """
    # Combine the labels from all labelers into one big DataFrame
    all_labels = pd.concat(labels_list)
    # Strip questionmarks to avoid including uncertain labels as a separate label type
    event_types = np.unique(all_labels[label_col].str.rstrip("?"))
    n_voters = len(labels_list)
    # Specify column names so we don't get errors when there are no labels present
    all_tallied_labels_df = pd.DataFrame(
        columns=[onset_col, offset_col, label_col, "votes", "n_voters", "consensus"]
    )
    for event_type in event_types:
        # Get all labels (both certain and uncertain) for this event type
        this_event_labels = all_labels[
            (all_labels[label_col] == event_type) | (all_labels[label_col] == event_type + "?")
        ]
        # Create DataFrames for the label onset times (when the vote tally should be incremented to
        # count that vote) and label offset times (when the vote tally should be decremented) with
        # the appropriate increment / decrement weights (based on whether the label was marked as
        # uncertain or not).
        onset_df = pd.DataFrame(this_event_labels, columns=[onset_col, "increment"])
        onset_df.increment = label_weight
        onset_df.loc[
            this_event_labels[label_col] == event_type + "?", "increment"
        ] = uncertain_label_weight
        onset_df.rename(columns={onset_col: "time"}, inplace=True)
        offset_df = pd.DataFrame(this_event_labels, columns=[offset_col, "increment"])
        offset_df.increment = -label_weight
        offset_df.loc[
            this_event_labels[label_col] == event_type + "?", "increment"
        ] = -uncertain_label_weight
        offset_df.rename(columns={offset_col: "time"}, inplace=True)
        # Do a cumulative sum of the vote increment and decrements over time to get the total vote
        # tally for each time segment.
        increment_df = onset_df.append(offset_df, ignore_index=True)
        increment_df.sort_values(by=["time"], inplace=True)
        increment_df["tally"] = increment_df["increment"].cumsum()
        # Build a dataframe with the time segments and corresponding tallies for this event.
        # Note: converting to a list gets rid of the index so that things get assigned positionally
        tallied_labels_df = pd.DataFrame(increment_df["time"][:-1].tolist(), columns=["time"])
        tallied_labels_df.rename(columns={"time": onset_col}, inplace=True)
        tallied_labels_df[offset_col] = increment_df["time"][1:].tolist()
        tallied_labels_df[label_col] = event_type
        tallied_labels_df["votes"] = increment_df["tally"][:-1].tolist()
        tallied_labels_df["n_voters"] = n_voters
        tallied_labels_df["consensus"] = tallied_labels_df["votes"] / n_voters
        # Remove rows with zero votes (where no one labeled anything)
        tallied_labels_df = tallied_labels_df[tallied_labels_df["votes"] != 0]
        # Append the tally for this event type to the overall tally for all event types
        all_tallied_labels_df = all_tallied_labels_df.append(tallied_labels_df, ignore_index=True)
    # Sort the overall tally by onset time and regenerate its index
    all_tallied_labels_df.sort_values(by=[onset_col], inplace=True)
    all_tallied_labels_df.reset_index(drop=True, inplace=True)
    return all_tallied_labels_df


def combine_multiple_labelings(
    labels_list,
    consensus_threshold=None,
    vote_threshold=None,
    onset_col="onset",
    offset_col="offset",
    label_col="event_label",
    label_weight=1.0,
    uncertain_label_weight=0.5,
):
    """
    Combines multiple sets of labels for the same recording into a single set of labels where there
    was agreement among the multiple labelings and returns the cleaned labels DataFrame.

    Note that this method assumes that there are no overlapping, identical labels in the label 
    data from any one person (if there is, it will count as multiple votes from a single person).  
    Use the clean_overlapping_labels on the label data from each person first to ensure this is not 
    the case.  However, it is OK for different labels to overlap (e.g. a 'cough' label and a 'rale' 
    label) within a single label file.

    Args:
        labels_list (list of DataFrame): A list of Pandas DataFrame objects containing the label 
            data (onset time, offset time, and event label) from each labeling source.
        consensus_threshold (float): Either this or vote_threshold must be specified (if both are
            specified, vote_threshold will be ignored). Specifies the percentage (in the range 
            [0,1]) of labelers that must agree on a label for that label to be included in the 
            final result.
        vote_threshold (float): Either this or consensus_threshold must be specified (this is 
            ignored if both are specified). Specifies the number of votes from labelers that must
            agree for a label to be included in the final result.  Note that uncertain labels (with 
            a question mark appended) may be counted as a fractional vote.
        onset_col (str): The name of the column containing the starting time for each label.
        offset_col (str): The name of the column containing the ending time for each label.
        label_col (str): The name of the column containing the label strings.
        label_weight (float): The weighting given to labels without a questionmark appended.
        uncertain_label_weight (float): The weighting given to labels with a questionmark appended
            to denote uncertainty.

    Returns:
        DataFrame: A data frame containing the start time, end time, and label for each period of 
            time where a sufficient number of labeling sources agreed on the label.
    """
    label_tally = tally_multiple_labelings(
        labels_list=labels_list,
        onset_col=onset_col,
        offset_col=offset_col,
        label_col=label_col,
        uncertain_label_weight=uncertain_label_weight,
        label_weight=label_weight,
    )
    if consensus_threshold:
        combined_labels = label_tally.loc[
            label_tally["consensus"] >= consensus_threshold, [onset_col, offset_col, label_col]
        ]
    elif vote_threshold:
        combined_labels = label_tally.loc[
            label_tally["votes"] >= vote_threshold, [onset_col, offset_col, label_col]
        ]
    else:
        raise RuntimeError("Must specify either consensus_threshold or vote_threshold")
    return clean_overlapping_labels(
        combined_labels, onset_col=onset_col, offset_col=offset_col, label_col=label_col
    )


def compare_to_true_labels(
    labels, true_labels, onset_col="onset", offset_col="offset", label_col="event_label",
):
    """
    Compares a provided labels dataframe against a dataframe containing what are considered to be 
    the true labels and returns performance metrics.  Note that any labels with question marks 
    appended are ignored.  Also, currently it combines the results across all label types.

    Args:
        labels (DataFrame): A Pandas DataFrame object containing the labels to be evaluated.
        true_labesl (DataFrame): A Pandas DataFrame object containing the true labels.
        onset_col (str): The name of the column containing the starting time for each label.
        offset_col (str): The name of the column containing the ending time for each label.
        label_col (str): The name of the column containing the label strings.
    
    Returns:
        (fp_seconds, fn_seconds, tp_seconds):  A tuple containing the total, summed duration in 
            seconds of all false positives, false negatives, and true positive portions of the 
            labels parameter as compared against the true_labels parameter.  Note that if there are
            multiple types of labels, the values will be summed across all label types.
    """
    true_labels_copy = true_labels.copy()
    # Pass two copies of the true labels in so that their votes count double.  This allows me to 
    # distinguish between false positives (which receive one vote total) and false negatives (which
    # receive a doubled vote of 2 from the true labels).
    tallied_labels = tally_multiple_labelings(
        [labels, true_labels, true_labels_copy],
        uncertain_label_weight=0.0,
        onset_col=onset_col,
        offset_col=offset_col,
        label_col=label_col,
    )
    tp = tallied_labels[tallied_labels["votes"] == 3]
    fn = tallied_labels[tallied_labels["votes"] == 2]
    fp = tallied_labels[tallied_labels["votes"] == 1]
    tp_seconds = (tp[offset_col] - tp[onset_col]).sum()
    fn_seconds = (fn[offset_col] - fn[onset_col]).sum()
    fp_seconds = (fp[offset_col] - fp[onset_col]).sum()
    return fp_seconds, fn_seconds, tp_seconds


def clean_typos_in_label_files(
    file_paths, accepted_labels, onset_col="onset", offset_col="offset", label_col="event_label"
):
    """
    Fixes typos in label strings for all the specified label files by interactively prompting the
    user on how each variant should be replaced.  For any files that are modified, it backs the 
    original file up to a file with ".original" appended to the file name (after the extension)
    unless that file name already exists (in which case it considers it already backed up and does
    not overwrite that file).

    Args:
        file_paths (list of Path): A list of pathlib Path objects pointing to each label file to 
            clean.
        accepted_labels (list of str): A list of strings specifying which labels are acceptable.  
            Variants of each of those labels with a question mark appended will automatically be 
            considered acceptable as well.
    """
    accepted_labels = [a.strip().lower() for a in accepted_labels]
    accepted_labels = accepted_labels + [a + "?" for a in accepted_labels]
    entry_for_delete_option = "d"
    menu_options = [(a, a) for a in accepted_labels] + [
        (entry_for_delete_option, "delete this label")
    ]
    labels_to_delete = []
    replacement_dict = {a: a for a in accepted_labels}
    for file_path in file_paths:
        try:
            labels = load_labels(file_path)
        except:
            print(f"Could not parse file.  Skipping {file_path}")
            continue
        bad_labels_idx = ~labels[label_col].isin(accepted_labels)
        if not bad_labels_idx.any():
            continue
        # Determine how to handle any bad labels we haven't seen before
        bad_labels = labels[bad_labels_idx][label_col]
        for bad_label in bad_labels:
            if bad_label in replacement_dict or bad_label in labels_to_delete:
                continue
            if bad_label.strip().lower() in replacement_dict:
                replacement_dict[bad_label] = replacement_dict[bad_label.strip().lower()]
            elif bad_label.strip() in labels_to_delete:
                labels_to_delete.append(bad_label)
            else:
                # Prompt the user for how to handle it
                selection = show_menu(
                    f"Select from the above to replace '{bad_label}'>", menu_options
                )
                if selection == entry_for_delete_option:
                    labels_to_delete.append(bad_label)
                    if bad_label.strip() not in labels_to_delete:
                        labels_to_delete.append(bad_label.strip())
                else:
                    replacement_dict[bad_label] = selection
        # Remove rows with labels that should be deleted
        labels = labels[~labels[label_col].isin(labels_to_delete)]
        # Replace the remaining bad labels
        labels[label_col] = labels[label_col].map(replacement_dict)
        # labels.replace({label_col: replacement_dict}, inplace=True) # This gives errors with overlapping values
        # Back up the original file if it hasn't been already
        backup_path = Path(str(file_path) + ".original")
        if not backup_path.exists():
            file_path.rename(backup_path)
        labels.to_csv(file_path, sep="\t", header=False, index=False, float_format="%.6f")
    # Print summary
    print(f"Labels deleted: {labels_to_delete}")
    print(f"Labels replaced: {replacement_dict}")


def show_menu(prompt, option_dict):
    """
    Shows a menu on the command line and ensures that the user selects a valid option.

    Args:
        prompt (str): The prompt to display (below the options).
        option_dict (dict): A dict mapping the string the user should enter to select an option 
            to a string giving a description of that option.
    """
    # Ensure the options the user input will be matched against are strings
    option_dict = {str(k): v for k, v in option_dict}
    got_valid_input = False
    while not got_valid_input:
        for item, description in option_dict.items():
            print(f"\t{item})\t{description}")
        user_input = input(prompt)
        got_valid_input = user_input in option_dict
    return user_input
