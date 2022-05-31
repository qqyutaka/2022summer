# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms. 

import re
import itertools
import pandas as pd
import numpy as np
import scipy as sp

from pathlib import Path


class Dataset:
    # List of regular expressions to parse known filename formats.
    filename_regexes = [
        (
            r"^(.*[\\/])?(?P<dataset_tag>[^_]+)_(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_"
            r"(?P<hour>\d{2})\.(?P<minute>\d{2})\.(?P<second>\d{2})_ch(?P<channel>\d+)\.[a-zA-Z]+$"
        ),
        (
            r"^(.*[\\/])?(?P<dataset_tag>[^_]+)_mic(?P<channel>\d+)_"
            r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_"
            r"(?P<hour>\d{2})\.(?P<minute>\d{2})\.(?P<second>\d{2})\.[a-zA-Z]+$"
        ),
    ]
    # Class variable to store the last working filename_regex so we can try it first for the next
    # file without having to search through all the others first.
    last_working_filename_regex = None
    # filename_re = (
    #    r"^(.*[\\/])?(?P<dataset_tag>[^_]+)_(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_"
    #    r"(?P<hour>\d{2})\.(?P<minute>\d{2})\.(?P<second>\d{2})_ch(?P<channel>\d+)\.[a-zA-Z]+$"
    # )

    def __init__(self, root_folder, follows_folder_structure=True):
        """
        Initializes a Dataset object with the given root folder.  When the data follows our 
        convention for folder structure (yyyy-mm-dd/HH/), then the root folder should be the folder
        containing all the yyyy-mm-dd/ subfolders.  Otherwise, it is just the top level folder 
        containing all the data.

        Args:
            root_folder (str): The root folder for the dataset.  This is the folder containing all 
                the yyyy-mm-dd/ folders when the folder structure convention is followed, or just
                the top level folder containing all the data otherwise.
            follows_folder_structure (bool): Whether or not the folder follows our convention for
                folder structure of yyyy-mm-dd/HH/.  Setting this to True for folders that follow 
                the folder structure convention will speed up the process of finding relevant paths.
        """
        self.root_folder = Path(root_folder)
        self.follows_folder_structure = follows_folder_structure

    def walk_date_range(self, pattern, start_timestamp, end_timestamp):
        """
        Gets all the file paths falling within the specified timestamp range.

        Args:
            pattern (str): The pattern (or glob) the returned filenames must follow.  For example,
                "_ch2.flac" to get files for channel 2 from a dataset.
            start_timestamp (pd.Timestamp-like): The starting timestamp (inclusive) of the time 
                range over which to get files.  This can be anything that the pd.Timestamp() 
                constructor accepts.
            end_timestamp (pd.Timestamp-like): The ending timestamp (exclusive) of the time range 
                over which to get files.
        """
        start_timestamp = pd.Timestamp(start_timestamp)
        end_timestamp = pd.Timestamp(end_timestamp)
        if self.follows_folder_structure:
            file_paths = []
            # Construct hour timestamps over the range
            start_hour = start_timestamp.floor("H")
            folder_hours = pd.date_range(start_hour, end_timestamp, freq="H", closed="left")
            if folder_hours.size >= 1:
                # Start folder (may only get some files)
                folder_path = self.root_folder / folder_hours[0].strftime("%Y-%m-%d/%H")
                if folder_path.is_dir():
                    file_paths.extend(
                        self.filter_date_range(
                            tuple(folder_path.glob(pattern)), start_timestamp, end_timestamp
                        )
                    )
            if folder_hours.size >= 3:
                # Middle folders (get all files)
                for folder_idx in range(1, folder_hours.size - 1):
                    folder_path = self.root_folder / folder_hours[folder_idx].strftime(
                        "%Y-%m-%d/%H"
                    )
                    if folder_path.is_dir():
                        file_paths.extend(folder_path.glob(pattern))
            if folder_hours.size >= 2:
                # End folder (may only get some files)
                folder_path = self.root_folder / folder_hours[-1].strftime("%Y-%m-%d/%H")
                if folder_path.is_dir():
                    file_paths.extend(
                        self.filter_date_range(
                            folder_path.glob(pattern), start_timestamp, end_timestamp
                        )
                    )
            return file_paths
        else:
            return self.filter_date_range(
                [f for f in self.root_folder.rglob(pattern) if f.is_file()],
                start_timestamp,
                end_timestamp,
            )

    def walk_time_of_day_range(self, pattern, start_timestamp, end_timestamp, start_time, end_time):
        """
        Gets all the file paths falling within the specified timestamp range and time of day range.

        Args:
            pattern (str): The pattern (or glob) the returned filenames must follow.  For example,
                "_ch2.flac" to get files for channel 2 from a dataset.
            start_timestamp (pd.Timestamp-like): The starting timestamp (inclusive) of the time 
                range over which to get files.  This can be anything that the pd.Timestamp() 
                constructor accepts.
            end_timestamp (pd.Timestamp-like): The ending timestamp (exclusive) of the time range 
                over which to get files.
            start_time (pd.Timedelta-like): The starting time of day (inclusive) of the time of day 
                range over which to get files.  Represented as the difference in time (Timedelta)
                between the time and the beginning of the given day.  This can be anything that the
                pd.Timedelta() constructor accepts.
            end_time (pd.Timedelta-like): The ending time of day (inclusive) of the time of day 
                range over which to get files.  
        """
        start_timestamp = pd.Timestamp(start_timestamp)
        end_timestamp = pd.Timestamp(end_timestamp)
        start_time = pd.Timedelta(start_time)
        end_time = pd.Timedelta(end_time)
        if self.follows_folder_structure:
            file_paths = []
            # Construct hour timestamps over the range
            start_hour = start_timestamp.floor("H")
            folder_hours = pd.date_range(start_hour, end_timestamp, freq="H", closed="left")
            # Filter out hours from the wrong time of day
            start_tod_hour = start_time.floor("H")
            folder_tod = folder_hours - folder_hours.floor("D")
            if start_time > end_time:
                folder_hours = folder_hours[
                    (start_tod_hour <= folder_tod) | (folder_tod < end_time)
                ]
            else:
                folder_hours = folder_hours[
                    (start_tod_hour <= folder_tod) & (folder_tod < end_time)
                ]
            # Walk through each hour folder assembling the list of paths
            for folder_hour in folder_hours:
                folder_path = self.root_folder / folder_hour.strftime("%Y-%m-%d/%H")
                if folder_path.is_dir():
                    file_paths.extend(
                        self.filter_time_of_day_range(
                            folder_path.glob(pattern),
                            start_timestamp,
                            end_timestamp,
                            start_time,
                            end_time,
                        )
                    )
            return file_paths
        else:
            return self.filter_time_of_day_range(
                self.root_folder.rglob(pattern),
                start_timestamp,
                end_timestamp,
                start_time,
                end_time,
            )

    @classmethod
    def filter_date_range(cls, file_paths, start_timestamp, end_timestamp):
        """
        Filters the input list of paths down to those that fall within the specified timestamp range.

        Args:
            file_paths (list of str): The list of paths to filter from.
            start_timestamp (pd.Timestamp-like): The starting timestamp (inclusive) of the time 
                range over which to get files.  This can be anything that the pd.Timestamp() 
                constructor accepts.
            end_timestamp (pd.Timestamp-like): The ending timestamp (exclusive) of the time range 
                over which to get files.
        """
        # We need file_paths twice, so convert to a list in case it's a generator that would get
        # consumed on the first use.
        file_paths = list(file_paths)
        start_timestamp = pd.Timestamp(start_timestamp)
        end_timestamp = pd.Timestamp(end_timestamp)
        (_, timestamps, _) = cls.parse_file_paths(file_paths)
        mask = (start_timestamp <= timestamps) & (timestamps < end_timestamp)
        return [f for f, m in zip(file_paths, mask) if m]

    @classmethod
    def filter_time_of_day_range(
        cls, file_paths, start_timestamp, end_timestamp, start_time, end_time
    ):
        """
        Filters the input list of paths down to those that fall within the specified timestamp range
        and time of day range.

        Args:
            file_paths (list of str): The list of paths to filter from.
            start_timestamp (pd.Timestamp-like): The starting timestamp (inclusive) of the time 
                range over which to get files.  This can be anything that the pd.Timestamp() 
                constructor accepts.
            end_timestamp (pd.Timestamp-like): The ending timestamp (exclusive) of the time range 
                over which to get files.
            start_time (pd.Timedelta-like): The starting time of day (inclusive) of the time of day 
                range over which to get files.  Represented as the difference in time (Timedelta)
                between the time and the beginning of the given day.  This can be anything that the
                pd.Timedelta() constructor accepts.
            end_time (pd.Timedelta-like): The ending time of day (inclusive) of the time of day 
                range over which to get files.  
        """
        # We need file_paths twice, so convert to a list in case it's a generator that would get
        # consumed on the first use.
        file_paths = list(file_paths)
        start_timestamp = pd.Timestamp(start_timestamp)
        end_timestamp = pd.Timestamp(end_timestamp)
        start_time = pd.Timedelta(start_time)
        end_time = pd.Timedelta(end_time)
        (_, timestamps, _) = cls.parse_file_paths(file_paths)
        time_of_day = timestamps - timestamps.floor("D")
        if start_time > end_time:
            # start_time is after end_time, so the time of day range should wrap across midnight
            mask = (
                (start_timestamp <= timestamps)
                & (timestamps < end_timestamp)
                & ((start_time <= time_of_day) | (time_of_day < end_time))
            )
        else:
            mask = (
                (start_timestamp <= timestamps)
                & (timestamps < end_timestamp)
                & (start_time <= time_of_day)
                & (time_of_day < end_time)
            )
        return [f for f, m in zip(file_paths, mask) if m]

    @classmethod
    def parse_file_path(cls, file_path):
        """
        Parses the given file path and returns a tuple containing it's dataset tag, timestamp, and
        channel number.

        Args:
            file_path (path-like): The file path to parse.

        Returns:
            (str, pd.Timestamp, int): Tuple containing the file's dataset tag, timestamp, and 
                channel number.
        """
        match = None
        if cls.last_working_filename_regex:
            # Try the last regex that worked if available
            match = re.match(cls.last_working_filename_regex, str(file_path))
        if not match:
            # Try to find a working regex from the list for known file naming schemes
            for filename_regex in cls.filename_regexes:
                match = re.match(filename_regex, str(file_path))
                if match:
                    # Remember which regex worked
                    cls.last_working_filename_regex = filename_regex
                    break
            else:
                # None of the known regexes worked (the for loop terminated wihtout hitting a break)
                raise RuntimeError("Unable to parse the path " + str(file_path))
        return (
            match.group("dataset_tag"),
            pd.Timestamp(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
                int(match.group("hour")),
                int(match.group("minute")),
                int(match.group("second")),
            ),
            int(match.group("channel")),
        )

    @classmethod
    def parse_file_paths(cls, file_paths):
        """
        Parses the given file paths and returns a list of tuples containing their dataset tags, 
        timestamps, and channel numbers.

        Args:
            file_paths (list of path-like): The file paths to parse.

        Returns:
            (list of str, list of pd.Timestamp, list of int): A tuple containing a list of each 
                file's dataset tag, a list of each file's timestamp, and a list of each file's 
                channel number.
        """
        if file_paths:
            dataset_tags, timestamps, channels = zip(*[cls.parse_file_path(p) for p in file_paths])
        else:
            dataset_tags, timestamps, channels = [], [], []
        return dataset_tags, pd.DatetimeIndex(timestamps), np.array(channels)

    @classmethod
    def build_file_name(cls, dataset_tag, file_timestamp, channel, extension="flac"):
        """
        Constructs a file name from the given dataset tag, timestamp, and channel number.

        Args:
            dataset_tag (str): The dataset tag to prepend to the file name.
            file_timestamp (pd.Timestamp-like): The timestamp for the file.
            channel (int): The file's channel number.
            extension (str, optional): The extension for the file.  Defaults to "flac".
        """
        file_timestamp = pd.Timestamp(file_timestamp)
        return (
            dataset_tag
            + "_"
            + file_timestamp.strftime("%Y-%m-%d_%H.%M.%S")
            + "_ch%d.%s" % (channel, extension)
        )

    def build_file_path(self, dataset_tag, file_timestamp, channel, extension="flac"):
        """
        Constructs a full file path from the given dataset tag, timestamp, and channel number.

        Args:
            dataset_tag (str): The dataset tag to prepend to the file name.
            file_timestamp (pd.Timestamp-like): The timestamp for the file.
            channel (int): The file's channel number.
            extension (str, optional): The extension for the file.  Defaults to "flac".
        """
        file_timestamp = pd.Timestamp(file_timestamp)
        return (
            self.root_folder
            / file_timestamp.strftime("%Y-%m-%d")
            / file_timestamp.strftime("%H")
            / self.build_file_name(dataset_tag, file_timestamp, channel, extension)
        )

    @classmethod
    def build_file_paths(cls, dataset_tags, file_timestamps, channels):
        # TODO: Allow just passing one dataset_tag and/or channel
        raise RuntimeError("Not yet implemented.")


if __name__ == "__main__":
    filenames = [
        "LT3-G2_2014-03-24_12.21.00_ch2.flac",
        "Ark2-102_2017-10-31_23.08.14_ch1.flac",
        "COM1_2015-01-01_00.00.00_ch1.wav",
        "2015-01-01/00/COM1_2015-01-01_00.00.00_ch1.wav",
    ]
    for f in filenames:
        print(Dataset.parse_file_path(f))
    z = Dataset.parse_file_paths(filenames)
    (dataset_tags, timetstamps, channels) = Dataset.parse_file_paths(filenames)
    print(dataset_tags)
    print(timetstamps)
    print(channels)
    lt3_folder = "E:/processed_audio_2014-10_PDRC_LT3_g3"
    ds = Dataset(root_folder=lt3_folder, follows_folder_structure=True)
    file_paths = ds.walk_date_range(
        "*_ch2.flac", pd.Timestamp(2014, 11, 2, 12, 35, 10), pd.Timestamp(2014, 11, 4, 2, 2, 1, 5)
    )
    print("%d paths retrieved" % len(file_paths))
    print(file_paths[0])
    print(file_paths[-1])
    file_paths = ds.walk_time_of_day_range(
        "*_ch2.flac", "2014-11-2 12:35:10", "2014/11/4 2:2:1", "05:03:24", "05:09:10"
    )
    print("%d paths retrieved" % len(file_paths))
    print(file_paths[0])
    print(file_paths[-1])
    file_paths = ds.walk_time_of_day_range(
        "*_ch2.flac",
        pd.Timestamp(2014, 11, 2, 12, 35, 10),
        pd.Timestamp(2014, 11, 4, 2, 2, 1),
        "05:03:24",
        "05:09:00",
    )
    print("%d paths retrieved" % len(file_paths))
    print(file_paths[0])
    print(file_paths[-1])
    file_paths = ds.walk_time_of_day_range(
        "*_ch2.flac",
        "2/11/2014 12:35:10",
        pd.Timestamp(2014, 11, 4, 2, 2, 1),
        "23:58:59",
        "00:01:10",
    )
    print("%d paths retrieved" % len(file_paths))
    print(file_paths[0])
    print(file_paths[-1])
    ds = Dataset(root_folder=lt3_folder, follows_folder_structure=False)
    file_paths = ds.walk_date_range(
        "*_ch2.flac", pd.Timestamp(2014, 11, 2, 12, 35, 10), pd.Timestamp(2014, 11, 4, 2, 2, 1, 5)
    )
    print("%d paths retrieved" % len(file_paths))
    print(file_paths[0])
    print(file_paths[-1])
    file_paths = ds.walk_time_of_day_range(
        "*_ch2.flac",
        pd.Timestamp(2014, 11, 2, 12, 35, 10),
        pd.Timestamp(2014, 11, 4, 2, 2, 1),
        "05:03:24",
        "05:09:10",
    )
    print("%d paths retrieved" % len(file_paths))
    print(file_paths[0])
    print(file_paths[-1])
    file_paths = ds.walk_time_of_day_range(
        "*_ch2.flac",
        pd.Timestamp(2014, 11, 2, 12, 35, 10),
        pd.Timestamp(2014, 11, 4, 2, 2, 1),
        "05:03:24",
        "05:09:00",
    )
    print("%d paths retrieved" % len(file_paths))
    print(file_paths[0])
    print(file_paths[-1])
    file_paths = ds.walk_time_of_day_range(
        "*_ch2.flac",
        pd.Timestamp(2014, 11, 2, 12, 35, 10),
        pd.Timestamp(2014, 11, 4, 2, 2, 1),
        "23:58:59",
        "00:01:10",
    )
    print("%d paths retrieved" % len(file_paths))
    print(file_paths[0])
    print(file_paths[-1])
    (tag, ts, ch) = ds.parse_file_path(file_paths[0])
    print(ds.build_file_name(tag, ts, ch))
    print(ds.build_file_path(tag, ts, ch))
    (tags, tss, chs) = ds.parse_file_paths(file_paths)
    print(tags)
    print(tss)
    print(chs)
