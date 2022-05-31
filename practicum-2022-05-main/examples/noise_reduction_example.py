# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms.

from pathlib import Path

from audiot.audio_signal import AudioSignal
from audiot.noise_reducer import NoiseReducer
from datetime import datetime, timedelta

if __name__ == "__main__":
    project_folder = Path(__file__).absolute().parents[1]
    # List of files to remove nosie from
    input_file_list = [
        project_folder / "test_data/LT3-G3_2014-11-05_15.58.00_ch2.flac",
        project_folder / "test_data/TRF0_mic14_2020-12-17_01.20.00.flac",
    ]
    print("-" * 80)
    start_time = datetime.now()
    for input_file_path in input_file_list:
        output_file_path = input_file_path.parent / (
            input_file_path.stem + "_noise_reduced" + input_file_path.suffix
        )
        print(f"{input_file_path}")
        signal = AudioSignal.from_file(input_file_path)
        print(f"Removing noise...")
        noise_reduced_signal = NoiseReducer.reduce_noise(signal)
        print(f"{output_file_path}")
        noise_reduced_signal.write(output_file_path)
        print("-" * 80)
    duration = datetime.now() - start_time
    print(f"Done!  Time taken = {duration}")
