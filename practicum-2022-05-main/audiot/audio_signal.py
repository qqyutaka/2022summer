# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms. 

import numpy as np
import scipy as sp
import soundfile


class AudioSignal:
    """
    Represents an audio signal or waveform with one or more channels.

    Attributes:
        signal (ndarray): 2D array containing the waveforms for each channel, shape (n_samples, 
            n_channels).
        sample_rate (float): The sample rate (in Hz) for the waveform.
        file_path (Path): The path to the file from which the data was read, if any.
        file_format (str): The format of the file from which the data was read, if any.
        subtype (str): The subtype of the data read from a file, if applicable.
    """

    def __init__(self, signal, sample_rate, file_path=None, file_format=None, subtype=None):
        """
        Initializes an AudioSignal object.

        Args:
            signal (array-like): The waveforms for each channel, shape (n_samples, n_channels).
            sample_rate (float): The sample rate (in Hz) for the waveform.
        """
        self.signal = np.asarray(signal)
        # Guarantee a 2d array so indexing is consistent whether there's one or multiple channels.
        if self.signal.ndim == 1:
            self.signal = self.signal[:, np.newaxis]
        self.sample_rate = float(sample_rate)
        self.file_path = file_path
        self.file_format = file_format
        self.subtype = subtype

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        self._sample_rate = sample_rate

    @property
    def n_samples(self):
        """Gets the number of samples in the waveform for each channel."""
        return self.signal.shape[0]

    @property
    def n_channels(self):
        """Gets the number of channels in the signal."""
        return self.signal.shape[1]

    @property
    def duration(self):
        """Gets the duration of the signal in seconds."""
        return float(self.n_samples) / self.sample_rate

    def write(self, output_path, sample_rate=None, subtype=None):
        """
        Writes the audio signal out to an audio file (format determined by the file extension).

        Args:
            output_path (string): The path and file name to write out to.
            sample_rate (float): The sample rate to write the data out at.  Defaults to the stored
                sample rate if not overridden.
            subtype (str): The subtype (format) to write the data out.  See 
                soundfile.available_subtypes() for possible values.  Defaults to the stored subtype 
                if any, and otherwise uses soundfile's default subtype for the file type (see 
                soundfile.defaultsubtype()).
        """
        soundfile.write(output_path, self.signal, int(self.sample_rate + 0.5), self.subtype)

    def slice_samples(self, start_sample_idx, end_sample_idx):
        return AudioSignal(
            signal=self.signal[start_sample_idx:end_sample_idx],
            sample_rate=self.sample_rate,
            file_path=self.file_path,
            file_format=self.file_format,
            subtype=self.subtype,
        )

    def slice_time(self, slice_start_seconds, slice_end_seconds):
        return self.slice_samples(
            start_sample_idx=max(0, int(slice_start_seconds * self.sample_rate + 0.5)),
            end_sample_idx=min(self.n_samples, int(slice_end_seconds * self.sample_rate + 0.5)),
        )

    def slice_time_with_context(
        self, start_seconds, end_seconds, context_before=0, context_after=0
    ):
        actual_context_before = min(context_before, start_seconds)
        actual_context_after = min(context_after, self.duration - end_seconds)
        return (
            self.slice_time(
                start_seconds - actual_context_before, end_seconds + actual_context_after
            ),
            actual_context_before,
            actual_context_after,
        )

    def fade_ends(self, fade_duration, fade_start=True, fade_end=True):
        fade_n_samples = int(fade_duration * self.sample_rate + 0.5)
        ramp = np.arange(0, 1, 1 / fade_n_samples)[:, np.newaxis]
        if fade_start:
            self.signal[:fade_n_samples] = self.signal[:fade_n_samples] * ramp
        if fade_end:
            self.signal[-fade_n_samples:] = self.signal[-fade_n_samples:] * ramp[::-1, :]

    @classmethod
    def from_file(cls, file_path):
        """
        Returns an AudioSignal object read in from the specified audio file.

        Args:
            file_path (string): The path to the audio file to read in.
        """
        info = soundfile.info(str(file_path))
        (signal, sample_rate) = soundfile.read(str(file_path), always_2d=True)
        return cls(
            signal=signal,
            sample_rate=sample_rate,
            file_path=file_path,
            file_format=info.format,
            subtype=info.subtype,
        )

    @classmethod
    def from_bytes_io(cls, bytes_buffer):
        """
        Returns an AudioSignal object read in from a BytesIO buffer.

        Args:
            bytes_buffer (BytesIO): A BytesIO buffer object containing the binary contents of an
                audio file (flac, wav, etc).
        """
        bytes_buffer.seek(0)
        info = soundfile.info(bytes_buffer)
        bytes_buffer.seek(0)
        (signal, sample_rate) = soundfile.read(bytes_buffer, always_2d=True)
        return cls(
            signal=signal,
            sample_rate=sample_rate,
            file_format=info.format,
            subtype=info.subtype,
        )

    @classmethod
    def concatenate(cls, *audio_signals):
        """
        Concatenates multiple AudioSignal objects into one AudioSignal object.

        Args:
            *audio_signals: Variable length argument list containing each AudioSignal to be
                concatenated, in the order in which they should be concatenated.

        Returns:
            The AudioSignal object containing the concatenated waveforms.

        Raises:
            ValueError: If the signals are not compatible to be concatenated (different numbers of 
                channels or different sample rates).
        """
        n_channels = audio_signals[0].n_channels
        sample_rate = audio_signals[0].sample_rate
        file_format = audio_signals[0].file_format
        subtype = audio_signals[0].subtype
        if not all(sig.n_channels == n_channels for sig in audio_signals):
            raise ValueError("Cannot concatenate signals with different numbers of channels.")
        if not all(sig.sample_rate == sample_rate for sig in audio_signals):
            raise ValueError("Cannot concatenate signals with different sample rates.")
        if not all(sig.file_format == file_format for sig in audio_signals):
            file_format = None
        if not all(sig.subtype == subtype for sig in audio_signals):
            subtype = None
        return cls(
            signal=np.vstack([sig.signal for sig in audio_signals]),
            sample_rate=sample_rate,
            file_format=file_format,
            subtype=subtype,
        )

    @classmethod
    def stack_channels(cls, *audio_signals):
        """
        Stacks multiple AudioSignal objects as different channels (or sets of channels) in one 
        AudioSignal object.

        Args:
            *audio_signals: Variable length argument list containing each AudioSignal to be
                stacked, in the order in which they should be stacked.

        Returns:
            The AudioSignal object containing all the signals stacked together as different 
                channels.

        Raises:
            ValueError: If the signals are not compatible to be stacked (different lengths or 
                different sample rates).
        """
        sample_rate = audio_signals[0].sample_rate
        n_samples = audio_signals[0].n_samples
        file_format = audio_signals[0].file_format
        subtype = audio_signals[0].subtype
        if not all(sig.n_samples == n_samples for sig in audio_signals):
            raise ValueError("Cannot stack signal channels with different numbers of samples.")
        if not all(sig.sample_rate == sample_rate for sig in audio_signals):
            raise ValueError("Cannot stack signal channels with different sample rates.")
        if not all(sig.file_format == file_format for sig in audio_signals):
            file_format = None
        if not all(sig.subtype == subtype for sig in audio_signals):
            subtype = None
        return cls(
            signal=np.hstack([sig.signal for sig in audio_signals]),
            sample_rate=sample_rate,
            file_format=file_format,
            subtype=subtype,
        )


if __name__ == "__main__":
    # Test code, assumes "7.wav" exists in the current folder.
    print("s1" + "-" * 80)
    s1 = AudioSignal.from_file("7.wav")
    print(s1.signal.shape)
    print(s1.n_samples)
    print(s1.n_channels)
    print(s1.signal[2, 0])
    print(s1.duration)
    print(s1.file_path)
    print(s1.file_format)
    print(s1.subtype)

    print("s2" + "-" * 80)
    s2 = AudioSignal.stack_channels(s1, s1, s1)
    print(s2.signal.shape)
    print(s2.n_samples)
    print(s2.n_channels)
    print(s2.signal[2, 0])
    print(s2.duration)
    print(s2.file_path)
    print(s2.file_format)
    print(s2.subtype)
    s2.write("7_2ch.flac")
    print("read back in...")
    s2 = AudioSignal.from_file("7_2ch.flac")
    print(s2.signal.shape)
    print(s2.n_samples)
    print(s2.n_channels)
    print(s2.signal[2, 0])
    print(s2.duration)
    print(s2.file_path)
    print(s2.file_format)
    print(s2.subtype)

    print("s3" + "-" * 80)
    s3 = AudioSignal.concatenate(s1, s1, s1)
    print(s3.signal.shape)
    print(s3.n_samples)
    print(s3.n_channels)
    print(s3.signal[2, 0])
    print(s3.duration)
    print(s3.file_path)
    print(s3.file_format)
    print(s3.subtype)
    s3.write("7_concat.flac")

    print("s4" + "-" * 80)
    s4 = AudioSignal.from_file("7_concat.flac")
    print(s4.signal[2, 0])
    print(s4.file_path)
    print(s4.file_format)
    print(s4.subtype)
