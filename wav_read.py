"""
Created 12-26-17 by Matthew C. McCallum
"""

# Local modules
from .wav_fmt import *

# Local submodules
# None.

# Thirdparty modules
import numpy as np

# Python library imports
import wave
import struct


class WavRead( object ):
    """
    A wave reader object for getting samples from file
    """
    SAMPLE_FMT_NONE = 'None'
    SAMPLE_FMT_INT_INTERLEAVED = 'Interleaved Integers'
    SAMPLE_FMT_FLOAT_ARRAY = 'Float 2D Array'

    def __init__( self, filehandle ):
        """
        Constructor.

        Args:
            filehandle -> str or seekable file - A string or seekable file like object describing either the filename and
            path of the audio wav file to be read, or a stream to the file data itself.
        """
        self._file = filehandle
        self._data = None
        self._data_fmt = self.SAMPLE_FMT_NONE

        with wave.open(self._file, 'rb') as audio:
            self._num_frames = audio.getnframes()
            self._fmt = WavFmt.FromWav( audio )

        if type(self._file) is not str:
            self._file.seek(0)

    def ReadSamplesInterleavedInt( self ):
        """
        Reads all samples from the wav file as integers in an interleaved list.
        This replaces any previous data read from the wav file.

        Return:
            list(int) - A list of interleaved samples from the audio file.
        """
        with wave.open( self._file, 'rb' ) as audio:
            data = audio.readframes( self._num_frames )
        if type(self._file) is not str:
            self._file.seek(0)
        data = struct.unpack( self._fmt.PackingString( self._num_frames ), data )

        self._data_fmt = self.SAMPLE_FMT_INT_INTERLEAVED
        self._data = list(data)

        return self._data

    def ReadSamplesFloat( self ):
        """
        Reads all samples from the wav file as floats in the range -1.0 <= sample <= 1.0.
        This replaces any previous data read from the wav file.

        Return:
            np.ndarray - An array of dimensions (num_channels, num_frames) containing float valued audio samples.
        """
        self.ReadSamplesInterleavedInt()

        return_array = np.zeros( ( self._fmt.n_channels, self._num_frames ) )
        for channel in range( self._fmt.n_channels ):
            return_array[channel,:] = np.array( self._data[channel::self._fmt.n_channels] )/( 2.0**self._fmt.bit_depth )

        self._data_fmt = self.SAMPLE_FMT_FLOAT_ARRAY
        self._data = return_array

        return self._data

    @property
    def fmt( self ):
        """
        Get the audio format object describing the file audio parameters, e.g., bit-depth, number channels, etc..

        Return:
            WavFmt - An object containing the audio format parameters.
        """
        return self._fmt

    @property
    def data( self ):
        """
        Any data that has been previously read in form the wave file.

        Return:
            ? - Audio sample data in the format that was most recently read from file.
        """
        return self._data

    @property
    def data_fmt( self ):
        """
        The current audio sample format of the data.

        Return:
            str - A string constant describing the format in which the class's data is in.
        """
        return self._data_fmt

    @property
    def audio_length( self ):
        """
        The length of the audio in the file in seconds.

        Return:
            float - The duration of the audio file in seconds.
        """
        return self._num_frames/self._fmt.samp_rate
    