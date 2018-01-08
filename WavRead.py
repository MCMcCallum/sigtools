"""
Created 12-26-17 by Matthew C. McCallum
"""

# Local modules
from SigTools import WavFmt

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

    def __init__( self, filename ):
        """
        Constructor.

        Args:
            filename -> str - A string describing the filename and path of the audio wav file to be read.
        """
        self._filename = filename
        with wave.open(self._filename, 'r') as audio:
            self._num_frames = audio.getnframes()
            self._fmt = WavFmt.FromWav( audio )
        self._unpack_fmt = '<%i' % ( self._num_frames*self._num_channels )
        if self._bit_depth == 16:
            self._unpack_fmt += 'h'
        elif self._bit_depth == 32:
            self._unpack_fmt += 'i'
        else:
            raise Exception( 'Unsupporeted bit depth for wave file.' )

    def GetSamples( self ):
        """
        Get all samples from the wav file as floats in the range -1.0 <= sample <= 1.0

        Return:
            np.ndarray - An array of dimensions (num_channels, num_frames) containing float valued audio samples.
        """
        with wave.open( self._filename, 'r' ) as audio:
            data = audio.readframes( self._num_frames )
        data = struct.unpack( self._unpack_fmt, data )

        return_array = np.zeros( ( self._fmt.n_channels, self._num_frames ) )
        for channel in range( self._fmt.n_channels ):
            return_array[channel,:] = np.array( data[channel::self._fmt.n_channels] )/( 2.0**self._fmt.bit_depth )

        return return_array

    @property
    def fmt( self ):
        """
        Get the audio format object describing the file audio parameters, e.g., bit-depth, number channels, etc..

        Return:
            WavFmt - An object containing the audio format parameters.
        """
        return self._fmt
