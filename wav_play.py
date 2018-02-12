"""
Created 01-07-17 by Matthew C. McCallum
"""

# Local modules
# None.

# Local submodules
from sigtools import WavFmt

# Thirdparty modules
import numpy as np
import pyaudio as pa

# Python library imports
import struct

class WavPlay( object ):
    """
    A class for playing back audio from data.
    """

    def __init__( self, fmt ):
        """
        Constructor

        Args:
            fmt - WavFmt - The audio format to be played back.
        """
        self._fmt = fmt
        self._pa = pa.PyAudio()

    def Play( self, audio ):
        """
        Play a numpy array as audio.

        Args:
            audio - np.ndarray - The audio to be played back. The numpy array is expected to be float values between
            -1.0 and 1.0, of dimensions (num_channels, num_frames).
        """
        if audio.ndim > 1:
            assert audio.shape[0] == self._fmt.n_channels
        else:
            assert self._fmt.n_channels == 1
            audio = audio.reshape((1, len(audio)))

        num_frames = audio.shape[1]

        interleaved_array = np.zeros( audio.shape[1] * self._fmt.n_channels )
        for chan in range( self._fmt.n_channels ):
            interleaved_array[chan::self._fmt.n_channels] = audio[chan,:]
        interleaved_array = interleaved_array*( 2**( self._fmt.bit_depth-1 ) )
        interleaved_array = interleaved_array.astype( int )
        interleaved_array = interleaved_array.tolist()
        audio_string = struct.pack( self._fmt.PackingString( num_frames ), *interleaved_array )

        stream = self._pa.open( format=self._pa.get_format_from_width( self._fmt.bit_depth/8 ),
                                channels=self._fmt.n_channels,
                                rate=self._fmt.samp_rate,
                                output=True )
        stream.write( audio_string )
        stream.stop_stream()
        stream.close()


if __name__=='__main__':
    # A little test script that can come in handy...
    fmt = WavFmt( 44100, 2, 16 )
    x = np.sin( 500/fmt.samp_rate*2*np.pi*np.arange( fmt.samp_rate ) )
    x = np.vstack( (x, x) )
    plyr = WavPlay( fmt )
    plyr.Play( x )