"""
Created 01-07-17 by Matthew C. McCallum
"""

# Local modules
# None.

# Local submodules
# None.

# Thirdparty modules
# None.

# Python library imports
import wave

class WavFmt( object ):
    """
    An object for encapsulating audio parameters for when reading from file, writing to file, playback etc..
    """

    def __init__( self, samp_rate, n_channels, bit_depth ):
        """
        Constructor.

        Args:
            samp_rate -> int - The sampling rate in Hz of the audio.

            n_channels -> int - The number of audio channels.

            bit_depth -> int - The number of bits describing each sample.
        """
        self.samp_rate = samp_rate
        self.n_channels = n_channels
        self.bit_depth = bit_depth

    @classmethod
    def FromWav( cls, wav_file ):
        """
        Create an audio format based on an open wave.wave_read object.

        Args:
            wav_file -> wave.wave_read - An open file to get the audio format parameters from

        Return:
            WavFmt - An instantiated WavFmt object.
        """
        samp_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        bit_depth = wav_file.getsampwidth() * 8
        return cls( samp_rate, n_channels, bit_depth )

    @classmethod
    def FromFilename( cls, filename ):
        """
        Construct a WavFmt object populated with parameters to match a file on disk.

        Args:
            filename -> str - The filename to read the wav parameters from.

        Return:
            WavFmt - An instantiated WavFmt object.
        """
        with wave.open( filename, 'r' ) as audio:
            the_instance = cls.FromWav( audio )
        return the_instance

    def PackingString( self, num_frames ):
        """
        Get the string for packing or unpacking a given number of frames using the struct module.

        Args:
            num_frames -> int - The number of frames to cover in the packing string

        Return:
            str - The string to be used with the struct module for unpacking, packing the given number of frames for the
            current audio format described in this object.
        """
        unpack_fmt = '<%i' % ( num_frames * self.n_channels )
        if self.bit_depth == 16:
            unpack_fmt += 'h'
        elif self.bit_depth == 32:
            unpack_fmt += 'i'
        else:
            raise Exception('Unsupporeted bit depth format for packing data.')
        return unpack_fmt
