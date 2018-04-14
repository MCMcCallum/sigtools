"""
Created 04-11-18 by Matthew C. McCallum
"""


# Local imports
from .mp3_read import Mp3Read
from .wav_read import WavRead

# Third party imports
from data_access import *

# Standard library imports
import os


def MakeAudioReader(url):
    """
    Factory function for making an AudioRead type object of various types, dependent on the file type.

    Args:
        url -> str - The URL of the audio file thing to read.

    Return:
        AudioRead - An object for reading from files.
    """
    if os.path.splitext(url)[1] == '.mp3':
        return Mp3Read(get_stream(url, 'rb'))
    elif os.path.splitext(url)[1] == '.wav':
        return WavRead(get_stream(url, 'rb'))