"""
Created 04-11-18 by Matthew C. McCallum
"""


from abc import ABC, abstractmethod


class AudioRead(ABC):
    """
    An abstract base class defining everything that you should expect from a derived audio reader class.
    """

    @abstractmethod
    def ReadSamplesFloat(self):
        """
        Reads all samples from the wav file as floats in the range -1.0 <= sample <= 1.0.

        Return:
            numpy.ndarray - An array of dimensions (num_channels, num_frames) containing float valued audio samples.
        """
        pass

    @abstractmethod
    def ReadSamplesInterleavedInt(self):
        """
        Reads all samples from file as integers in an interleaved list.

        Return:
            list(int) - A list of interleaved samples from the audio file.
        """
        pass

    @property
    @abstractmethod
    def data(self):
        """
        Any data that has been previously read in form the audio file.

        Return:
            ? - Audio sample data in the format that was most recently read from file.
        """
        pass

    @property
    @abstractmethod
    def audio_length(self):
        """
        The length of the audio in the file in seconds.

        Return:
            float - The duration of the audio file in seconds.
        """
        pass

    @property
    @abstractmethod
    def fmt(self):
        """
        Get the audio format object describing the file audio parameters, e.g., bit-depth, number channels, etc..

        Return:
            WavFmt - An object containing the audio format parameters.
        """
        pass