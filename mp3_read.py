"""
Created 04-11-18 by Matthew C. McCallum
"""


# Local imports
from .wav_read import *
from .audio_read import *

# Third party imports
import mutagen.mp3

# Python standard library imports
import tempfile
import subprocess
import shutil
import os


class Mp3Read(AudioRead):
    """
    A class for reading data and metadata from mp3 files.
    """

    WAV_SAMP_RATE = 44100
    WAV_BIT_DEPTH = 16
    WAV_FFMPEG_FMT = 'pcm_s' + str(WAV_BIT_DEPTH) + 'le'

    def __init__(self, filename):
        """
        Constructor.

        Args:
            filename -> str - The url of the file to be read.
        """
        # Note: all the file reading / decoding / conversion to wav is done just in time, to prevent additional overhead
        # when you might only need the class for something like getting the mp3 audio length.
        self._file = filename
        self._temp_file = None
        self._temp_filename = None
        self._data = None
        n_channels = mutagen.mp3.MP3(self._file).info.channels
        if type(self._file) is not str:
            self._file.seek(0)
        self._fmt = WavFmt(samp_rate=self.WAV_SAMP_RATE, n_channels=n_channels, bit_depth=self.WAV_BIT_DEPTH)

    def __del__(self):
        """
        Destructor.

        This isn't really necessary, but just to be explicit about getting rid of that temporary file this class keeps
        around.
        """
        del self._temp_file
        self._temp_filename = None
        self._data = None

    def ConvertFile(self):
        """
        Converts the file to a temporary wav file. Once converted this wav file will stick around as long as this object
        exists.
        """
        self._temp_file = tempfile.NamedTemporaryFile(mode='r+b', suffix='.wav')
        self._temp_filename = self._temp_file.name

        # Write wav data
        if type(self._file) is not str:
            # Copy to a local location first in case it is remote...
            temp_mp3_file = tempfile.NamedTemporaryFile(mode='r+b', suffix='.mp3')
            temp_mp3_file.write(self._file.read())
            self._file.seek(0)
            fname = temp_mp3_file.name
        else:
            fname = self._file
        subprocess.run(
            ["ffmpeg", "-loglevel", "panic", "-i", fname, "-vn", "-acodec", self.WAV_FFMPEG_FMT, "-ac",
             str(self._fmt.n_channels), "-ar", str(self.WAV_SAMP_RATE), "-f", "wav", 'pipe:1'], stdout=self._temp_file)

        # Fix file size as ffmpeg output via std stream doesn't include a file size.
        self._temp_file.seek(0)
        file_length = self._temp_file.seek(0, 2)
        self._temp_file.seek(4)
        self._temp_file.write(struct.pack('i', file_length - 8))
        self._temp_file.seek(0)
        test_data = self._temp_file.read(10000)
        data_start = test_data.find(b'data')
        self._temp_file.seek(data_start + 4)
        self._temp_file.write(struct.pack('i', file_length - data_start - 8))
        self._temp_file.seek(0)

        # Update channels in case the mp3 metadata was wrong before
        wav_file = WavRead(self._temp_filename)
        self._fmt.n_channels = wav_file.fmt.n_channels

    def SaveWav(self, directory, filename=None):
        """
        Copies the MP3 data into a local wav file as specified by directory and filename.

        Args:
            directory -> str - A string describing the location to save the file to.

            filename -> str - The base filename to save the wav file to, if not provided the wav file will have
            the same name as the original mp3 file, but with a wav extension.
        """
        # TODO [matthew.mccallum 06.09.18]: This currently assumes the file the class is configured with and the
        # provided arguments are strings. I should generalize this to file streams so that it can be saved on NFS,
        # or S3 for example.
        if not self._temp_file:
            self.ConvertFile()

        if type(self._file) is not str:
            filename = self._file.name
        else:
            filename = self._file

        save_filename = os.path.join(directory, os.path.splitext(os.path.basename(filename))[0]+".wav")

        shutil.copy(self._temp_filename, save_filename)

    def ReadSamplesFloat(self):
        """
        Reads all samples from the wav file as floats in the range -1.0 <= sample <= 1.0.
        This replaces any previous data read from the wav file.

        Return:
            np.ndarray - An array of dimensions (num_channels, num_frames) containing float valued audio samples.
        """
        if not self._temp_file:
            self.ConvertFile()

        # At this stage the wav file is just opened each time it is needed, this should be pretty light weight.
        wav_file = WavRead(self._temp_filename)
        self._data = wav_file.ReadSamplesFloat()
        return self._data

    def ReadSamplesInterleavedInt(self):
        """
        Reads all samples as integers in an interleaved list.
        This replaces any previous data read from file.

        Return:
            list(int) - A list of interleaved samples from the audio file.
        """
        if not self._temp_file:
            self.ConvertFile()

        # At this stage the wav file is just opened each time it is needed, this should be pretty light weight.
        wav_file = WavRead(self._temp_filename)
        self._data = wav_file.ReadSamplesInterleavedInt()
        return self._data

    @property
    def data(self):
        """
        Any data that has been previously read in form the wave file.

        Return:
            ? - Audio sample data in the format that was most recently read from file.
        """
        return self._data

    @property
    def audio_length(self):
        """
        The length of the audio in the file in seconds.

        Return:
            float - The duration of the audio file in seconds.
        """
        if self._temp_file:
            # Get the audio length from the wav file (more accurate)
            return WavRead(self._temp_filename).audio_length
        else:
            # Get the audio length from the mp3 file (more efficient)
            # TODO [matthew.mccallum 04.11.18]: Test the below length getting with a BufferedIOBase derived object.
            length = mutagen.mp3.MP3(self._file).info.length
            self._file.seek(0)
            return length

    @property
    def fmt( self ):
        """
        Get the audio format object describing the file audio parameters, e.g., bit-depth, number channels, etc..

        Return:
            WavFmt - An object containing the audio format parameters.
            Note: Because of the way this class is implemented, the format really is a wav format rather than an mp3
            format. That is, all mp3 data is converted to wav data before being read.
        """
        return self._fmt
