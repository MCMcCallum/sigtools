"""
Created 04-11-18 by Matthew C. McCallum
"""


# Local imports
from sigtools import MakeAudioReader

# Third party imports
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ioff()

# Python standard library imports
import unittest
import time


class TestAudioRead(unittest.TestCase):

    def test_mp3_audio_data(self):
        """
        Test that the mp3 data property is set correctly and plot the data to see if it looks right.
        """
        url = './resources/Simple.mp3'
        reader = MakeAudioReader(url)
        data = reader.ReadSamplesFloat()
        self.assertEqual(id(data), id(reader._data)) # Make sure the property is set correctly.
        start = 2*44100
        end = 3*44100
        plt.plot(data[0, start:end])
        plt.show()

    def test_wav_audio_data(self):
        """
        Test that the wav data property is set correctly and plot the data to see if it looks right.
        """
        url = './resources/Simple.wav'
        reader = MakeAudioReader(url)
        data = reader.ReadSamplesFloat()
        self.assertEqual(id(data), id(reader._data)) # Make sure the property is set correctly.
        start = 2 * 44100
        end = 3 * 44100
        plt.plot(data[0, start:end])
        plt.show()

    def test_mp3_audio_length(self):
        """
        Test that the correct length is retrieved from the mp3 audio, before and after reading form file.
        """
        url = './resources/Simple.wav'
        reader = MakeAudioReader(url)
        self.assertAlmostEqual(reader.audio_length, 9.137, places=3)  # Check the correct length is retrieved.
        reader.ReadSamplesFloat()
        self.assertAlmostEqual(reader.audio_length, 9.137, places=3)  # Check the correct length is retrieved.

    def test_wav_audio_length(self):
        """
        Test that the correct length is retrieved from the wav audio
        """
        url = './resources/Simple.wav'
        reader = MakeAudioReader(url)
        self.assertAlmostEqual(reader.audio_length, 9.137, places=3) # Check the correct length is retrieved.


if __name__ == '__main__':
    unittest.main()
