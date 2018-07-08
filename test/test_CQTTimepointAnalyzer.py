"""
Created once upon a time by Matthew C. McCallum
"""


# Local imports
from sigtools import WavRead
from sigtools import CQTTimepointAnalyzer

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

# Python standard library imports
import unittest


class TestCQTTimepointAnalyzer(unittest.TestCase):

    def test_harmonic_signal(self):
        """
        Plots the timepoint CQT of a simple harmonic signal for visual inspection.
        """
        wav_reader = WavRead('./resources/Simple.wav')
        signal = wav_reader.ReadSamplesFloat()
        signal = np.sum(signal, axis=0)
        samp_rate = wav_reader.fmt.samp_rate
        samples_per_octave = 12
        octaves = 8
        min_freq = 40
        times = np.linspace(0.0, 5.0, 200)
        analyzer = CQTTimepointAnalyzer(samp_rate, samples_per_octave, octaves, min_freq)
        result = analyzer.Analyze(signal, times)
        plt.imshow(np.log(result))
        plt.show()


if __name__ == '__main__':
    unittest.main()