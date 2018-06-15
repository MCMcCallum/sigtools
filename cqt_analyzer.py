"""
Created 06-14-18 by Matt C. McCallum
"""

# Local imports
# None.

# Third party imports
import librosa

# Python standard library imports
# None.

# TODO [matthew.mccallum 06.15.18]: Can some of this code be shared with the CQTTimepointAnalyzer class?
class CQTAnalyzer(object):
    """
    This is a very simple wrapper around the librosa CQT class to simplify the interface a little, and store
    settings in the object across calls to Analyze(...).
    """

    def __init__(self, samp_rate, samples_per_octave, octaves, min_freq, hop, num_windows):
        """
        Constructor.

        Args:
            samp_rate: float - Sampling rate in Hz of the audio data that will be provided to the analyzer.

            samples_per_octave: int - The number of CQT samples between each octave.

            octaves: int - The number of octaves in the output data, starting at min_freq.

            min_freq: float - Minimum frequency analyzed in the CQT in Hz.

            hop: int - The number of samples to skip between successive CQT analysis windows.
            
            num_windows: int - The number of CQT windows to analyze following the requested analysis point.
        """
        self._samp_rate = samp_rate
        self._hop = hop
        self._min_freq = min_freq
        self._octaves = octaves
        self._samples_per_octave = samples_per_octave
        self._num_windows = num_windows

    def Analyze(self, audio_sig, start_idx):
        """
        This function will perform a CQT analysis of signal starting at the index requested. The analysis will not
        go to the end of the signal, but output a fixed number of analysis windows following the requested index.

        Args:
            audio_sig: np.ndarray(float) - A 1D numpy array of floats, each representing an individual sample.

            start_idx: int - The index in audio_sig at which the first analysis window is centered. Any time before
            this index will be padded by reflecting the signal that was after it, according to Librosa.
        """
        audio_sig = audio_sig[start_idx:(start_idx + hop*self._num_windows + self._samp_rate)]
        return librosa.core.cqt(audio_sig, 
                                self._samp_rate, 
                                self._hop, 
                                self._min_freq, 
                                self._octaves*self._samples_per_octaves, 
                                self._samples_per_octave, 
                                tuning=0.0,
                                real=True)[:,:self._num_windows]
        