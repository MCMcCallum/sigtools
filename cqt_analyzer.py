"""
Created 06-14-18 by Matt C. McCallum
"""

# Local imports
# None.

# Third party imports
from .librosa_cqt_scipy_resample import cqt
import numpy as np

# Python standard library imports
# None.

# TODO [matthew.mccallum 06.15.18]: Can some of this code be shared with the CQTTimepointAnalyzer class?
class CQTAnalyzer(object):
    """
    This is a very simple wrapper around the librosa CQT class to simplify the interface a little, and store
    settings in the object across calls to Analyze(...).
    """

    def __init__(self, samples_per_octave, octaves, min_freq, hop, num_windows, samp_rate):
        """
        Constructor.

        Args:
            samples_per_octave: int - The number of CQT samples between each octave.

            octaves: int - The number of octaves in the output data, starting at min_freq.

            min_freq: float - Minimum frequency analyzed in the CQT in Hz.

            hop: float - The number of seconds to skip between successive CQT analysis windows.
            
            num_windows: int - The number of CQT windows to analyze following the requested analysis point.

            samp_rate: int - The sampling rate in Hz to be used for analysis - note this will affect the hop size and
            should be reconfigured for each audio sample if necessary.
        """
        self._hop = hop
        self._min_freq = min_freq
        self._octaves = octaves
        self._samples_per_octave = samples_per_octave
        self._num_windows = num_windows
        self._samp_rate = samp_rate

    def Analyze(self, audio_sig, start_idx, num_windows=None):
        """
        This function will perform a CQT analysis of signal starting at the index requested. The analysis will not
        go to the end of the signal, but output a fixed number of analysis windows following the requested index.

        Args:
            audio_sig: np.ndarray(float) - A 1D numpy array of floats, each representing an individual sample.

            start_idx: int - The index of the first window to return. A block of _num_windows will be returned
            following this first window. The window at index 0 is centered at sample 0 in audio_sig. All remaining
            windows are centered at audiosig[idx*self._hop].

            samp_rate: float - Sampling rate in Hz of the audio data that will be provided to the analyzer.

            num_windows: int - The number of windows to analyze from the provided starting index with the configured
            hop number of samples in between each.
        """
        # If no number of windows is provided, use default
        if not num_windows:
            num_windows = self._num_windows

        # TODO [matthew.mccallum 06.15.18]: Here I add on 1.0 seconds to catch the last window, and chop off any excess later. 
        # This is a little sloppy, but there is only so much thyme. I should really caclulate the additional audio required.
        audio_sig = audio_sig[int(start_idx*self.hop):int(start_idx*self.hop + self.hop*num_windows + samp_rate)] 

        # TODO [matt.c.mccallum 08.21.18]: Here we make sure the number of samples is not close to a prime number to avoid problems
        # resampling with scipy.
        if(len(audio_sig)%4):
            audio_sig = np.concatenate((audio_sig, [0.0]*(4-(len(audio_sig)%4))))

        return np.abs(cqt(audio_sig, 
                        samp_rate, 
                        self.hop, 
                        self._min_freq, 
                        self._octaves*self._samples_per_octave, 
                        self._samples_per_octave, 
                        tuning=0.0)[:,:num_windows])

    @property
    def window_rate(self):
        """
        Type: float

        Returns the effective windowing rate of the CQT anaylsis in Hz.
        """
        return 1.0/self.hop

    @property
    def hop(self):
        """
        Type: int

        The number of samples by which to hop by between each CQT analysis window.
        """
        hop_samples = self._hop*self._samp_rate
        hop_samples = int(2**(self._octaves - 1) * round(hop_samples/(2**(self._octaves - 1))))
        return hop_samples

    @property
    def samp_rate(self):
        """
        Type: int

        The sampling rate that the provided samples will be considered to be at upon analysis.
        In Hz.
        """
        return self._samp_rate
    @samp_rate.setter
    def samp_rate(self, samp_rate):
        self._samp_rate = samp_rate

    @property
    def num_windows(self):
        """
        Type: int

        Allow the number of windows to be configurable.
        This property is the maximum number of CQT windows returned by a call to Analyze.
        """
        return self._num_windows
    @num_windows.setter
    def num_windows(self, num_wins):
        self._num_windows = num_wins
        