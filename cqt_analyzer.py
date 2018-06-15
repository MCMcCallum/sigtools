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

    def __init__(self, samples_per_octave, octaves, min_freq, hop, num_windows):
        """
        Constructor.

        Args:
            samples_per_octave: int - The number of CQT samples between each octave.

            octaves: int - The number of octaves in the output data, starting at min_freq.

            min_freq: float - Minimum frequency analyzed in the CQT in Hz.

            hop: float - The number of seconds to skip between successive CQT analysis windows.
            
            num_windows: int - The number of CQT windows to analyze following the requested analysis point.
        """
        self._hop = hop
        self._min_freq = min_freq
        self._octaves = octaves
        self._samples_per_octave = samples_per_octave
        self._num_windows = num_windows

    def Analyze(self, audio_sig, start_idx, samp_rate):
        """
        This function will perform a CQT analysis of signal starting at the index requested. The analysis will not
        go to the end of the signal, but output a fixed number of analysis windows following the requested index.

        Args:
            audio_sig: np.ndarray(float) - A 1D numpy array of floats, each representing an individual sample.

            start_idx: int - The index of the first window to return. A block of _num_windows will be returned
            following this first window. The window at index 0 is centered at sample 0 in audio_sig. All remaining
            windows are centered at audiosig[idx*self._hop].

            samp_rate: float - Sampling rate in Hz of the audio data that will be provided to the analyzer.
        """
        hop_samples = self._hop*samp_rate
        # TODO [matthew.mccallum 06.15.18]: Here I add on 1.0 seconds to catch the last window, and chop off any excess later. 
        # This is a little sloppy, but there is only so much thyme. I should really caclulate the additional audio required.
        audio_sig = audio_sig[start_idx*hop_samples:(start_idx*hop_samples + hop_samples*self._num_windows + self._samp_rate)] 
        return librosa.core.cqt(audio_sig, 
                                samp_rate, 
                                hop_samples, 
                                self._min_freq, 
                                self._octaves*self._samples_per_octaves, 
                                self._samples_per_octave, 
                                tuning=0.0,
                                real=True)[:,:self._num_windows]

    @property
    def window_rate(self):
        """
        Returns the effective windowing rate of the CQT anaylsis in Hz.
        """
        return 1.0/self._hop
        