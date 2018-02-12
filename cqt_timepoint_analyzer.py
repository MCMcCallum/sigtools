
# Third party imports
import librosa.filters
import scipy.fftpack as fft
import numpy as np


class CQTTimepointAnalyzer( object ):
    """
    Analyzes the pseudo CQT of a signal at arbitrary time points.
    """

    def __init__(self, samp_rate, samples_per_octave, octaves, min_freq):
        """
        Constructor.

        Args:
            samp_rate: float - Sampling rate in Hz of the audio data that will be provided to the analyzer.

            samples_per_octave: int - The number of CQT samples between each octave.

            octaves: int - The number of octaves in the output data, starting at min_freq.

            min_freq: float - Minimum frequency analyzed in the CQT in Hz.

        """
        self._samples_per_octave = samples_per_octave
        self._octaves = octaves
        self._samp_rate = samp_rate
        self._fmin = min_freq
        self._sparsity = 0.05  # percentage of energy that can be discarded from each filter kernel
        self._window_size = 0
        self._n_fft = 0
        self._basis = np.array([[]])

        self._initKernel()

    def _initKernel(self):
        """
        Initializes the CQT kernel according to the current set of object parameters.
        Specifically these values have an effect:
            - self._samp_rate
            - self._fmin
            - self._sparsity
            - self._octaves
            - self._samples_per_octave

        """
        # Create time domain basis for cqt
        basis, basis_lengths = librosa.filters.constant_q(self._samp_rate,
                                                          fmin=self._fmin,
                                                          n_bins=self._octaves*self._samples_per_octave,
                                                          bins_per_octave=self._samples_per_octave)
        # Filters are padded up to the nearest integral power of 2
        self._n_fft = basis.shape[1]
        self._window_size = self._n_fft//2
        # re-normalize bases with respect to the FFT window length
        basis *= basis_lengths[:, np.newaxis] / float(self._n_fft)
        # FFT and retain only the non-negative frequencies
        self._basis = fft.fft(basis, n=self._n_fft, axis=1)[:, :(self._n_fft // 2) + 1]
        # sparsify the basis
        self._basis = np.abs(librosa.util.sparsify_rows(self._basis,
                                                        quantile=self._sparsity))
        # Get filter lengths for normalization
        self._filt_lengths = librosa.filters.constant_q_lengths(self._samp_rate,
                                                                self._fmin,
                                                                n_bins=self._octaves*self._samples_per_octave)

    @property
    def samp_rate(self):
        """
        float - The sampling rate of the signal to be analyzed by this object, in Hz.
        """
        return self._samp_rate
    @samp_rate.setter
    def samp_rate(self, value):
        self._samp_rate = value
        self._initKernel()

    def Analyze(self, signal, time_points):
        """
        Analyzes the CQT of a signal at windows centered at the provided time points.

        Args:
            signal: np.ndarray(float) - A 1D array containing the time-domain signal starting at time 0 seconds.

            time_points: np.ndarray(float) - A 1D array containg the times in seconds at which to obtain CQTs.

        Return:
            np.ndarray(float) - A pseudo CQT of the signal at the time points provided with pitch along the first
            dimension and time along the second dimension.

        """
        # Note: By padding the signal below, we effectively shift the time of each value in time_points backward by the
        # length we pad it by. This is exactly what we want, as it will cause the windows to be centered on each
        # time point.
        signal = np.pad(signal, pad_width=self._window_size//2, mode='constant', constant_values=0.0)
        time_inds = [int(point*self._samp_rate) for point in time_points]
        windows = [signal[start:(start+self._window_size)].reshape((self._window_size,1)) for start in time_inds]
        windows = np.hstack(windows)

        # Analyze signal
        spec = np.abs(fft.fft(windows, n=self._n_fft, axis=0)[:(self._n_fft // 2) + 1, :])
        cqt = self._basis.dot(spec)
        cqt *= np.sqrt(self._filt_lengths[:, np.newaxis] / self._n_fft)

        return cqt
