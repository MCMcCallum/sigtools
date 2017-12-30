"""
Created 12-27-17 by Matthew C. McCallum
"""

# Local imports
from SigTools import *

# Imports from local modules
# None.

# Third party libraries
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ioff()

# Python library imports
import unittest


class TestARKalman( unittest.TestCase ):
    """
    Test cases for the autoregressive Kalman filter.
    """

    def setUp( self ):
        """
        Simple set up function, to create some signals that are useful for test cases.
        """
        # Some generic signal parameters
        sig_len_s = 3.0
        samp_rate = 44100.0
        sig_len = int( samp_rate*sig_len_s )
        win_len = 1024
        fft_size = win_len*2.0

        # Make a sinusoidal signal for predicting
        self._freq_fft_bin = int( 0.333*fft_size )
        self._freq = self._freq_fft_bin/fft_size*samp_rate
        self._freq_rad = self._freq/samp_rate*2*np.pi
        self._sinusoid = np.cos( np.arange( sig_len )*self._freq_rad )

        # Make a noise signal for uncertainty
        self._noise_var = 1.0
        self._white_noise = np.sqrt( self._noise_var )*np.random.randn( sig_len )

        self._window = np.hamming( win_len )
        self._overlap = 0.5
        self._spec_analysis = Spectrogram( self._window, int( fft_size ), self._overlap )

        self._verbose = False
        self._graph = False

    def set_reset_complex_plot( self ):
        """
        Function for initializing a plot of a few complex phasors.
        """
        plt.close('all')
        plt.figure()
        ax = plt.gca()
        ax.axhline()
        ax.axvline()
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.hold( True )

    def plot_complex( self, num, style='r:x' ):
        """
        Plot a single complex number as a phasor.

        Args:
             num -> complex - A complex number to be plotted on the complex plane.

             style -> str - A string describing the plotting style according to matplotlib conventions.
        """
        x = np.array( [0, np.real(num)] )
        y = np.array( [0, np.imag(num)] )
        plt.plot( x, y, style )

    def test_sin_plus_noise( self ):
        """
        Test case for tracking a sinusoid in noise.
        """
        # Set up test parameters
        signal = self._sinusoid + self._white_noise
        self._spec_analysis.Analyze( signal )
        prediction_sig = self._spec_analysis.spec[self._freq_fft_bin,:]
        self._spec_analysis.Analyze( self._sinusoid )
        ground_truth_sig = self._spec_analysis.spec[self._freq_fft_bin,:]
        phase_advancement = self._spec_analysis.frame_inc*self._freq_rad

        # Set up Kalman filter
        uncertainty = np.sum( np.power( self._window, 2.0 ) )*self._noise_var
        kalman_filter = ARKalman( increment=self._spec_analysis.frame_inc,
                                  start_freq=self._freq_rad,
                                  observation_dim=1,
                                  transition_uncertainty=10.0,
                                  measurement_uncertainty=uncertainty,
                                  num_ar_coeffs=3,
                                  ac_size=50,
                                  ar_reestimation=False )

        # Push in signal frame by frame and plot results if desired.
        all_results = np.zeros( len( prediction_sig ), dtype='complex' )
        for index in range( len( prediction_sig ) ):
            all_results[index] = kalman_filter.Push( prediction_sig[index], self._freq_rad ).item( ( 0, 0 ) )

            if self._graph and self._verbose:
                self.set_reset_complex_plot()
                # Plot oracle mean
                self.plot_complex( ground_truth_sig[index], 'b-o' )
                # Plot prediction based on oracle
                if index > 0:
                    self.plot_complex( ground_truth_sig[index-1]*np.exp(1j*phase_advancement), 'k--o' )
                # Plot noise signal
                self.plot_complex( prediction_sig[index], 'r:x' )
                # Plot Kalman mean
                self.plot_complex( all_results[index], 'g-x' )
                plt.axis('equal')
                plt.show()

        # Plot the tracking over time.
        if self._graph:
            # Plot magnitude adaptation
            plt.plot( np.abs( all_results[:100] ) )
            plt.hold( True )
            plt.plot( np.abs( prediction_sig[:100] ), 'r-' )
            plt.show()

        # Check test conditions
        adaptation_frames = 100 # => The number of frames where the Kalman filter is still expected to be adapting to
                                # steady state.
        sq_mean = np.mean( np.abs( all_results[adaptation_frames:] ) )**2.0
        variance = np.var( np.abs( all_results[adaptation_frames:] ) )
        SNR = 10*np.log10( sq_mean/variance )
        self.assertGreater( SNR, 35.0, 'Kalman filter output too noisy.' )
        if self._verbose:
            print( ' ' )
            print( 'Test sin plus noise complete, with SNR: {0}'.format( SNR ) )
            print( ' ' )

    def test_sine_plus_impulse( self ):
        """
        Test case for tracking a sinusoid with a big impulse in the middle.
        The impulse should have a minimal perturbation on the sinusoid magnitude tracking at least.

        TODO [matthew.mccallum 30.12.17]: A more thorough test would test phase perturbation too.
        """
        # Set up test parameters
        impulse_frame_number = 150
        signal = self._sinusoid + self._white_noise
        signal[self._spec_analysis.frame_inc*impulse_frame_number + 100] = 500.0
        self._spec_analysis.Analyze( signal )
        prediction_sig = self._spec_analysis.spec[self._freq_fft_bin, :]
        self._spec_analysis.Analyze( self._sinusoid )

        # Set up Kalman filter
        uncertainty = np.sum( np.power( self._window, 2.0 ) )*self._noise_var
        kalman_filter = ARKalman( increment=self._spec_analysis.frame_inc,
                                  start_freq=self._freq_rad,
                                  observation_dim=1,
                                  transition_uncertainty=10.0,
                                  measurement_uncertainty=uncertainty,
                                  num_ar_coeffs=3,
                                  ac_size=50,
                                  ar_reestimation=False )

        # Push in signal frame by frame and plot results if desired.
        all_results = np.zeros( len( prediction_sig ), dtype='complex' )
        for index in range( len( prediction_sig ) ):
            all_results[index] = kalman_filter.Push( prediction_sig[index], self._freq_rad ).item( ( 0, 0 ) )

        # Plot the tracking over time.
        if self._graph:
            # Plot magnitude adaptation
            plt.plot( np.abs( all_results ) )
            plt.hold( True )
            plt.plot( np.abs( prediction_sig ), 'r-' )
            plt.show()

        # Check test conditions
        adaptation_frames = 100  # => The number of frames where the Kalman filter is still expected to be adapting to
        # steady state.
        sq_mean = np.mean(np.abs(all_results[adaptation_frames:])) ** 2.0
        variance = np.var(np.abs(all_results[adaptation_frames:]))
        SNR = 10 * np.log10(sq_mean / variance)
        self.assertGreater(SNR, 30.0, 'Kalman filter output too noisy.')
        if self._verbose:
            print(' ')
            print('Test sin plus noise complete, with SNR: {0}'.format(SNR))
            print(' ')

    def test_chirp( self ):
        """
        """
        pass

    def test_sine_onset( self ):
        """
        """
        pass

    def test_sine_crossover( self ):
        """
        """
        pass


if __name__ == '__main__':
    unittest.main()