'''
Created 12-27-17 by Matthew C. McCallum
'''

from SigTools import *

import unittest
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


class TestARKalman( unittest.TestCase ):
    '''

    '''

    def setUp( self ):
        '''
        '''
        # Make a signal for predicting
        sig_len_s = 3.0
        samp_rate = 44100.0
        sig_len = int( samp_rate*sig_len_s )
        win_len = 1024
        fft_size = win_len*2.0
        self._freq_fft_bin = int( 0.333*fft_size )
        self._freq = self._freq_fft_bin/fft_size*samp_rate
        self._freq_rad = self._freq/samp_rate*2*np.pi
        self._sinusoid = np.cos( np.arange( sig_len )*self._freq_rad )
        self._noise_var = 1.0
        self._white_noise = np.sqrt( self._noise_var )*np.random.randn( sig_len )

        self._window = np.hamming( win_len )
        self._overlap = 0.5
        self._spec_analysis = Spectrogram( self._window, int( fft_size ), self._overlap )

        self._verbose = False
        self._graph = False

    def set_reset_complex_plot( self ):
        '''
        '''
        plt.figure()
        ax = plt.gca()
        ax.axhline()
        ax.axvline()
        plt.hold( True )

    def plot_complex( self, num, style='r:x' ):
        '''
        '''
        x = np.array( [0, np.real(num)] )
        y = np.array( [0, np.image(num)] )
        plt.plot( x, y, style )

    def test_sin_plus_noise( self ):
        '''
        '''
        signal = self._sinusoid + self._white_noise
        self._spec_analysis.Analyze( signal )
        prediction_sig = self._spec_analysis.spec[self._freq_fft_bin,:]
        self._spec_analysis.Analyze( self._sinusoid )
        ground_truth_sig = self._spec_analysis.spec[self._freq_fft_bin,:]
        phase_advancement = self._spec_analysis.frame_inc*self._freq_rad

        uncertainty = np.sum( np.power( self._window, 2.0 ) )*self._noise_var
        kalman_filter = ARKalman( increment=self._spec_analysis.frame_inc,
                                  start_freq=self._freq_rad,
                                  observation_dim=1,
                                  transition_uncertainty=1.0,
                                  measurement_uncertainty=uncertainty,
                                  num_ar_coeffs=3,
                                  ac_size=6 )

        all_results = np.zeros( len( prediction_sig ), dtype='complex' )
        for index in range( len( prediction_sig ) ):
            all_results[index] = kalman_filter.Push( prediction_sig[index], self._freq_rad ).item( ( 0, 0 ) )
            if index>200:
                plt.plot( np.abs( all_results[:200] ) )
                plt.hold( True )
                plt.plot( np.abs( prediction_sig[:200] ), 'r-' )
                plt.show()

            if self._graph:
                self.set_reset_complex_plot()
                # Plot oracle mean
                self.plot_complex( ground_truth_sig[index], 'b--o' )
                # Plot prediction based on oracle
                if index > 0:
                    self.plot_complex( ground_truth_sig[index-1], 'k--o' )
                # Plot noise signal
                self.plot_complex( prediction_sig[index], 'r:x' )
                # Plot Kalman mean
                self.plot_complex( result, 'g-x' )

        if self._graph:
            pass
            # Plot spectrogram
            # Plot harmonic component
            # Plot noise component

    def test_sine_plus_impulse( self ):
        '''
        '''
        pass

    def test_chirp( self ):
        '''
        '''
        pass

    def test_sine_onset( self ):
        '''
        '''
        pass

    def test_sine_crossover( self ):
        '''
        '''
        pass


if __name__ == '__main__':
    unittest.main()