'''
Created 12-26-17 by Matthew C. McCallum
'''

import numpy as np
import math

class Spectrogram( object ):
	'''
	Class for analyzing the spectrogram of a signal.
	'''

	def __init__( self, window, fft_size, overlap ):
		'''
		Constructor.

		Args:
			window -> np.ndarray - A 1D array containing the windowing function for
			STFT analysis.

			fft_size -> int - The number of DFT bins to analyze.

			overlap -> float - A percentage overlap between successive frames.
		'''
		self._window = window
		self._fft_size = fft_size
		self._overlap = overlap
		self._win_len = len( self._window )
		self._frame_inc = int( ( 1 - self._overlap )*self._win_len )
		self._num_frames = 0
		self._spec = np.array([])

	def Analyze( self, signal ):
		'''
		Analyze a given signal from start to finish. The first index of the first analysis
		window will align with the first sample of signal, and the final window will be the
		last complete analysis window that fits entirely within the length of signal for
		the current windowing size / overlap configuration.

		Args:
			signal -> np.ndarray - A 1D array containing the signal to be analyzed.
		'''
		self._num_frames = math.floor( ( len( signal ) - self._win_len )/self._frame_inc ) + 1
		frame_indices = np.arange( self._num_frames, dtype='int32' )
		freq_indices = np.arange( self._win_len, dtype='int32' )
		spec_indices = np.add( *np.meshgrid( frame_indices*self._frame_inc, freq_indices ) )
		self._spec = np.fft.fft( np.dot( np.diag( self._window ), signal[spec_indices] ), self._fft_size, axis=0 )

	@property
	def spec( self ):
		'''
		np.ndarray - A 2D array containing the complex spectrogram of the last input signal to be analyzed.
		'''
		return self._spec

	@property
	def frame_inc( self ):
		'''
		int - The number of samples elapsed between the start of each consecutive window in analysis.
		'''
		return self._frame_inc