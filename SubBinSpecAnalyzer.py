'''
Created 12-26-17 by Matthew C. McCallum
'''

import numpy as np

class SubBinSpecAnalyzer( object ):
	'''
	An analyzer for re-estimating magnitude and phase between integer bin indices in
	a spectrogram.
	'''

	def __init__( self, fft_size, window, resolution_mult ):
		'''
		Constructor.

		Args:
			fft_size -> int - The FFT size used for analysis of any magnitudes and phases
			this class wishes to re-estimate.

			window -> np.ndarray - A 1D array containing the STFT analysis window used for
			any magnitudes or phases this class wishes to re-estimate.

			resolution_mult -> int - The number of intervals between each STFT bin that this
			class uses as a lookup table to re-estimate magnitudes and phases. The higher this
			number is the more accurate it is to a point.
		'''
		sub_bin_spec = np.fft.fft( window, fft_size*resolution_mult );
		self._subbin_resolution = resolution_mult
		self._subbin_mag = np.abs( sub_bin_spec )
		self._subbin_mag = np.concatenate( ( self._subbin_mag[:resolution_mult], self._subbin_mag[-resolution_mult:] ) )
		self._subbin_phase = np.angle( sub_bin_spec )
		self._subbin_phase = np.concatenate( ( self._subbin_phase[:resolution_mult], self._subbin_phase[-resolution_mult:] ) )
		self._win_energy = np.sum( window )

	def GetMag( self, bin_mags, bin_offsets ):
		'''
		Re-estimate a set of magnitudes at a given non-integer offset from their original bin indices.

		Args:
			bin_mags - np.ndarray - An array of the bin magnitudes at integer DFT indices that are representative 
			of a peak at a potentially non-integer index.

			bin_offsets - np.ndarray - An array of float offsets to re-estimate the magnitude of each of the bins provided
			in bin_mags at.
		'''
		assert all( np.abs( bin_offsets ) < 1.0*( self._subbin_resolution-1 )/self._subbin_resolution ), 'Tried to get subbin accuracy outside of the range of the SubBinSpecAnalyzer.'
		indices = np.around( bin_offsets*self._subbin_resolution ).astype( 'int32' )
		return bin_mags/self._subbin_mag[indices]*self._win_energy

	def GetPhase( self, bin_phases, bin_offsets ):
		'''
		Re-estimate a set of phases at a given non-integer offset from their original bin indices.

		Args:
			bin_mags - np.ndarray - An array of the bin phases at integer DFT indices that are representative 
			of a peak at a potentially non-integer index.

			bin_offsets - np.ndarray - An array of float offsets to re-estimate the phase of each of the bins provided
			in bin_mags at.
		'''
		assert all( np.abs( bin_offsets ) < 1.0*( self._subbin_resolution-1 )/self._subbin_resolution ), 'Tried to get subbin accuracy outside of the range of the SubBinSpecAnalyzer.'
		indices = np.around( bin_offsets*self._subbin_resolution ).astype( 'int32' )
		return bin_phases + self._subbin_phase[indices]
