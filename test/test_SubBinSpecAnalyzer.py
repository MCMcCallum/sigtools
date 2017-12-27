'''
Created 12-26-17 by Matthew C. McCallum
'''

from SigTools import *

import unittest
import numpy as np



class SubBinSpecAnalyzer_TestCase( unittest.TestCase ):
	'''
	Test the accuracy of the SubBinSpecAnalyzer
	'''

	def setUp( self ):
		'''
		Some initial setup that will be useful for most of the tests.
		'''
		self.verbose = False

		# Signal parameters
		self.freq = 4095.0
		self.phase = 0
		self.amplitude = 1.8
		self.samp_rate = 44100.0
		self.sig_len = int( 5*self.samp_rate )
		self.sinusoid = self.amplitude*np.cos( np.arange( self.sig_len )*self.freq/self.samp_rate*2.0*np.pi + self.phase )

		# STFT parameters
		win_len_secs = 0.02
		win_len = int( win_len_secs*self.samp_rate )
		self.overlap = 1.0 - 1.0/2.0/2.0/2.0/2.0
		self.window = np.ones( win_len )
		self.fft_size = int( win_len*4 )
		self.spec_analyzer = Spectrogram( self.window, self.fft_size, self.overlap )
		self.spec_analyzer.Analyze( self.sinusoid )

		# The analyzer to test
		resolution_mult = 100
		self.sub_bin_analyzer = SubBinSpecAnalyzer( self.fft_size, self.window, resolution_mult )

	def test_magnitude_analysis( self ):
		'''
		Check the magnitude result of the SubBinSpecAnalyzer for a single sinusoid at it's positive
		frequency peak.
		'''
		# Parameters for where to test the magnitude
		sub_bin = self.freq/self.samp_rate*self.fft_size
		frame = 10
		bin = round( sub_bin )
		offset = sub_bin - bin

		# Calculate the analysis and expected result.
		estimated_mag = self.sub_bin_analyzer.GetMag( np.array([np.abs( self.spec_analyzer.spec[bin, frame] )]), np.array([offset]) )
		expected_mag = self.amplitude*np.sum( self.window )/2.0

		# TODO [matthew.mccallum 12.22.17]: I should be able to calculate the allowable accuracy below
		#									due to the sideband interference of the negative frequency
		#									sinusoid component.

		self.assertLess( abs( estimated_mag - expected_mag ), 1.0 ) 

		if self.verbose:
			print( ' ' )
			print( 'OFFSET WAS: {0}'.format(offset) )
			print( 'NEAREST MAG: {0}'.format(np.array([np.abs( self.spec_analyzer.spec[bin, frame] )])) )
			print( 'MAG: {0} vs {1}'.format(estimated_mag, expected_mag) )

	def test_phase_analysis( self ):
		'''
		Check the phase result of the SubBinSpecAnalyzer for a single sinusoid at it's positive
		frequency peak.
		'''
		# Parameters for where to test the phase
		sub_bin = self.freq/self.samp_rate*self.fft_size
		frame = 10
		bin = round( sub_bin )
		offset = sub_bin - bin
		resolution_mult = 100

		# Calculate the analysis and expected result.
		estimated_phase = self.sub_bin_analyzer.GetPhase( np.array([np.angle( self.spec_analyzer.spec[bin, frame] )]), np.array([offset]) )
		expected_phase = ( frame*self.spec_analyzer.frame_inc*self.freq/self.samp_rate*2*np.pi )%( 2*np.pi )

		# TODO [matthew.mccallum 12.22.17]: I should be able to calculate the allowable accuracy below
		#									due to the sideband interference of the negative frequency
		#									sinusoid component.

		phase_diff = abs( estimated_phase - expected_phase )
		self.assertLess( min( min( phase_diff, abs( 2*np.pi-phase_diff ) ), phase_diff+2*np.pi ), 0.001 ) 

		if self.verbose:
			print( ' ' )
			print( 'OFFSET WAS: {0}'.format(offset) )
			print( 'NEAREST MAG: {0}'.format(np.array([np.abs( self.spec_analyzer.spec[bin, frame] )])) )
			print( 'PHASE: {0} vs {1}'.format(estimated_phase, expected_phase) )



if __name__=='__main__':
	unittest.main()