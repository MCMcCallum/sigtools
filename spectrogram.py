"""
Created 12-26-17 by Matthew C. McCallum
"""

# Local modules
# None.

# Thirdparty modules
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ioff()

# Python library modules
import math
import copy
from functools import singledispatch


class Spectrogram( object ):
    """
    Class for analyzing the spectrogram of a signal.
    """

    def __init__( self, window, fft_size, overlap, num_frames=0 ):
        """
        Constructor.

        Args:
            window -> np.ndarray - A 1D array containing the windowing function for
            STFT analysis.

            fft_size -> int - The number of DFT bins to analyze.

            overlap -> float - A percentage overlap between successive frames.
        """
        self._window = window
        self._fft_size = fft_size
        self._overlap = overlap
        self._win_len = len( self._window )
        self._frame_inc = int( ( 1 - self._overlap )*self._win_len )
        self._num_frames = num_frames
        self._spec = np.zeros( (fft_size, num_frames), dtype='complex' )

    def Analyze( self, signal ):
        """
        Analyze a given signal from start to finish. The first index of the first analysis
        window will align with the first sample of signal, and the final window will be the
        last complete analysis window that fits entirely within the length of signal for
        the current windowing size / overlap configuration.

        Args:
            signal -> np.ndarray - A 1D array containing the signal to be analyzed.
        """
        self._num_frames = math.floor( ( len( signal ) - self._win_len )/self._frame_inc ) + 1
        frame_indices = np.arange( self._num_frames, dtype='int32' )
        freq_indices = np.arange( self._win_len, dtype='int32' )
        spec_indices = np.add( *np.meshgrid( frame_indices*self._frame_inc, freq_indices ) )
        self._spec = np.fft.fft( np.dot( np.diag( self._window ), signal[spec_indices] ), self._fft_size, axis=0 )

    def Synthesise( self ):
        """
        Synthesise the spectrogram as a 1D real signal using the overlap-add method with no synthesis windowing.

        Return:
            np.ndarray 1D - The synthesised signal using overlap-add.
        """
        output_sig = np.zeros( ( 1, self._num_frames*self._frame_inc + self._win_len ) )
        time_windows = np.fft.ifft( self._spec, axis=0 )
        time_windows = time_windows[:self._win_len,:]
        time_windows = np.real( time_windows )
        for win_num in range( self._num_frames ):
            output_sig[0,(win_num*self._frame_inc):(win_num*self._frame_inc+self._win_len)] += time_windows[:,win_num]
        return output_sig

    @property
    def spec( self ):
        """
        np.ndarray - A 2D array containing the complex spectrogram of the last input signal to be analyzed.
        """
        return self._spec

    @property
    def frame_inc( self ):
        """
        int - The number of samples elapsed between the start of each consecutive window in analysis.
        """
        return self._frame_inc

    def __getitem__( self, item ):
        """
        Item getter for getting spectrogram slices.

        Return:
            Spectrogram - The spectrogram containing a view on the requested subset of data.
        """
        new_spec = copy.copy( self )
        new_spec._spec = self._spec[:,item]
        return new_spec

    def __setitem__( self, idx, value ):
        """
        Item setter for setting spectrogram subsets.

        Args:
            idx - int, slice, range - The frames which to update.

            value - Spectrogram or np.ndarray - The data to update the spectrogram with.
        """
        # TODO [matthew.mccallum 01.07.17]: This conditional below is a bit slow for simply setting an item, but since
        # functools.singledispatch only helps with type inference on the first argument, we're stuck with it for now.
        if isinstance( value, np.ndarray ):
            self._spec[:,idx] = value
        elif isinstance( value, Spectrogram ):
            self._spec[:,idx] = value.spec
        else:
            raise TypeError

    def __add__( self, other ):
        """
        Add this spectrogram data to another via complex addition.

        Args:
            other -> Spectrogram - A spectrogram of equal size containing values to be added to this spectrogram.

        Return:
            Spectrogram - A new spectrogram object containing the resulting data.
        """
        new_spec = copy.copy( self )
        new_spec._spec = self._spec + other.spec
        return new_spec

    def __sub__( self, other ):
        """
        Subtract another spectrogram from this one via complex subtraction.

        Args:
            other -> Spectrogram - A spectrogram of equal size containing values to be subtracted from this spectrogram.

        Return:
            Spectrogram - A new spectrogram object containing the resulting data.
        """
        new_spec = copy.copy( self )
        new_spec._spec = self._spec - other.spec
        return new_spec

    def __len__( self ):
        """
        Get the number of frames in this spectrogram.

        Return:
            int - The number of frames in this spectrogram.
        """
        return self._spec.shape[1]

    def Plot( self ):
        """
        Plots the current spectrogram and displays it on screen.
        """
        plt.imshow( 10*np.log10( np.abs( self._spec ) ) )
        plt.show()

