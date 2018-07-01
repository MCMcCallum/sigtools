"""
Created 06-10-18 Matt C. McCallum

This is a library of functions that provide reusable common signal processing type operations.
Usually these should be integrated into the class or data type that uses them, although, some
data types such as numpy arrays are not suited to subclassing perhaps due to overhead, and so
general purpose functions for these types of objects may be found here.
"""


# Local imports
# None.

# Third party imports
import numpy as np

# Python standard library imports
# None.


def log_scale(data, dynamic_range):
    """
    Scales data logarithmically clipping all values a certain number of decibels below the maximum.

    Args:
        data -> np.ndarray - A numpy array of data of any shape, with real valued elements that are to
        be scaled logarithmically. These values should not be power values but linear (i.e., not squared).

        dynamic_range -> float - A number of decibels below the maximum for which values will be maintained.
        Any value below this dynamic range will be floored at: floor = maximum - dynamic_range.
    """
    data[data<0.00001] = 0.00001
    data = 20*np.log10(data)
    max_val = np.amax(data)
    min_val = max_val - dynamic_range
    inds = np.where(np.isnan(data))
    data[inds] = min_val
    data[data<min_val] = min_val
    return data


def peaks(signal):
    """
    Returns the indices of all peaks in a provided 1D signal. Currently if there is a plateau,
    This function will return the index of the first value of the plateau only.

    Args:
        signal -> np.ndarray - A 1D numpy array containing the signal to find peak indices for.

    Return:
        np.ndarray - A 1D numpy array containing the indices of every peak in the signal.
    """
    # TODO [matthew.mccallum 06.30.18]: Add options here for plateaus here - the user should be able to
    # choose to return the first value of a plateau, last value of a plateau, neither, or both.
    assert(signal.ndim == 1) # <= This function is limited to analyzing 1D signals for now.
    signal = np.concatenate((np.array([0.0]), signal, np.array([0.0])))
    diff_sig = signal[1:] - signal[:-1]
    peaks = diff_sig[1:]*diff_sig[:-1]
    peaks = peaks <= 0
    peaks = np.logical_and(peaks, (signal[1:-1] > signal[:-2]))
    peak_idcs = np.nonzero(peaks)[0]
    return peak_idcs
    

def time_string(seconds):
    """
    Returns a string that is a track-time format representation of the provided seconds value.

    Args:
        seconds -> float - A number of seconds to convert into minutes, seconds and milliseconds

    Return:
        str - A string in track-time format, e.g., <minutes>:<seconds>:<milliseconds>
    """
    minutes = int(seconds/60.0)
    seconds = seconds - minutes*60
    whole_secs = int(seconds)
    frac_secs = int((seconds - whole_secs)*100)
    return str(minutes) + ":" + str('{:02d}').format(whole_secs) + ":" + str('{:02d}').format(frac_secs)
