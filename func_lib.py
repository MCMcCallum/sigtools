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
