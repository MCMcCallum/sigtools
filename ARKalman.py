"""
Created 12-26-17 by Matthew C. McCallum
"""

# Local imports
# None.

# Imports from local modules
# None.

# Thirdparty imports
import numpy as np
import numpy.linalg as la
import spectrum as sp

# Python library imports
# None.

class ARKalman( object ):
    """
    A Kalman filter specifically designed for autoregressive (AR) tracking of frequency domain sinusoid peaks over time.
    """

    def __init__( self, increment, start_freq, observation_dim=1, transition_uncertainty=0.01, measurement_uncertainty=0.01, num_ar_coeffs=3, ac_size=6, ar_reestimation=True ):
        """
        Constructor.

        Args:
            increment -> int - The number of samples elapsed between each observation pushed into the Kalman filter

            start_freq -> float - The frequency to initialise the AR prediction tracking to. This adjusts the phase of
            current and past estimations to predict the current frame.

            observation -> int - The dimensionality of each observation pushed into the Kalman filter.

            transition_uncertainty -> float - The variance of each successive observation from the prediction provided
            by the AR tracking

            measurement_uncertainty -> float - The variance of each observation pushed into the Kalman filter about its
            predictable component.

            num_ar_coeffs -> int - The number of autoregressive coefficients used to predict each frame from past
            predictions.

            ac_size -> int - Number of past observations used in re-estimating the autocorrelation coefficients with
            newly pushed data if enabled.

            ar_reestimation -> bool - Whether to re-estimate the autocorrelation coefficients used for the Kalman
            prediction step each time new data is pushed into the filter.
        """

        # Initialise history for updating autoregressive matrix
        self._history = np.matlib.zeros( ( ac_size, 1 ), dtype='complex' )
        self._num_frames = 0

        # Initialise AR transition matrix.
        self._increment = increment
        expected_phase = np.exp(1j*self._increment*start_freq)
        self._ar_transition = np.matlib.zeros( ( num_ar_coeffs, num_ar_coeffs ), dtype='complex' )
        self._ar_reestimation = ar_reestimation
        self._update_transition( expected_phase )

        # Initialise observation matrix.
        self._observation = np.matlib.zeros( ( observation_dim, num_ar_coeffs ), dtype='complex' )
        self._observation[-1,-1] = 1.0

        # Initialise system and measurement noise.
        self._system_noise = np.matlib.eye( num_ar_coeffs )*transition_uncertainty
        self._predictive_var = np.matlib.zeros( self._system_noise.shape )
        self._measurement_noise = np.matlib.zeros( ( observation_dim, observation_dim ) )
        self._measurement_noise[-1,-1] = measurement_uncertainty

        # Set up internal state.
        self._state = np.matlib.zeros( ( num_ar_coeffs, 1 ), dtype='complex' )

    @property
    def obs_uncertainty( self ):
        """
        The uncertainty of observations pushed into the Kalman filter.
        This could be updated dynamically based on estimation of noise in each of the magnitude/phase observations.

        TODO [matthew.mccallum 12.30.17]: This is currently limited to 1D observations, while parts of this class
        operate on higher dimensional observations. This functionality should be expanded to be consistent.
        """
        return self._measurement_noise[-1,-1]

    @obs_uncertainty.setter
    def obs_uncertainty( self, value ):
        """
        Set the observation uncertainty.

        TODO [matthew.mccallum 12.30.17]: This is currently limited to 1D observations, expand to the general case.
        """
        self._measurement_noise[-1,-1] = value

    def Push( self, complex_component, frequency ):
        """
        Estimate next hidden variable.

        Args:
            complex_component -> complex - A complex observation to be filtered by the Kalman filter.

            frequency -> float - The frequency this complex observation was observed at in radians per sample.

        Return:
            complex - The latest estimated hidden variable transformed into the observation domain.
        """

        # Update history for any updates on the transition matrix estimation.
        # TODO [matthew.mccallum 12.27.17]: We should only really update the Kalman coefficients if we have a high local SNR...
        #									otherwise we should really lock them up as the new AR process will be distorted by
        #									any noise correlation.
        expected_phase = np.exp( 1j*self._increment*frequency )
        self._history = np.roll(self._history, -1, axis=0)*expected_phase
        self._history[-1] = complex_component
        self._update_transition( expected_phase )

        # Predict state and variance
        self._state = np.dot( self._ar_transition, self._state )
        self._predictive_var += self._system_noise

        # Kalman gain
        temp = la.inv( np.dot( np.dot( self._observation, self._predictive_var ), self._observation.H ) + self._measurement_noise )
        kalman_gain = np.dot( np.dot( self._predictive_var, self._observation.H ), temp )

        # Correct prediction amd variance
        observation = np.matlib.zeros( ( self._observation.shape[0], 1 ), dtype='complex' )
        observation[-1] = complex_component
        self._state = self._state + np.dot( kalman_gain, observation - np.dot( self._observation, self._state ) )
        self._predictive_var = np.dot( np.eye( self._predictive_var.shape[0] ) - np.dot( kalman_gain, self._observation ), self._predictive_var )

        return np.dot(self._observation, self._state)

    def _update_transition( self, expected_phase ):
        """
        Update the AR transition matrix.

        Args:
            expected_phase -> complex - A phasor that rotates a complex variable on multiplication by the amount that
            successive observations are expected to rotate by.
        """

        # If enough history, then update the AR coefficeints if requested.
        if self._ar_reestimation and ( self._num_frames > len( self._history ) ):
            coeffs, _, _ = sp.aryule( self._history, self._ar_transition.shape[0] )
            self._ar_transition[-1,:] = np.fliplr( -coeffs.reshape( 1, self._ar_transition.shape[0] ) )*expected_phase
        # Otherwise just update according to the most recent frequency
        else:
            self._ar_transition[-1, :] = expected_phase / self._ar_transition.shape[1]
            if self._ar_transition.shape[1] > 1:
                self._ar_transition[:-1, 1:] = expected_phase * np.eye( self._ar_transition.shape[1] - 1 )
            self._num_frames += 1
