"""
Created 12-26-17 by Matthew C. McCallum
"""

import numpy as np
import numpy.linalg as la
import spectrum as sp


class ARKalman( object ):

    def __init__( self, increment, start_freq, observation_dim=1, transition_uncertainty=0.01, measurement_uncertainty=0.01, num_ar_coeffs=3, ac_size=6 ):
        """
        Constructor.

        Args:

        """
        # Initialise history for updating autoregressive matrix
        self._history = np.matlib.zeros( ( ac_size, 1 ), dtype='complex' )

        # Initialise AR transition matrix.
        expected_phase = np.exp(1j*increment*start_freq)
        self._ar_transition = np.matlib.zeros( ( num_ar_coeffs, num_ar_coeffs ), dtype='complex' )
        self._ar_transition[-1,:] = expected_phase/num_ar_coeffs
        if num_ar_coeffs>1:
            self._ar_transition[:-1,1:] = expected_phase*np.eye( num_ar_coeffs-1 )

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

        self._num_frames = 0

    @property
    def obs_uncertainty( self ):
        """
        """
        return self._measurement_noise[-1,-1]

    @obs_uncertainty.setter
    def obs_uncertainty( self, value ):
        """
        """
        self._measurement_noise[-1,-1] = value

    def Push( self, complex_component, frequency ):
        """
        Estimate next hidden variable.

        Args:

        """
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

        # Re-estimate AR - This could be done before or after the Kalman filtering...
        # TODO [matthew.mccallum 12.27.17]: We should only really update the Kalman coefficients if we have a high local SNR...
        #									otherwise we should really lock them up as the new AR process will be distorted by
        #									any noise correlation.
        self._history = np.roll(self._history, -1, axis=0)
        self._history[-1] = complex_component
        if False:
            # TODO [matthew.mccallum 12.28.17]: Currently re-estimating AR coefficients from data does not work, fix this.
            coeffs, _, _ = sp.aryule( self._history, self._ar_transition.shape[0], norm='unbiased' )
            self._ar_transition[-1,:] = np.fliplr( coeffs.reshape( 1, self._ar_transition.shape[0] ) )
        self._num_frames += 1

        return np.dot( self._observation, self._state )
